#!/usr/bin/env python3
"""Run Qwen3-VL LoRA grounding inference and build a competition submission."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm


DEFAULT_ROOT = Path("/root/autodl-tmp")
BBOX_RE = re.compile(
    r'["\']?bbox_2d["\']?\s*[:=]\s*\[\s*'
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
    re.IGNORECASE,
)
ARRAY_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Qwen3-VL LoRA checkpoint or infer the official test set."
    )
    parser.add_argument("--mode", choices=("val", "test"), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--val-json", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--test-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--depth-min-mm", type=float, default=300.0)
    parser.add_argument("--depth-max-mm", type=float, default=20000.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N records. Zero means all records.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore an existing progress file and start a new run.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def make_prompt(query: str) -> str:
    return (
        "<image>\nImage 1 is the visible RGB image.\n"
        "<image>\nImage 2 is the aligned thermal infrared image.\n"
        "<image>\nImage 3 is the aligned depth map; brighter valid pixels are closer "
        "and black pixels may be invalid.\n"
        f"Locate the following target in Image 1: {query.strip()}\n"
        'Return only JSON in this format: {"bbox_2d":[x1,y1,x2,y2]}. '
        "Coordinates must be integers from 0 to 1000."
    )


def parse_bbox_1000(text: str) -> tuple[list[float], bool]:
    match = BBOX_RE.search(text) or ARRAY_RE.search(text)
    if match is None:
        return [0.0, 0.0, 1000.0, 1000.0], False

    values = [float(value) for value in match.groups()]
    if not all(math.isfinite(value) for value in values):
        return [0.0, 0.0, 1000.0, 1000.0], False

    # The model was trained on 0-1000 coordinates. This also tolerates a model
    # that unexpectedly emits already-normalized coordinates.
    if max(abs(value) for value in values) <= 1.000001:
        values = [value * 1000.0 for value in values]

    x1, x2 = sorted((values[0], values[2]))
    y1, y2 = sorted((values[1], values[3]))
    x1, y1, x2, y2 = [min(1000.0, max(0.0, value)) for value in (x1, y1, x2, y2)]
    if x2 - x1 < 1.0:
        x2 = min(1000.0, x1 + 1.0)
        x1 = min(x1, x2 - 1.0)
    if y2 - y1 < 1.0:
        y2 = min(1000.0, y1 + 1.0)
        y1 = min(y1, y2 - 1.0)
    return [x1, y1, x2, y2], x1 < x2 and y1 < y2


def normalized_bbox(box_1000: list[float]) -> list[float]:
    box = [round(min(1.0, max(0.0, value / 1000.0)), 6) for value in box_1000]
    if box[2] <= box[0]:
        box[2] = min(1.0, round(box[0] + 0.001, 6))
    if box[3] <= box[1]:
        box[3] = min(1.0, round(box[1] + 0.001, 6))
    return box


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def convert_depth(source: Path, destination: Path, minimum_mm: float, maximum_mm: float) -> Path:
    if destination.is_file():
        return destination
    if maximum_mm <= minimum_mm:
        raise ValueError("--depth-max-mm must be larger than --depth-min-mm")

    with Image.open(source) as image:
        depth = np.asarray(image)
    if depth.ndim == 3 and depth.shape[2] in (3, 4):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image.convert("RGB").save(destination, compress_level=4)
        return destination
    if depth.ndim != 2:
        raise ValueError(f"Expected a single-channel depth map: {source}")

    depth = depth.astype(np.float32, copy=False)
    valid = depth > 0
    visual = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth[valid], minimum_mm, maximum_mm)
        scaled = 1.0 + (maximum_mm - clipped) * 254.0 / (maximum_mm - minimum_mm)
        visual[valid] = np.rint(scaled).astype(np.uint8)
    rgb = np.repeat(visual[:, :, None], 3, axis=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(destination, compress_level=4)
    return destination


def resolve_input_path(test_root: Path, json_path: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Invalid image path in JSON: {raw!r}")
    relative = Path(raw.strip().replace("\\", "/"))
    candidates = (
        [relative]
        if relative.is_absolute()
        else [test_root / relative, json_path.parent.parent / relative, json_path.parent / relative]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Image referenced by JSON was not found: {raw!r}")


def find_test_json(test_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return require_file(explicit, "Test JSON")
    preferred = test_root / "queries" / "queries.json"
    if preferred.is_file():
        return preferred.resolve()
    candidates = sorted(test_root.rglob("*.json"))
    if len(candidates) != 1:
        listed = "\n".join(f"  {path}" for path in candidates[:20]) or "  none"
        raise RuntimeError(f"Could not select one test JSON. Candidates:\n{listed}\nUse --test-json.")
    return candidates[0].resolve()


def safe_depth_destination(depth_dir: Path, source: Path) -> Path:
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    return depth_dir / f"{source.stem}_{digest}.png"


def load_swift_engine(model: Path, adapter: Path, batch_size: int):
    try:
        from swift import InferRequest, RequestConfig, TransformersEngine
    except ImportError:
        from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

    print("Loading Qwen3-VL-8B base model and LoRA adapter...")
    engine = TransformersEngine(
        str(model),
        adapters=[str(adapter)],
        model_type="qwen3_vl",
        torch_dtype=None,
        max_batch_size=batch_size,
    )
    return engine, InferRequest, RequestConfig


def run_engine_batch(
    engine: Any,
    infer_request_class: Any,
    request_config: Any,
    records: list[dict[str, Any]],
) -> list[str]:
    requests = [
        infer_request_class(
            messages=[{"role": "user", "content": record["prompt"]}],
            images=[str(path) for path in record["images"]],
        )
        for record in records
    ]
    responses = engine.infer(requests, request_config=request_config)
    return [response.choices[0].message.content or "" for response in responses]


def save_preview(records: list[dict[str, Any]], output: Path, title: str) -> None:
    records = records[:9]
    if not records:
        return
    tile_w, tile_h = 480, 430
    canvas = Image.new("RGB", (tile_w * 3, tile_h * 3), "white")
    font = ImageFont.load_default()
    for index, record in enumerate(records):
        with Image.open(record["images"][0]) as source:
            image = source.convert("RGB")
        image.thumbnail((tile_w, 340))
        draw = ImageDraw.Draw(image)
        width, height = image.size
        px = record["bbox_1000"]
        xy = [px[0] / 1000 * width, px[1] / 1000 * height, px[2] / 1000 * width, px[3] / 1000 * height]
        draw.rectangle(xy, outline=(255, 40, 40), width=4)
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        tile.paste(image, ((tile_w - width) // 2, 0))
        tile_draw = ImageDraw.Draw(tile)
        query = str(record.get("query", ""))[:100]
        lines = [query[pos : pos + 58] for pos in range(0, len(query), 58)][:2]
        tile_draw.multiline_text((8, 348), "\n".join(lines), fill="black", font=font, spacing=4)
        tile_draw.text((8, 400), f"bbox={record['bbox_normalized']}", fill=(180, 0, 0), font=font)
        canvas.paste(tile, ((index % 3) * tile_w, (index // 3) * tile_h))
    canvas.save(output)
    print(f"{title}: {output}")


def run_validation(args: argparse.Namespace, engine: Any, InferRequest: Any, RequestConfig: Any) -> None:
    val_json = require_file(
        args.val_json or args.root / "prepared_mid3k" / "val.jsonl", "Validation JSONL"
    )
    output_dir = (args.output_dir or args.root / "inference_mid3k_val").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with val_json.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if args.limit and len(records) >= args.limit:
                break
            source = json.loads(line)
            gt = json.loads(source["messages"][-1]["content"])["bbox_2d"]
            records.append(
                {
                    "id": str(index),
                    "query": source["messages"][0]["content"].split("Locate the following target in Image 1:", 1)[-1].split("\nReturn only", 1)[0].strip(),
                    "prompt": source["messages"][0]["content"],
                    "images": [Path(path) for path in source["images"]],
                    "gt": [float(value) for value in gt],
                }
            )

    config = RequestConfig(max_tokens=args.max_new_tokens, temperature=0)
    prediction_path = output_dir / "val_predictions.jsonl"
    preview_records: list[dict[str, Any]] = []
    scores: list[float] = []
    parse_success = 0
    with prediction_path.open("w", encoding="utf-8") as output:
        progress = tqdm(total=len(records), desc="Validating", unit="query")
        for batch in chunks(records, args.batch_size):
            raw_outputs = run_engine_batch(engine, InferRequest, config, batch)
            for record, raw in zip(batch, raw_outputs):
                prediction, parsed = parse_bbox_1000(raw)
                score = iou(prediction, record["gt"])
                scores.append(score)
                parse_success += int(parsed)
                result = {
                    "id": record["id"],
                    "query": record["query"],
                    "gt_bbox_1000": record["gt"],
                    "pred_bbox_1000": [round(value, 3) for value in prediction],
                    "iou": round(score, 6),
                    "parse_ok": parsed,
                    "raw_response": raw,
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                preview_records.append(
                    {
                        **record,
                        "bbox_1000": prediction,
                        "bbox_normalized": normalized_bbox(prediction),
                    }
                )
            output.flush()
            progress.update(len(batch))
            progress.set_postfix(acc50=f"{sum(s >= 0.5 for s in scores) / len(scores):.3f}")
        progress.close()

    accuracy = sum(score >= 0.5 for score in scores) / len(scores) if scores else 0.0
    mean_iou = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "records": len(scores),
        "parse_success": parse_success,
        "parse_success_rate": parse_success / len(scores) if scores else 0.0,
        "mean_iou": mean_iou,
        "acc_at_0_5": accuracy,
    }
    (output_dir / "val_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_preview(preview_records, output_dir / "val_preview.png", "Validation preview")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Predictions: {prediction_path}")


def iter_test_records(data: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(data, dict):
        result = list(data.items())
    elif isinstance(data, list):
        result = [(str(index), value) for index, value in enumerate(data)]
    else:
        raise TypeError("Test JSON must contain a top-level object or list.")
    for key, value in result:
        if not isinstance(value, dict):
            raise TypeError(f"Query {key!r} is not a JSON object.")
    return result


def validate_submission(original: Any, submission: Any) -> None:
    original_records = iter_test_records(original)
    submission_records = iter_test_records(submission)
    if [key for key, _ in original_records] != [key for key, _ in submission_records]:
        raise ValueError("Submission query IDs or ordering changed.")
    for (key, before), (_, after) in zip(original_records, submission_records):
        expected = copy.deepcopy(before)
        expected.pop("bbox", None)
        actual = copy.deepcopy(after)
        bbox = actual.pop("bbox", None)
        if actual != expected:
            raise ValueError(f"Fields other than bbox changed for query {key}.")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox format for query {key}: {bbox!r}")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox):
            raise ValueError(f"Non-numeric bbox for query {key}: {bbox!r}")
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"Out-of-range or empty bbox for query {key}: {bbox!r}")


def run_test(args: argparse.Namespace, engine: Any, InferRequest: Any, RequestConfig: Any) -> None:
    test_root = require_dir(args.test_root or args.root / "competition_test", "Test root")
    test_json = find_test_json(test_root, args.test_json)
    output_dir = (args.output_dir or args.root / "inference_submission").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "depth_visual"

    with test_json.open("r", encoding="utf-8") as stream:
        original = json.load(stream)
    submission = copy.deepcopy(original)
    source_records = iter_test_records(original)
    target_records = dict(iter_test_records(submission))
    selected = source_records[: args.limit or None]

    run_identity = {
        "model": str(args.model),
        "adapter": str(args.adapter),
        "test_json": str(test_json),
        "image_tokens": args.image_tokens,
        "depth_min_mm": args.depth_min_mm,
        "depth_max_mm": args.depth_max_mm,
    }
    config_path = output_dir / "run_config.json"
    state_path = output_dir / "inference_state.jsonl"
    if args.fresh:
        # Start with distinct files; no existing files or directories are deleted.
        suffix = hashlib.sha1(os.urandom(16)).hexdigest()[:8]
        state_path = output_dir / f"inference_state_{suffix}.jsonl"
        config_path = output_dir / f"run_config_{suffix}.json"
    elif config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != run_identity:
            raise RuntimeError(
                f"Existing progress belongs to a different run: {config_path}\n"
                "Use another --output-dir or add --fresh."
            )
    config_path.write_text(json.dumps(run_identity, indent=2), encoding="utf-8")

    completed: dict[str, dict[str, Any]] = {}
    if state_path.is_file():
        with state_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    item = json.loads(line)
                    completed[str(item["id"])] = item
    for key, item in completed.items():
        if key in target_records:
            target_records[key]["bbox"] = item["bbox"]

    pending: list[dict[str, Any]] = []
    prepare_progress = tqdm(selected, desc="Preparing test", unit="query")
    for key, record in prepare_progress:
        if key in completed:
            continue
        required = ("visible", "infrared", "depth", "query")
        missing = [field for field in required if field not in record]
        if missing:
            raise KeyError(f"Query {key} is missing fields: {missing}")
        visible = resolve_input_path(test_root, test_json, record["visible"])
        infrared = resolve_input_path(test_root, test_json, record["infrared"])
        raw_depth = resolve_input_path(test_root, test_json, record["depth"])
        depth_visual = convert_depth(
            raw_depth,
            safe_depth_destination(depth_dir, raw_depth),
            args.depth_min_mm,
            args.depth_max_mm,
        )
        pending.append(
            {
                "id": key,
                "query": str(record["query"]),
                "prompt": make_prompt(str(record["query"])),
                "images": [visible, infrared, depth_visual],
            }
        )
    prepare_progress.close()

    request_config = RequestConfig(max_tokens=args.max_new_tokens, temperature=0)
    preview_records: list[dict[str, Any]] = []
    parse_failures = sum(not bool(item.get("parse_ok", True)) for item in completed.values())
    progress = tqdm(total=len(selected), initial=len(selected) - len(pending), desc="Inferring test", unit="query")
    with state_path.open("a", encoding="utf-8") as state:
        for batch in chunks(pending, args.batch_size):
            raw_outputs = run_engine_batch(engine, InferRequest, request_config, batch)
            for record, raw in zip(batch, raw_outputs):
                box_1000, parsed = parse_bbox_1000(raw)
                bbox = normalized_bbox(box_1000)
                parse_failures += int(not parsed)
                target_records[record["id"]]["bbox"] = bbox
                item = {
                    "id": record["id"],
                    "bbox": bbox,
                    "bbox_1000": [round(value, 3) for value in box_1000],
                    "parse_ok": parsed,
                    "raw_response": raw,
                }
                state.write(json.dumps(item, ensure_ascii=False) + "\n")
                completed[record["id"]] = item
                preview_records.append(
                    {**record, "bbox_1000": box_1000, "bbox_normalized": bbox}
                )
            state.flush()
            progress.update(len(batch))
            progress.set_postfix(parse_fail=parse_failures)
    progress.close()

    save_preview(preview_records, output_dir / "test_preview.png", "Test preview")
    if args.limit and args.limit < len(source_records):
        preview_path = output_dir / f"partial_first_{args.limit}.json"
        preview_path.write_text(
            json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Dry run completed: {len(selected)}/{len(source_records)} queries")
        print(f"Partial result (do not submit): {preview_path}")
        print(f"Run without --limit to resume and finish all queries. State: {state_path}")
        return

    if len(completed) < len(source_records):
        raise RuntimeError(
            f"Only {len(completed)}/{len(source_records)} queries are complete. "
            "Run again without --limit to resume."
        )

    validate_submission(original, submission)
    submission_path = output_dir / test_json.name
    temporary = submission_path.with_suffix(submission_path.suffix + ".tmp")
    temporary.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(submission_path)
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(submission_path, arcname=submission_path.name)

    print("\nSubmission completed")
    print(f"Queries      : {len(source_records)}")
    print(f"Parse failure: {parse_failures} (fallback was a valid full-image box)")
    print(f"JSON         : {submission_path}")
    print(f"ZIP          : {zip_path}")
    print(f"ZIP contents : {submission_path.name}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")

    args.root = args.root.expanduser().resolve()
    args.model = require_dir(args.model or args.root / "models", "Base model")
    args.adapter = require_dir(
        args.adapter
        or args.root
        / "output"
        / "qwen3vl8b_mid3k_full"
        / "v0-20260913-004713"
        / "checkpoint-450",
        "LoRA adapter",
    )
    require_file(args.model / "config.json", "Base model config")
    require_file(args.adapter / "adapter_config.json", "LoRA adapter config")
    require_file(args.adapter / "adapter_model.safetensors", "LoRA adapter weights")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["IMAGE_MAX_TOKEN_NUM"] = str(args.image_tokens)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    engine, InferRequest, RequestConfig = load_swift_engine(
        args.model, args.adapter, args.batch_size
    )
    if args.mode == "val":
        run_validation(args, engine, InferRequest, RequestConfig)
    else:
        run_test(args, engine, InferRequest, RequestConfig)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Completed test records are saved and can be resumed.")
        sys.exit(130)
