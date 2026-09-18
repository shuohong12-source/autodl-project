#!/usr/bin/env python3
"""Prepare a sampled RGBDT500 training set for Qwen3-VL grounding."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm


ROOT = Path("/root/autodl-tmp")
PROMPT_PREFIX = (
    "<image>\nImage 1 is the visible RGB image.\n"
    "<image>\nImage 2 is the aligned thermal infrared image.\n"
    "<image>\nImage 3 is the aligned depth map; brighter valid pixels are closer "
    "and black pixels may be invalid.\n"
)
PROMPT_SUFFIX = (
    'Return only JSON in this format: {"bbox_2d":[x1,y1,x2,y2]}. '
    "Coordinates must be integers from 0 to 1000."
)


@dataclass(frozen=True)
class Sample:
    sequence: str
    filename: str
    rgb: Path
    infrared: Path
    depth: Path
    box: tuple[float, float, float, float]
    width: int
    height: int


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "caption", "prepare"))
    parser.add_argument("--source", type=Path, default=ROOT / "RGBDT500_subset50")
    parser.add_argument("--output", type=Path, default=ROOT / "prepared_rgbdt500")
    parser.add_argument("--model", type=Path, default=ROOT / "models")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--image-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-sequences", type=int, default=5)
    parser.add_argument("--depth-min-mm", type=float, default=300.0)
    parser.add_argument("--depth-max-mm", type=float, default=20000.0)
    parser.add_argument("--caption-limit", type=int, default=0)
    parser.add_argument(
        "--exclude-sequences",
        default="",
        help="Comma-separated sequence IDs to omit from train and val (for example 009,049,057).",
    )
    return parser.parse_args()


def sequences_under(source: Path) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {source}")
    sequences = sorted(p for p in source.iterdir() if p.is_dir() and p.name.isdigit())
    if not sequences:
        raise RuntimeError(f"No numbered sequence directories found under {source}")
    return sequences


def parse_groundtruth(sequence: Path) -> tuple[list[Sample], int]:
    label_file = sequence / "groundtruth.txt"
    if not label_file.is_file():
        raise FileNotFoundError(label_file)
    rows = label_file.read_text(encoding="utf-8-sig").splitlines()
    samples: list[Sample] = []
    invalid = 0
    for line_number, line in enumerate(rows, 1):
        if not line.strip():
            continue
        cells = next(csv.reader([line]))
        if len(cells) != 5:
            raise ValueError(f"{label_file}:{line_number}: expected filename,x,y,w,h")
        filename = cells[0].strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.(?:png|jpe?g)", filename, flags=re.IGNORECASE):
            raise ValueError(f"{label_file}:{line_number}: unsafe filename {filename!r}")
        try:
            box = tuple(float(value.strip()) for value in cells[1:])
        except ValueError as error:
            raise ValueError(f"{label_file}:{line_number}: invalid box") from error
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError(f"{label_file}:{line_number}: non-finite box")
        rgb = sequence / "color" / filename
        infrared = sequence / "infrared" / filename
        depth = sequence / "depth" / (Path(filename).stem + ".png")
        for path in (rgb, infrared, depth):
            if not path.is_file():
                raise FileNotFoundError(f"{label_file}:{line_number}: missing {path}")
        if box[2] <= 0 or box[3] <= 0:
            invalid += 1
            continue
        with Image.open(rgb) as image:
            width, height = image.size
        x, y, w, h = box
        if x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError(f"{label_file}:{line_number}: box outside {width}x{height}: {box}")
        samples.append(
            Sample(sequence.name, filename, rgb.resolve(), infrared.resolve(), depth.resolve(), box, width, height)
        )
    return samples, invalid


def load_samples(source: Path) -> tuple[dict[str, list[Sample]], int]:
    samples_by_sequence: dict[str, list[Sample]] = {}
    invalid = 0
    for sequence in tqdm(sequences_under(source), desc="Checking sequences", unit="seq"):
        samples, skipped = parse_groundtruth(sequence)
        samples_by_sequence[sequence.name] = samples
        invalid += skipped
    return samples_by_sequence, invalid


def box_to_1000(sample: Sample) -> list[int]:
    x, y, w, h = sample.box
    values = [
        round(x / sample.width * 1000),
        round(y / sample.height * 1000),
        round((x + w) / sample.width * 1000),
        round((y + h) / sample.height * 1000),
    ]
    values = [min(1000, max(0, value)) for value in values]
    if values[2] <= values[0]:
        values[2] = min(1000, values[0] + 1)
        values[0] = min(values[0], values[2] - 1)
    if values[3] <= values[1]:
        values[3] = min(1000, values[1] + 1)
        values[1] = min(values[1], values[3] - 1)
    return values


def draw_target(sample: Sample, image_path: Path, crop_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(sample.rgb) as source:
        rgb = source.convert("RGB")
    x, y, w, h = sample.box
    rectangle = (round(x), round(y), round(x + w), round(y + h))
    crop = rgb.crop(rectangle)
    crop.save(crop_path)
    draw = ImageDraw.Draw(rgb)
    thickness = max(3, round(min(rgb.size) / 240))
    for offset in range(thickness):
        draw.rectangle(
            (rectangle[0] - offset, rectangle[1] - offset,
             rectangle[2] + offset, rectangle[3] + offset),
            outline=(255, 25, 25),
        )
    rgb.save(image_path)


def read_captions(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames is None or not {"sequence", "caption"}.issubset(rows.fieldnames):
            raise ValueError(f"{path} must have sequence and caption columns")
        return {
            row["sequence"].strip(): row["caption"].strip()
            for row in rows if row.get("sequence") and row.get("caption")
        }


def write_captions(path: Path, captions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sequence", "caption"])
        writer.writeheader()
        for sequence, caption in sorted(captions.items()):
            writer.writerow({"sequence": sequence, "caption": caption})
    temporary.replace(path)


def clean_caption(response: str) -> str:
    value = response.strip().splitlines()[0].strip(" \t`\"'*-:;,.!") if response.strip() else ""
    value = re.sub(r"^(?:the target is|it is|this is|there is)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" \t`\"'*-:;,.!")
    if not value or len(value.split()) > 25 or len(value) > 180:
        raise ValueError(f"Caption needs review: {response!r}")
    return value


def make_caption_review(
    samples_by_sequence: dict[str, list[Sample]], captions: dict[str, str], output: Path
) -> None:
    font = ImageFont.load_default()
    pairs = [(sequence, samples[0]) for sequence, samples in sorted(samples_by_sequence.items()) if samples]
    for page_number, start in enumerate(range(0, len(pairs), 10), 1):
        page = Image.new("RGB", (1000, 5 * 280), "white")
        for index, (sequence, sample) in enumerate(pairs[start : start + 10]):
            row, column = divmod(index, 2)
            with Image.open(sample.rgb) as source:
                image = source.convert("RGB")
            x, y, w, h = sample.box
            ImageDraw.Draw(image).rectangle((x, y, x + w, y + h), outline="red", width=12)
            image.thumbnail((480, 230))
            page.paste(image, (column * 500 + 10, row * 280 + 10))
            caption = f"{sequence}: {captions.get(sequence, '[MISSING]')}"
            ImageDraw.Draw(page).multiline_text(
                (column * 500 + 10, row * 280 + 237),
                "\n".join(textwrap.wrap(caption, width=65)[:2]),
                fill="black", font=font,
            )
        destination = output / f"caption_review_{page_number:02d}.jpg"
        page.save(destination, quality=85)
    print(f"Caption review sheets: {output / 'caption_review_*.jpg'}")


def generate_captions(args: argparse.Namespace, samples_by_sequence: dict[str, list[Sample]]) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    captions_path = output / "captions.csv"
    captions = read_captions(captions_path)
    todo = [(sequence, samples[0]) for sequence, samples in sorted(samples_by_sequence.items())
            if samples and not captions.get(sequence)]
    if args.caption_limit:
        todo = todo[:args.caption_limit]
    if todo:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ["IMAGE_MAX_TOKEN_NUM"] = str(args.image_tokens)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable, so loading Qwen3-VL-8B would use CPU RAM and "
                "may be killed. Start a GPU instance, then check nvidia-smi and "
                "python -c 'import torch; print(torch.cuda.is_available())'."
            )
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        free_bytes, _ = torch.cuda.mem_get_info(0)
        print(
            f"Caption GPU: {torch.cuda.get_device_name(0)}, "
            f"free VRAM: {free_bytes / 1024**3:.1f} GiB, dtype: {dtype}"
        )
        from swift import InferRequest, RequestConfig, TransformersEngine

        model = args.model.resolve()
        if not (model / "config.json").is_file():
            raise FileNotFoundError(f"Base Qwen3-VL model not found: {model}")
        print(f"Loading base Qwen3-VL model for {len(todo)} target descriptions...")
        engine = TransformersEngine(
            str(model), model_type="qwen3_vl", max_batch_size=1,
            torch_dtype=dtype, device_map="cuda:0",
        )
        config = RequestConfig(max_tokens=80, temperature=0)
        for sequence, sample in tqdm(todo, desc="Captioning", unit="seq"):
            boxed = output / "caption_inputs" / f"{sequence}_boxed.png"
            crop = output / "caption_inputs" / f"{sequence}_crop.png"
            draw_target(sample, boxed, crop)
            request = InferRequest(
                messages=[{"role": "user", "content": (
                    "<image>Image 1 is a full RGB scene. A red rectangle marks the target.\n"
                    "<image>Image 2 is a crop of that same target.\n"
                    "Describe ONLY the target in a concise English noun phrase. "
                    "Include its object category and clearly visible distinguishing appearance. "
                    "Do not guess details, do not describe the background, and do not mention the rectangle. "
                    "Reply with the noun phrase only."
                )}],
                images=[str(boxed), str(crop)],
            )
            raw = engine.infer([request], request_config=config)[0].choices[0].message.content or ""
            try:
                captions[sequence] = clean_caption(raw)
            except ValueError as error:
                print(f"Sequence {sequence}: {error}")
                continue
            write_captions(captions_path, captions)
    make_caption_review(samples_by_sequence, captions, output)
    print(f"Captions: {len(captions)}/{sum(bool(samples) for samples in samples_by_sequence.values())}")
    print(f"Edit incorrect descriptions in {captions_path} before running prepare.")


def convert_depth(source: Path, target: Path, minimum_mm: float, maximum_mm: float) -> None:
    if target.is_file():
        return
    if maximum_mm <= minimum_mm:
        raise ValueError("depth-max-mm must be greater than depth-min-mm")
    with Image.open(source) as image:
        depth = np.asarray(image)
    if depth.ndim != 2:
        raise ValueError(f"Expected single-channel depth PNG: {source}")
    depth = depth.astype(np.float32, copy=False)
    valid = depth > 0
    visual = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth[valid], minimum_mm, maximum_mm)
        visual[valid] = np.rint(
            1.0 + (maximum_mm - clipped) * 254.0 / (maximum_mm - minimum_mm)
        ).astype(np.uint8)
    rgb = np.repeat(visual[:, :, None], 3, axis=2)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(target, compress_level=4)


def make_record(sample: Sample, caption: str, depth_visual: Path) -> dict:
    center_x = (sample.box[0] + sample.box[2] / 2) / sample.width
    if center_x < 1 / 3:
        location = "on the left side of the image"
    elif center_x > 2 / 3:
        location = "on the right side of the image"
    else:
        location = "near the center of the image"
    description = f"{caption.strip().rstrip('.')}, {location}"
    prompt = PROMPT_PREFIX + f"Locate the following target in Image 1: {description}.\n" + PROMPT_SUFFIX
    return {
        "images": [str(sample.rgb), str(sample.infrared), str(depth_visual.resolve())],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps({"bbox_2d": box_to_1000(sample)}, separators=(",", ":"))},
        ],
    }


def prepare(args: argparse.Namespace, samples_by_sequence: dict[str, list[Sample]], invalid: int) -> None:
    output = args.output.resolve()
    excluded = {value.strip() for value in args.exclude_sequences.split(",") if value.strip()}
    unknown = excluded - set(samples_by_sequence)
    if unknown:
        raise ValueError(f"Unknown sequence IDs in --exclude-sequences: {sorted(unknown)}")
    selected = {
        sequence: samples for sequence, samples in samples_by_sequence.items()
        if sequence not in excluded
    }
    captions = read_captions(output / "captions.csv")
    missing = sorted(sequence for sequence, samples in selected.items()
                     if samples and not captions.get(sequence))
    if missing:
        raise RuntimeError(f"Missing captions for {len(missing)} sequences: {missing}. Run caption first.")
    sequence_names = sorted(selected)
    sampling_manifest = args.source.resolve() / "sampling_manifest.json"
    if sampling_manifest.is_file():
        plan = json.loads(sampling_manifest.read_text(encoding="utf-8"))
        val_names = set(plan["val_sequence_ids"])
        if not val_names or val_names >= set(sequence_names):
            raise ValueError("Sampling manifest must specify a nonempty proper validation split")
        if not val_names <= set(sequence_names):
            raise ValueError("Sampling manifest references missing validation sequences")
        if val_names & excluded:
            raise ValueError("Excluded sequences overlap the fixed validation split")
    else:
        if not 0 < args.val_sequences < len(selected):
            raise ValueError("val-sequences must be between 1 and number of sequences - 1")
        random.Random(args.seed).shuffle(sequence_names)
        val_names = set(sequence_names[:args.val_sequences])
    train_path = output / "train.jsonl"
    val_path = output / "val.jsonl"
    depth_dir = output / "depth_visual"
    counts = {"train": 0, "val": 0}
    output.mkdir(parents=True, exist_ok=True)
    with train_path.open("w", encoding="utf-8") as train, val_path.open("w", encoding="utf-8") as val:
        for sequence in tqdm(sequence_names, desc="Converting", unit="seq"):
            stream = val if sequence in val_names else train
            split = "val" if sequence in val_names else "train"
            for sample in selected[sequence]:
                depth_visual = depth_dir / sequence / (Path(sample.filename).stem + ".png")
                convert_depth(sample.depth, depth_visual, args.depth_min_mm, args.depth_max_mm)
                record = make_record(sample, captions[sequence], depth_visual)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1

    manifest = {
        "source": str(args.source.resolve()),
        "source_sequences": len(sequence_names),
        "excluded_sequences": sorted(excluded),
        "invalid_empty_boxes_skipped": invalid,
        "train_sequences": len(sequence_names) - len(val_names),
        "val_sequences": len(val_names),
        "val_sequence_ids": sorted(val_names),
        "train_records": counts["train"],
        "val_records": counts["val"],
        "depth_min_mm": args.depth_min_mm,
        "depth_max_mm": args.depth_max_mm,
        "seed": args.seed,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    make_caption_review(selected, captions, output)
    print(json.dumps(manifest, indent=2))
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
    print("Do not train until the caption_review images have been checked.")


def main() -> None:
    args = arguments()
    samples_by_sequence, invalid = load_samples(args.source.resolve())
    total = sum(map(len, samples_by_sequence.values()))
    print(f"Sequences: {len(samples_by_sequence)}")
    print(f"Valid annotated triplets: {total}")
    print(f"Empty/invalid boxes skipped: {invalid}")
    if args.mode == "audit":
        return
    if args.mode == "caption":
        generate_captions(args, samples_by_sequence)
    else:
        prepare(args, samples_by_sequence, invalid)


if __name__ == "__main__":
    main()
