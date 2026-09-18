#!/usr/bin/env python3
"""Convert MID-3K RGB/Thermal/Depth person boxes to Qwen3-VL JSONL."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm.auto import tqdm


DEFAULT_ROOT = Path("/root/autodl-tmp/datasets")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/prepared_mid3k")


@dataclass(frozen=True)
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 2000 MID-3K triplets.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-size", type=int, default=1800)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--depth-min-mm",
        type=float,
        default=300.0,
        help="Nearest valid depth used for fixed visualization scaling.",
    )
    parser.add_argument(
        "--depth-max-mm",
        type=float,
        default=20000.0,
        help="Farthest valid depth used for fixed visualization scaling.",
    )
    parser.add_argument("--preview-samples", type=int, default=6)
    return parser.parse_args()


def png_index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    result = {
        path.stem: path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }
    if not result:
        raise RuntimeError(f"No PNG images found in {directory}")
    return result


def read_yolo_boxes(path: Path) -> list[Box]:
    boxes = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 columns")
        class_value, x_center, y_center, width, height = map(float, parts)
        if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
            raise ValueError(f"{path}:{line_number}: values outside [0, 1]")
        x1 = max(0.0, x_center - width / 2)
        y1 = max(0.0, y_center - height / 2)
        x2 = min(1.0, x_center + width / 2)
        y2 = min(1.0, y_center + height / 2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{path}:{line_number}: invalid box")
        boxes.append(Box(int(class_value), x1, y1, x2, y2))
    return boxes


def ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def make_query(boxes: list[Box], target_index: int) -> str:
    if len(boxes) == 1:
        return "the only person visible in the scene"

    ordered_indices = sorted(
        range(len(boxes)), key=lambda index: (boxes[index].center_x, boxes[index].center_y)
    )
    rank = ordered_indices.index(target_index)
    if rank == 0:
        return "the leftmost person in the scene"
    if rank == len(boxes) - 1:
        return "the rightmost person in the scene"
    return f"the {ordinal(rank + 1)} person from the left"


def normalized_1000(box: Box) -> list[int]:
    values = [
        round(box.x1 * 1000),
        round(box.y1 * 1000),
        round(box.x2 * 1000),
        round(box.y2 * 1000),
    ]
    values = [min(1000, max(0, value)) for value in values]
    if values[2] <= values[0]:
        values[2] = min(1000, values[0] + 1)
    if values[3] <= values[1]:
        values[3] = min(1000, values[1] + 1)
    return values


def convert_depth(
    source: Path,
    destination: Path,
    minimum_mm: float,
    maximum_mm: float,
) -> tuple[int, int, int]:
    if maximum_mm <= minimum_mm:
        raise ValueError("depth-max-mm must be larger than depth-min-mm")

    with Image.open(source) as image:
        depth = np.asarray(image, dtype=np.uint16)
    if depth.ndim != 2:
        raise ValueError(f"Expected a single-channel depth map: {source}")

    valid = depth > 0
    visual = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth[valid].astype(np.float32), minimum_mm, maximum_mm)
        # Valid near pixels are bright; zero/invalid depth remains black.
        scaled = 1.0 + (maximum_mm - clipped) * 254.0 / (maximum_mm - minimum_mm)
        visual[valid] = np.rint(scaled).astype(np.uint8)

    rgb_visual = np.repeat(visual[:, :, None], 3, axis=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_visual, mode="RGB").save(destination, compress_level=4)
    return int(valid.sum()), int((depth > maximum_mm).sum()), int((depth < minimum_mm).sum())


def make_record(
    rgb: Path,
    thermal: Path,
    depth_visual: Path,
    query: str,
    bbox: list[int],
) -> dict:
    prompt = (
        "<image>\nImage 1 is the visible RGB image.\n"
        "<image>\nImage 2 is the aligned thermal infrared image.\n"
        "<image>\nImage 3 is the aligned depth map; brighter valid pixels are closer "
        "and black pixels may be invalid.\n"
        f"Locate the following target in Image 1: {query}.\n"
        'Return only JSON in this format: {"bbox_2d":[x1,y1,x2,y2]}. '
        "Coordinates must be integers from 0 to 1000."
    )
    return {
        "images": [str(rgb), str(thermal), str(depth_visual.resolve())],
        "messages": [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": json.dumps({"bbox_2d": bbox}, separators=(",", ":")),
            },
        ],
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_preview(records: list[dict], output: Path, count: int) -> None:
    chosen = records[: min(count, len(records))]
    if not chosen:
        return
    figure, axes = plt.subplots(
        len(chosen), 3, figsize=(15, max(4, 3.2 * len(chosen))), squeeze=False
    )
    for row, record in enumerate(chosen):
        image_paths = [Path(path) for path in record["images"]]
        images = [Image.open(path).convert("RGB") for path in image_paths]
        bbox = json.loads(record["messages"][1]["content"])["bbox_2d"]
        query = record["messages"][0]["content"].split(
            "Locate the following target in Image 1: ", 1
        )[1].split(".\nReturn only", 1)[0]

        axes[row][0].imshow(images[0])
        x1, y1, x2, y2 = bbox
        axes[row][0].add_patch(
            Rectangle(
                (x1 / 1000 * images[0].width, y1 / 1000 * images[0].height),
                (x2 - x1) / 1000 * images[0].width,
                (y2 - y1) / 1000 * images[0].height,
                fill=False,
                edgecolor="#ff3030",
                linewidth=2,
            )
        )
        axes[row][0].set_title(f"RGB: {query}", fontsize=9)
        axes[row][1].imshow(images[1])
        axes[row][1].set_title("Thermal", fontsize=9)
        axes[row][2].imshow(images[2])
        axes[row][2].set_title("Depth visualization", fontsize=9)
        for axis in axes[row]:
            axis.axis("off")
        for image in images:
            image.close()

    figure.suptitle("MID-3K three-modality training preview", fontsize=13)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    rgb_dir = root / "MID-3K-rgb" / "images"
    thermal_dir = root / "MID-3K-thermal" / "images"
    depth_dir = root / "MID-3K-depth" / "images"
    labels_dir = root / "MID-3K-rgb" / "labels"

    rgb_index = png_index(rgb_dir)
    thermal_index = png_index(thermal_dir)
    depth_index = png_index(depth_dir)
    common_stems = sorted(set(rgb_index) & set(thermal_index) & set(depth_index))
    print(f"Exact RGB/Thermal/Depth triplets: {len(common_stems)}")

    scenes = []
    for stem in tqdm(common_stems, desc="Reading RGB labels", unit="scene"):
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.is_file():
            continue
        boxes = read_yolo_boxes(label_path)
        if boxes:
            scenes.append((stem, boxes))

    requested = args.train_size + args.val_size
    print(f"Non-empty aligned scenes: {len(scenes)}")
    if requested > len(scenes):
        raise ValueError(
            f"Requested {requested} scenes, but only {len(scenes)} are non-empty"
        )
    available_non_empty_scenes = len(scenes)

    rng = random.Random(args.seed)
    rng.shuffle(scenes)
    scenes = scenes[:requested]

    records = []
    depth_visual_dir = output / "depth_visual"
    total_valid = 0
    total_far = 0
    total_near_or_invalid = 0
    for stem, boxes in tqdm(scenes, desc="Converting MID-3K", unit="scene"):
        target_index = rng.randrange(len(boxes))
        target = boxes[target_index]
        query = make_query(boxes, target_index)
        depth_visual = depth_visual_dir / f"{stem}.png"
        valid, far, near_or_invalid = convert_depth(
            depth_index[stem],
            depth_visual,
            args.depth_min_mm,
            args.depth_max_mm,
        )
        total_valid += valid
        total_far += far
        total_near_or_invalid += near_or_invalid
        records.append(
            make_record(
                rgb_index[stem],
                thermal_index[stem],
                depth_visual,
                query,
                normalized_1000(target),
            )
        )

    train_records = records[: args.train_size]
    val_records = records[args.train_size :]
    train_path = output / "train.jsonl"
    val_path = output / "val.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    create_preview(train_records, output / "preview.png", args.preview_samples)

    manifest = {
        "seed": args.seed,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "source_triplets": len(common_stems),
        "non_empty_scenes": available_non_empty_scenes,
        "depth_visualization": {
            "minimum_mm": args.depth_min_mm,
            "maximum_mm": args.depth_max_mm,
            "valid_pixels": total_valid,
            "pixels_above_maximum": total_far,
            "pixels_below_minimum_including_zero": total_near_or_invalid,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nMID-3K conversion completed")
    print(f"Train : {len(train_records)} -> {train_path}")
    print(f"Val   : {len(val_records)} -> {val_path}")
    print(f"Depth : {depth_visual_dir}")
    print(f"Preview: {output / 'preview.png'}")
    print("First record:")
    print(json.dumps(train_records[0], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
