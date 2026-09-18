#!/usr/bin/env python3
"""Read-only integrity audit for the three MID-3K modality repositories."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm


EXPECTED_FILES = 3083
MODALITIES = ("rgb", "thermal", "depth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MID-3K RGB/Thermal/Depth data.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/root/autodl-tmp/datasets"),
        help="Directory containing the MID-3K repositories.",
    )
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() == ".png"
    )


def label_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() == ".txt"
    )


def find_best_images_dir(root: Path, modality: str) -> tuple[Path, list[Path]]:
    candidates = []
    for directory in root.rglob("images"):
        parent_name = directory.parent.name.lower()
        if "mid-3k" not in parent_name or modality not in parent_name:
            continue
        files = image_files(directory)
        candidates.append((len(files), directory, files))

    if not candidates:
        raise FileNotFoundError(f"No MID-3K {modality} images directory found")

    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    print(f"\n{modality.upper()} candidates:")
    for count, directory, _files in candidates:
        marker = "  <-- selected" if directory == candidates[0][1] else ""
        print(f"  {count:4d} images  {directory}{marker}")
    return candidates[0][1], candidates[0][2]


def check_images(modality: str, files: list[Path]) -> tuple[int, Counter]:
    corrupt = []
    dimensions: Counter = Counter()
    modes: Counter = Counter()
    for path in tqdm(files, desc=f"Checking {modality} images", unit="image"):
        try:
            with Image.open(path) as image:
                dimensions[image.size] += 1
                modes[image.mode] += 1
                image.verify()
        except Exception as error:
            corrupt.append((path, str(error)))

    print(f"  decoded: {len(files) - len(corrupt)}/{len(files)}")
    print(f"  dimensions: {dict(dimensions)}")
    print(f"  image modes: {dict(modes)}")
    for path, error in corrupt[:10]:
        print(f"  CORRUPT: {path}: {error}")
    return len(corrupt), dimensions


def check_labels(directory: Path) -> tuple[int, int, int]:
    files = label_files(directory)
    invalid = []
    annotation_count = 0
    empty_files = 0

    for path in tqdm(files, desc=f"Checking {directory.parent.name} labels", unit="label"):
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            empty_files += 1
        for line_number, line in enumerate(lines, start=1):
            annotation_count += 1
            parts = line.split()
            try:
                if len(parts) != 5:
                    raise ValueError(f"expected 5 columns, got {len(parts)}")
                class_id, x_center, y_center, width, height = map(float, parts)
                if class_id < 0:
                    raise ValueError("negative class id")
                if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
                    raise ValueError("coordinate outside [0, 1]")
                if width <= 0.0 or height <= 0.0:
                    raise ValueError("non-positive box size")
            except ValueError as error:
                invalid.append((path, line_number, line, str(error)))

    print(f"  labels: {len(files)}")
    print(f"  annotations: {annotation_count}")
    print(f"  empty label files: {empty_files}")
    print(f"  invalid annotations: {len(invalid)}")
    for path, line_number, line, error in invalid[:10]:
        print(f"  INVALID: {path}:{line_number}: {error}: {line}")
    return len(files), len(invalid), annotation_count


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    print(f"MID-3K audit root: {root}")
    selected: dict[str, tuple[Path, list[Path]]] = {}
    failed = False

    for modality in MODALITIES:
        images_dir, files = find_best_images_dir(root, modality)
        selected[modality] = (images_dir, files)
        print(f"  selected image count: {len(files)} (expected {EXPECTED_FILES})")
        if len(files) != EXPECTED_FILES:
            failed = True

        corrupt, dimensions = check_images(modality, files)
        labels_dir = images_dir.parent / "labels"
        labels_count, invalid_count, _annotations = check_labels(labels_dir)
        unexpected_sizes = [size for size in dimensions if size != (640, 512)]
        if unexpected_sizes:
            print(f"  WARNING: unexpected image sizes: {unexpected_sizes}")
        if (
            corrupt
            or invalid_count
            or labels_count != EXPECTED_FILES
            or unexpected_sizes
        ):
            failed = True

        print("  first image names:")
        for path in files[:5]:
            print(f"    {path.name}")

    stem_sets = {
        modality: {path.stem for path in files}
        for modality, (_directory, files) in selected.items()
    }
    exact_common = set.intersection(*stem_sets.values())
    print("\nCross-modality filename check:")
    print(f"  exact common stems: {len(exact_common)}")

    prefix_sets = {
        modality: {path.stem.split("_", 1)[0] for path in files}
        for modality, (_directory, files) in selected.items()
    }
    common_prefixes = set.intersection(*prefix_sets.values())
    print(f"  common prefixes before first underscore: {len(common_prefixes)}")

    csv_files = sorted(root.rglob("*.csv"))
    print("\nCSV metadata candidates:")
    if csv_files:
        for path in csv_files:
            print(f"  {path}")
    else:
        print("  none found")

    print("\nRESULT:", "CHECK FAILED" if failed else "BASIC INTEGRITY PASSED")
    print("Send this complete report before generating the pairing JSONL.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
