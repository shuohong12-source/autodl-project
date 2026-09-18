import json
from pathlib import Path

import torch
from tqdm import tqdm


ROOT = Path("/root/autodl-tmp")

ANN_DIR = ROOT / "datasets/rgbt_ground/rgbtvg_flir"
RGB_DIR = ROOT / "datasets/rgbt_ground/image_data/flir/rgb"
IR_DIR = ROOT / "datasets/rgbt_ground/image_data/flir/ir"
OUT_DIR = ROOT / "prepared_flir"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_coordinate(value, size):
    """Convert an absolute coordinate to an integer in the 0-1000 range."""
    normalized = round(float(value) / float(size) * 1000)
    return max(0, min(1000, normalized))


def convert_split(split):
    annotation_path = ANN_DIR / f"rgbtvg_flir_{split}.pth"
    output_path = OUT_DIR / f"{split}.jsonl"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation not found: {annotation_path}")

    print(f"\nLoading annotations: {annotation_path}")

    samples = torch.load(
        annotation_path,
        map_location="cpu",
        weights_only=False,
    )

    print(f"Found {len(samples)} annotations")

    written = 0
    missing = 0
    invalid = 0
    first_record = None

    with output_path.open("w", encoding="utf-8") as output_file:
        progress = tqdm(
            samples,
            desc=f"Converting {split}",
            unit="sample",
            dynamic_ncols=True,
        )

        for item in progress:
            if len(item) < 4:
                invalid += 1
                continue

            filename, image_size, bbox, phrase = item[:4]

            rgb_path = RGB_DIR / filename
            ir_path = IR_DIR / filename

            if not rgb_path.exists() or not ir_path.exists():
                missing += 1
                progress.set_postfix(
                    written=written,
                    missing=missing,
                    invalid=invalid,
                )
                continue

            try:
                image_width = float(image_size["width"])
                image_height = float(image_size["height"])

                x, y, box_width, box_height = map(float, bbox)

                x1 = x
                y1 = y
                x2 = x + box_width
                y2 = y + box_height

                if (
                    image_width <= 0
                    or image_height <= 0
                    or box_width <= 0
                    or box_height <= 0
                    or x2 <= x1
                    or y2 <= y1
                ):
                    invalid += 1
                    continue

                bbox_1000 = [
                    normalize_coordinate(x1, image_width),
                    normalize_coordinate(y1, image_height),
                    normalize_coordinate(x2, image_width),
                    normalize_coordinate(y2, image_height),
                ]

                prompt = (
                    "<image>\n"
                    "Image 1 is the visible RGB image.\n"
                    "<image>\n"
                    "Image 2 is the aligned thermal infrared image.\n"
                    f"Locate the following target in Image 1: {phrase}\n"
                    "Return only JSON in this format: "
                    '{"bbox_2d":[x1,y1,x2,y2]}. '
                    "Coordinates must be integers from 0 to 1000."
                )

                response = json.dumps(
                    {"bbox_2d": bbox_1000},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                record = {
                    "images": [
                        str(rgb_path.resolve()),
                        str(ir_path.resolve()),
                    ],
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                        {
                            "role": "assistant",
                            "content": response,
                        },
                    ],
                }

                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                if first_record is None:
                    first_record = record

                written += 1

                if written % 100 == 0:
                    progress.set_postfix(
                        written=written,
                        missing=missing,
                        invalid=invalid,
                    )

            except (KeyError, TypeError, ValueError):
                invalid += 1

    print(f"\nFinished converting {split}")
    print(f"Written : {written}")
    print(f"Missing : {missing}")
    print(f"Invalid : {invalid}")
    print(f"Output  : {output_path}")

    if first_record is not None:
        print("\nFirst converted record:")
        print(
            json.dumps(
                first_record,
                ensure_ascii=False,
                indent=2,
            )
        )


def main():
    print("RGBT-GroundBench to Qwen3-VL converter")
    print(f"RGB directory: {RGB_DIR}")
    print(f"IR directory : {IR_DIR}")
    print(f"Output       : {OUT_DIR}")

    convert_split("train")
    convert_split("val")

    print("\nAll conversions completed.")


if __name__ == "__main__":
    main()