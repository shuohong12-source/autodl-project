#!/usr/bin/env python3
"""Create a shuffled, non-destructive MID-3K/RGBDT500 training JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mid3k", type=Path, required=True)
    parser.add_argument("--rgbdt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def records(path: Path) -> list[dict]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if len(item.get("images", [])) != 3:
                raise ValueError(f"{path}:{number}: expected RGB, infrared, and depth images")
            if [message.get("role") for message in item.get("messages", [])] != ["user", "assistant"]:
                raise ValueError(f"{path}:{number}: expected one user and one assistant message")
            box = json.loads(item["messages"][1]["content"]).get("bbox_2d")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"{path}:{number}: invalid bbox_2d")
            result.append(item)
    if not result:
        raise ValueError(f"No records in {path}")
    return result


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    mid3k = records(args.mid3k)
    rgbdt = records(args.rgbdt)
    mixed = mid3k + rgbdt
    random.Random(args.seed).shuffle(mixed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for item in mixed:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"MID-3K: {len(mid3k)}; RGBDT500: {len(rgbdt)}; total: {len(mixed)}")
    print(f"Training file: {args.output}")


if __name__ == "__main__":
    main()
