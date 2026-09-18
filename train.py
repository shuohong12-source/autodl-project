#!/usr/bin/env python3
"""Launch Qwen3-VL-8B LoRA training with live visual monitoring."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm.auto import tqdm


DEFAULT_ROOT = Path("/root/autodl-tmp")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METRIC_PATTERNS = {
    "loss": re.compile(r"['\"]loss['\"]\s*:\s*([-+0-9.eE]+)"),
    "learning_rate": re.compile(
        r"['\"](?:learning_rate|learning_rate_0)['\"]\s*:\s*([-+0-9.eE]+)"
    ),
    "epoch": re.compile(r"['\"]epoch['\"]\s*:\s*([-+0-9.eE]+)"),
    "total_steps": re.compile(
        r"(?:Total|Num) (?:optimization|training) steps\s*=\s*([0-9,]+)",
        re.IGNORECASE,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen3-VL-8B LoRA and save live training charts."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--train-json", type=Path, default=None)
    parser.add_argument("--val-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use all 7000/608 samples. Without this flag, use a 100/50 smoke test.",
    )
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--image-tokens", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--preview-samples", type=int, default=4)
    parser.add_argument("--chart-every", type=int, default=5)
    tensorboard_group = parser.add_mutually_exclusive_group()
    tensorboard_group.add_argument(
        "--tensorboard",
        dest="tensorboard",
        action="store_true",
        help="Start TensorBoard on port 6006 (default).",
    )
    tensorboard_group.add_argument(
        "--no-tensorboard",
        dest="tensorboard",
        action="store_false",
        help="Do not start TensorBoard.",
    )
    parser.set_defaults(tensorboard=True)
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def make_subset(source: Path, target: Path, count: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open("r", encoding="utf-8") as src, target.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if line.strip():
                dst.write(line)
                written += 1
                if written >= count:
                    break
    if written == 0:
        raise RuntimeError(f"No records found in {source}")
    print(f"Created {target} with {written} samples")


def load_records(path: Path, count: int) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if len(records) >= count:
                    break
    return records


def count_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def extract_bbox(record: dict) -> list[float]:
    messages = record.get("messages", [])
    answer = next(
        message.get("content", "")
        for message in messages
        if message.get("role") == "assistant"
    )
    bbox = json.loads(answer)["bbox_2d"]
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox}")
    return [float(value) for value in bbox]


def extract_query(record: dict) -> str:
    user_text = next(
        message.get("content", "")
        for message in record.get("messages", [])
        if message.get("role") == "user"
    )
    marker = "Locate the following target in Image 1:"
    query = user_text.split(marker, 1)[-1].split("\nReturn only", 1)[0]
    return query.strip()


def create_dataset_preview(dataset: Path, output: Path, count: int) -> None:
    records = load_records(dataset, count)
    if not records:
        raise RuntimeError(f"No preview records found in {dataset}")

    figure, axes = plt.subplots(
        len(records), 2, figsize=(12, max(4, len(records) * 3.5)), squeeze=False
    )
    for row, record in enumerate(records):
        images = record.get("images", [])
        if len(images) != 2:
            raise ValueError(f"Expected two images, got {len(images)}")
        rgb_path, ir_path = map(Path, images)
        require_file(rgb_path, "RGB image")
        require_file(ir_path, "IR image")

        rgb = Image.open(rgb_path).convert("RGB")
        infrared = Image.open(ir_path).convert("RGB")
        x1, y1, x2, y2 = extract_bbox(record)
        px1, py1 = x1 / 1000 * rgb.width, y1 / 1000 * rgb.height
        px2, py2 = x2 / 1000 * rgb.width, y2 / 1000 * rgb.height

        axes[row][0].imshow(rgb)
        axes[row][0].add_patch(
            Rectangle(
                (px1, py1),
                px2 - px1,
                py2 - py1,
                fill=False,
                edgecolor="#ff3030",
                linewidth=2,
            )
        )
        axes[row][0].set_title(extract_query(record), fontsize=9, wrap=True)
        axes[row][0].axis("off")
        axes[row][1].imshow(infrared)
        axes[row][1].set_title("Thermal infrared", fontsize=9)
        axes[row][1].axis("off")

    figure.suptitle("Training samples: RGB target box and paired infrared", fontsize=13)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print(f"Dataset preview: {output}")


def create_live_dashboard(visual_dir: Path) -> Path:
    dashboard = visual_dir / "live_dashboard.html"
    dashboard.write_text(
        """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3-VL-8B training monitor</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; color: #202124; background: #f5f6f8; }
    header { padding: 16px 24px; color: white; background: #202124; }
    main { display: grid; grid-template-columns: minmax(320px, 1fr) minmax(420px, 1.5fr); gap: 16px; padding: 16px; }
    section { padding: 14px; background: white; border: 1px solid #dfe1e5; border-radius: 6px; }
    h1 { margin: 0; font-size: 20px; }
    h2 { margin: 0 0 10px; font-size: 16px; }
    p { margin: 7px 0 0; color: #bdc1c6; font-size: 13px; }
    img { display: block; width: 100%; height: auto; }
    #waiting { padding: 40px 12px; color: #5f6368; text-align: center; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Qwen3-VL-8B LoRA training monitor</h1>
    <p id="status">Waiting for the first training log...</p>
  </header>
  <main>
    <section>
      <h2>RGB target boxes and paired infrared images</h2>
      <img src="dataset_preview.png" alt="Dataset preview">
    </section>
    <section>
      <h2>Live metrics</h2>
      <div id="waiting">The chart appears after the first loss value is logged.</div>
      <img id="chart" alt="Live training chart" hidden>
    </section>
  </main>
  <script>
    const chart = document.getElementById('chart');
    const waiting = document.getElementById('waiting');
    const status = document.getElementById('status');
    function refresh() {
      const probe = new Image();
      const stamp = Date.now();
      probe.onload = () => {
        chart.src = `training_live.png?t=${stamp}`;
        chart.hidden = false;
        waiting.hidden = true;
        status.textContent = `Chart refreshed at ${new Date().toLocaleTimeString()}`;
      };
      probe.onerror = () => { status.textContent = 'Waiting for the first training log...'; };
      probe.src = `training_live.png?t=${stamp}`;
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return dashboard


def gpu_memory_gib(gpu: str) -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(result.stdout.strip().splitlines()[0]) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def update_live_chart(
    output: Path,
    steps: list[int],
    losses: list[float],
    learning_rates: list[float],
    gpu_memory: list[float | None],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(steps, losses, color="#e53935", linewidth=1.8)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Log step")
    axes[0].grid(alpha=0.25)

    lr_steps = steps[: len(learning_rates)]
    axes[1].plot(lr_steps, learning_rates, color="#1565c0", linewidth=1.8)
    axes[1].set_title("Learning rate")
    axes[1].set_xlabel("Log step")
    axes[1].grid(alpha=0.25)

    memory_points = [
        (step, value)
        for step, value in zip(steps, gpu_memory)
        if value is not None
    ]
    if memory_points:
        memory_steps, memory_values = zip(*memory_points)
        axes[2].plot(memory_steps, memory_values, color="#00897b", linewidth=1.8)
    axes[2].set_title("GPU memory (GiB)")
    axes[2].set_xlabel("Log step")
    axes[2].grid(alpha=0.25)

    updated_at = time.strftime("%H:%M:%S")
    figure.suptitle(f"Qwen3-VL-8B live training metrics | updated {updated_at}")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=130, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output)


def start_tensorboard(log_dir: Path) -> subprocess.Popen | None:
    executable = shutil.which("tensorboard")
    if executable is None:
        print("TensorBoard is not installed; continuing with PNG monitoring.")
        return None
    process = subprocess.Popen(
        [executable, "--logdir", str(log_dir), "--host", "0.0.0.0", "--port", "6006"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("TensorBoard started on port 6006")
    return process


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    model = (args.model or root / "models").resolve()
    source_train = (args.train_json or root / "prepared_flir/train.jsonl").resolve()
    source_val = (args.val_json or root / "prepared_flir/val.jsonl").resolve()
    run_name = "qwen3vl8b_flir_full" if args.full else "qwen3vl8b_flir_smoke"
    output_dir = (args.output_dir or root / "output" / run_name).resolve()
    visual_dir = output_dir / "visuals"
    logging_dir = output_dir / "tensorboard"

    require_file(model / "config.json", "Model config")
    require_file(source_train, "Training JSONL")
    require_file(source_val, "Validation JSONL")
    swift = shutil.which("swift")
    if swift is None:
        raise RuntimeError('MS-SWIFT is missing. Run: pip install "ms-swift>=4.0"')

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.full:
        train_json, val_json = source_train, source_val
    else:
        train_json = output_dir / "smoke_train.jsonl"
        val_json = output_dir / "smoke_val.jsonl"
        make_subset(source_train, train_json, 100)
        make_subset(source_val, val_json, 50)

    create_dataset_preview(
        train_json, visual_dir / "dataset_preview.png", args.preview_samples
    )
    dashboard = create_live_dashboard(visual_dir)
    print(f"Auto-refresh dashboard: {dashboard}")

    epochs = args.epochs if args.epochs is not None else (2.0 if args.full else 1.0)
    train_samples = count_records(train_json)
    steps_per_epoch = max(
        1, math.ceil(train_samples / args.gradient_accumulation)
    )
    estimated_total_steps = max(1, math.ceil(steps_per_epoch * epochs))
    interval = "100" if args.full else "10"
    command = [
        swift,
        "sft",
        "--model",
        str(model),
        "--model_type",
        "qwen3_vl",
        "--dataset",
        str(train_json),
        "--val_dataset",
        str(val_json),
        "--tuner_type",
        "lora",
        "--torch_dtype",
        "bfloat16",
        "--num_train_epochs",
        str(epochs),
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "1",
        "--learning_rate",
        str(args.learning_rate),
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--target_modules",
        "all-linear",
        "--freeze_vit",
        "true",
        "--freeze_aligner",
        "true",
        "--gradient_checkpointing",
        "true",
        "--vit_gradient_checkpointing",
        "false",
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation),
        "--eval_strategy",
        "steps",
        "--eval_steps",
        interval,
        "--save_steps",
        interval,
        "--save_total_limit",
        "2",
        "--logging_steps",
        "1",
        "--max_length",
        "4096",
        "--warmup_ratio",
        "0.05",
        "--dataset_num_proc",
        "2",
        "--dataloader_num_workers",
        "2",
        "--report_to",
        "tensorboard",
        "--logging_dir",
        str(logging_dir),
        "--output_dir",
        str(output_dir),
    ]

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment["IMAGE_MAX_TOKEN_NUM"] = str(args.image_tokens)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMP_NUM_THREADS"] = "4"

    tensorboard_process = start_tensorboard(logging_dir) if args.tensorboard else None
    log_path = output_dir / "training_console.log"
    print("Starting training:")
    print(" ".join(command))
    print(f"Console log: {log_path}")
    print(f"Live chart: {visual_dir / 'training_live.png'}")
    print(
        f"Progress estimate: {estimated_total_steps} optimizer steps "
        f"({train_samples} samples, {epochs:g} epoch(s))"
    )

    steps: list[int] = []
    losses: list[float] = []
    learning_rates: list[float] = []
    memory_values: list[float | None] = []
    recent_output: deque[str] = deque(maxlen=30)

    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    progress = tqdm(
        total=estimated_total_steps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.1,
    )

    def stop_child(_signum=None, _frame=None) -> None:
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGINT, stop_child)
    signal.signal(signal.SIGTERM, stop_child)

    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            assert process.stdout is not None
            for raw_line in process.stdout:
                log_file.write(raw_line)
                log_file.flush()
                clean_line = ANSI_RE.sub("", raw_line)
                recent_output.append(clean_line.rstrip())

                total_match = METRIC_PATTERNS["total_steps"].search(clean_line)
                if total_match:
                    detected_total = int(total_match.group(1).replace(",", ""))
                    if detected_total > 0:
                        progress.total = detected_total
                        progress.refresh()

                loss_match = METRIC_PATTERNS["loss"].search(clean_line)
                if not loss_match:
                    display_line = clean_line.strip()
                    if display_line and "%|" not in display_line:
                        progress.write(display_line)
                    continue
                try:
                    loss = float(loss_match.group(1))
                except ValueError:
                    continue

                lr_match = METRIC_PATTERNS["learning_rate"].search(clean_line)
                epoch_match = METRIC_PATTERNS["epoch"].search(clean_line)
                steps.append(len(steps) + 1)
                losses.append(loss)
                learning_rates.append(
                    float(lr_match.group(1)) if lr_match else float("nan")
                )
                memory = gpu_memory_gib(args.gpu)
                memory_values.append(memory)
                if progress.n < len(steps):
                    progress.update(len(steps) - progress.n)
                postfix = {"loss": f"{loss:.4f}"}
                if epoch_match:
                    postfix["epoch"] = epoch_match.group(1)
                if memory is not None:
                    postfix["GPU"] = f"{memory:.1f}GiB"
                progress.set_postfix(postfix, refresh=True)

                should_refresh_chart = (
                    len(steps) == 1
                    or len(steps) % max(1, args.chart_every) == 0
                    or len(steps) >= progress.total
                )
                if should_refresh_chart:
                    try:
                        update_live_chart(
                            visual_dir / "training_live.png",
                            steps,
                            losses,
                            learning_rates,
                            memory_values,
                        )
                    except Exception as error:
                        progress.write(
                            f"[monitor warning] Could not refresh PNG chart: {error}"
                        )
    finally:
        return_code = process.wait()
        if steps:
            try:
                update_live_chart(
                    visual_dir / "training_live.png",
                    steps,
                    losses,
                    learning_rates,
                    memory_values,
                )
            except Exception as error:
                progress.write(
                    f"[monitor warning] Could not write final PNG chart: {error}"
                )
        progress.close()
        if tensorboard_process is not None and tensorboard_process.poll() is None:
            tensorboard_process.terminate()

    if return_code != 0:
        print("\nTraining failed. Last output lines:", file=sys.stderr)
        for line in recent_output:
            print(line, file=sys.stderr)
        return return_code

    print("\nTraining completed successfully.")
    print(f"Checkpoints and logs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
