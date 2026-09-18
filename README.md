# Qwen3-VL-8B 三模态视觉定位实验

本项目用 Qwen3-VL-8B 和 LoRA，根据文字描述在图像中定位目标。输入为对齐的可见光 RGB、热红外和深度图，模型输出目标在 RGB 图中的边界框。训练和验证主要在 AutoDL 的 `/root/autodl-tmp` 下运行；本仓库保存数据准备、训练、验证及提交脚本，不包含模型权重或完整数据集。

## 当前状态

- 已准备 MID-3K、RGBDT500 样本，并完成多轮 LoRA 训练与验证。
- 榜单提交成绩约为 71%。榜单只提供总体分数，不能由此判断隐藏测试集的单题对错。
- 历史验证成绩来自不同数据集、样本数与图像 token 设置，不能直接拿绝对值相互比较。比较两个 checkpoint 时，必须使用同一份验证集和相同的推理参数。

## 环境与目录

建议沿用已跑通的 AutoDL GPU 环境。脚本使用 Python、PyTorch、MS-SWIFT、NumPy、Pillow、Matplotlib、tqdm 和 TensorBoard；缺少依赖时可在现有 PyTorch 环境中安装：

```bash
python -m pip install "ms-swift>=4.0" numpy pillow matplotlib tqdm tensorboard
```

以下命令假定脚本和数据位于 `/root/autodl-tmp`，模型目录为 `/root/autodl-tmp/models`：

```text
/root/autodl-tmp/
├── models/                    # Qwen3-VL-8B 基座模型
├── datasets/                  # MID-3K 等原始数据
├── RGBDT500_subset50/         # RGBDT500 抽样数据，按序列组织
├── prepared_mid3k/            # train.jsonl、val.jsonl、深度可视化图
├── prepared_rgbdt500/         # train.jsonl、val.jsonl、描述与检查图
├── competition_test/          # 官方测试图像及 queries/queries.json
└── output/                    # LoRA checkpoint、日志与 TensorBoard
```

原始数据、模型和 checkpoint 不在本仓库内；换机器时需要将它们和所用脚本一同迁移。JSONL 中使用绝对图像路径，换机器后还需确认这些路径仍然有效。

## 数据准备

### MID-3K

将三个模态放在 `datasets/MID-3K-rgb`、`datasets/MID-3K-thermal`、`datasets/MID-3K-depth` 后运行：

```bash
python check_mid3k.py --root /root/autodl-tmp/datasets
python prepare_mid3k.py
```

`prepare_mid3k.py` 默认生成 1800 条训练记录、200 条验证记录，并将原始深度转换为供视觉模型读取的 RGB 图。其查询主要由行人左右顺序生成，不代表官方测试集全部目标类型。

### RGBDT500 抽样数据

如果已有按序列解压的 `RGBDT500_subset50`，依次审核标注、生成描述、人工检查描述，再转换为训练格式：

```bash
python prepare_rgbdt500.py audit
python prepare_rgbdt500.py caption
# 检查 prepared_rgbdt500/caption_review_*.jpg，必要时修改 captions.csv
python prepare_rgbdt500.py prepare
```

训练与验证按**序列**划分，避免相邻帧同时出现在两边。不能跳过描述检查：模型生成的目标名称可能与框内物体不符。要从大型原始 `Train.zip` 制作较小上传包，可查看 `sample_rgbdt500_zip.py --help`；已上传的子集无需重新抽取。

### JSONL 格式

每行是一条独立 JSON 记录，包含按 RGB、热红外、深度顺序排列的 `images`，以及用户查询和目标框：

```json
{"images":["/path/rgb.png","/path/ir.png","/path/depth_visual.png"],"messages":[{"role":"user","content":"<image>\nImage 1 is the visible RGB image.\n<image>\nImage 2 is the aligned thermal infrared image.\n<image>\nImage 3 is the aligned depth map; brighter valid pixels are closer and black pixels may be invalid.\nLocate the following target in Image 1: the leftmost person in the scene.\nReturn only JSON in this format: {\"bbox_2d\":[x1,y1,x2,y2]}. Coordinates must be integers from 0 to 1000."},{"role":"assistant","content":"{\"bbox_2d\":[100,200,300,600]}"}]}
```

这里的 `bbox_2d` 是相对图像宽高映射到 **0–1000** 的 `[x1,y1,x2,y2]`。提交脚本会再转换为官方查询文件要求的 **0–1** 坐标。

## 训练与监控

先用少量数据确认流程可运行；不加 `--full` 时，脚本只取 100 条训练和 50 条验证记录：

```bash
python train_qwen3vl8b.py \
  --train-json /root/autodl-tmp/prepared_mid3k/train.jsonl \
  --val-json /root/autodl-tmp/prepared_mid3k/val.jsonl \
  --output-dir /root/autodl-tmp/output/mid3k_smoke
```

完整训练需显式加入 `--full`，并建议给每次实验单独的输出目录：

```bash
python train_qwen3vl8b.py --full --epochs 1 \
  --train-json /root/autodl-tmp/prepared_mid3k/train.jsonl \
  --val-json /root/autodl-tmp/prepared_mid3k/val.jsonl \
  --output-dir /root/autodl-tmp/output/mid3k_run
```

如需混合 MID-3K 与 RGBDT500 训练样本，先生成新 JSONL，再使用它训练：

```bash
python build_mix_v2.py \
  --mid3k /root/autodl-tmp/prepared_mid3k/train.jsonl \
  --rgbdt /root/autodl-tmp/prepared_rgbdt500/train.jsonl \
  --output /root/autodl-tmp/prepared_mix_v2/train.jsonl
```

`--adapters /path/to/checkpoint-N` 可使用已有 LoRA 权重初始化下一阶段训练；这不是恢复优化器状态的断点续训。训练终端有进度条，输出目录中有 `training_console.log`、`tensorboard/`、`visuals/training_live.png` 和 `visuals/live_dashboard.html`。脚本默认尝试启动 TensorBoard（端口 6006）；端口不可用时可加 `--no-tensorboard`，之后自行查看日志。检查训练结束信息和 checkpoint 文件，不能只凭目录存在判断训练成功。

## 验证与比较

使用指定 checkpoint 验证；不要依赖 `infer_qwen3vl8b.py` 内置的旧 checkpoint 默认路径：

```bash
python infer_qwen3vl8b.py --mode val \
  --val-json /root/autodl-tmp/prepared_mid3k/val.jsonl \
  --adapter /root/autodl-tmp/output/mid3k_run/<版本目录>/checkpoint-N \
  --output-dir /root/autodl-tmp/eval_mid3k_run
```

验证输出包含 `val_metrics.json`、`val_predictions.jsonl` 和 `val_preview.png`。预测文件逐条保存文字查询、真值框、预测框和 IoU；按本项目的 `Acc@0.5` 口径，`iou >= 0.5` 算命中。若要列出最差的 20 条验证记录：

```bash
python -c "import json; p='/root/autodl-tmp/eval_mid3k_run/val_predictions.jsonl'; a=[json.loads(s) for s in open(p)]; [print(x['id'], x['iou'], x['query']) for x in sorted(a,key=lambda x:x['iou'])[:20]]"
```

比较两次实验时，让它们都在**同一份** `val.jsonl` 上完成推理，然后运行：

```bash
python compare_grounding_runs.py \
  --dataset /root/autodl-tmp/prepared_mid3k/val.jsonl \
  --baseline /root/autodl-tmp/eval_old/val_predictions.jsonl \
  --candidate /root/autodl-tmp/eval_new/val_predictions.jsonl
```

公开验证集能定位具体错误；官方隐藏测试集没有真值框，不能从提交结果反推出哪些题错了。不同验证集的 Acc@0.5 也不能直接与榜单分数等同。

## 官方测试与提交

确认官方数据在 `/root/autodl-tmp/competition_test`，并保留 `queries/queries.json` 的原始结构。先做 5 条试跑：

```bash
python infer_qwen3vl8b.py --mode test \
  --test-root /root/autodl-tmp/competition_test \
  --adapter /root/autodl-tmp/output/mid3k_run/<版本目录>/checkpoint-N \
  --output-dir /root/autodl-tmp/inference_submission_trial \
  --limit 5
```

试跑产生的 `partial_first_5.json` **不能提交**。确认图像、输出框和文件结构无误后，在同一个输出目录去掉 `--limit`，脚本会利用 `inference_state.jsonl` 继续完成剩余查询：

```bash
python infer_qwen3vl8b.py --mode test \
  --test-root /root/autodl-tmp/competition_test \
  --adapter /root/autodl-tmp/output/mid3k_run/<版本目录>/checkpoint-N \
  --output-dir /root/autodl-tmp/inference_submission_trial
```

全部完成后脚本生成 `queries.json` 与 `submission.zip`；压缩包内为 `queries.json`。更换 checkpoint、图像 token 数或深度参数时，请使用**新的输出目录**，避免混用旧推理状态。提交前核对记录数、解析失败数，并确认竞赛平台当前要求的压缩包格式。

## 注意事项

- 官方测试数据仅用于推理与提交；不要把它加入训练或自行当成有真值的验证集。
- RGBDT500 是跟踪数据，相邻帧高度相似；增大帧数不等于增加同样多的独立场景。
- MID-3K 主要提供行人框和规则生成的描述，不能单独代表复杂的开放类别指代定位。
- 本 README 中的目录名和 checkpoint-N 是示例；实际训练版本目录由运行时产生。
- 不要为运行本项目批量删除原始数据或 checkpoint。新实验使用新输出目录即可。
