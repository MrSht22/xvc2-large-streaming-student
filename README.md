# X-VC2 Large Streaming Student

独立训练一个 50 Hz、40 ms 右上下文的 12-layer 768-d Emformer Student，用冻结的
CTC-GOP Wav2Vec2 Teacher Layer 20 和 phone CTC 监督。

## 模型合同

```text
16 kHz waveform
-> frozen Teacher convolutional frontend
-> 768-d projection
-> 12-layer Emformer, 12 heads, FFN 3072
-> 768-d hidden at 50 Hz
   |- 1024-d Teacher distillation projection (training only)
   `- 40-class phone CTC head
```

默认模型参数量：

```text
total      90,497,832
trainable  86,287,656
```

默认配置位于 `configs/student_12x768.yaml`。旧 8x512 checkpoint 只作为 matched
baseline，不用于恢复新模型权重。

## 安装与 Smoke

```bash
python -m pip install -e '.[dev]'
xvc2-student-smoke
xvc2-student-smoke --config configs/student_12x768.yaml
```

已有 `ctc-gop` 环境先运行：

```bash
xvc2-student-env-check --require-cuda
```

只有输出 `ctc_gop_student_environment=PASS` 才可直接复用。该检查覆盖依赖版本、
Torch/torchaudio minor version、CUDA 可见性、Emformer 和 Wav2Vec2FeatureEncoder API。
`cmudict` 不属于本仓库的训练期依赖：manifest 中的 phone IDs 必须在训练前预先生成。

第二条命令会实例化完整 90M 模型并执行一次合成前向/反向，CPU 会较慢。

## Manifest

训练 manifest 为 JSONL，phone ID 必须使用 Teacher 的 40 类词表，其中 `0` 为 blank：

```json
{
  "utterance_id": "84-121123-0001",
  "audio_path": "/absolute/path/84-121123-0001.wav",
  "phone_ids": [12, 8, 19, 7]
}
```

G2P 和数据审计应在训练前完成，不在 DataLoader 内动态访问网络或下载 NLTK 资源。

```bash
xvc2-student-audit manifest \
  --manifest train=/path/train.jsonl \
  --manifest validation=/path/validation.jsonl

xvc2-student-audit teacher \
  --teacher /path/is24/models/checkpoint-8000 \
  --config configs/student_12x768.yaml \
  --device cuda:0
```

第一条检查音频可读性、采样率、时长、重复 ID、phone ID 范围和 speaker/chapter split
leakage。第二条拒绝 Git LFS pointer，并验证 Layer 20 为 1024 维、词表为 40 类以及
Teacher/Student 50 Hz 帧长一致。

## 资源 Benchmark

单卡与两卡 DDP 分别运行：

```bash
CUDA_VISIBLE_DEVICES=0 xvc2-student-benchmark \
  --config configs/student_12x768.yaml --batch-size 1 --audio-seconds 3.2

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m xvc2_student.benchmark \
  --config configs/student_12x768.yaml --batch-size 1 --audio-seconds 3.2
```

输出包含每步耗时、global audio seconds/second 和每 rank 峰值显存。它使用合成 Teacher
target 和 CTC target，只验证工程与资源，不代表真实蒸馏收敛质量。

## 训练

单卡：

```bash
python -m xvc2_student.train \
  --config configs/student_12x768.yaml \
  --manifest /path/student_train_manifest.jsonl \
  --teacher /path/is24/models/checkpoint-8000 \
  --output-dir runs/student-12x768-v1 \
  --batch-size 2 \
  --grad-accum 8 \
  --num-workers 0
```

四卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m xvc2_student.train \
  --config configs/student_12x768.yaml \
  --manifest /path/student_train_manifest.jsonl \
  --teacher /path/is24/models/checkpoint-8000 \
  --output-dir runs/student-12x768-v1 \
  --batch-size 1 \
  --grad-accum 8 \
  --num-workers 0
```

继续训练时传入 `--resume runs/.../step-XXXXXX.pt`。Checkpoint 包含模型、optimizer、
scheduler、sampler epoch/position 和 Python/Torch/CUDA RNG state。

## Flow-OPD 结论

详见 `docs/FLOW_OPD_ASSESSMENT.md`。Flow-OPD 不直接适用于当前异构的 Wav2Vec2
Teacher -> Emformer Student 表征蒸馏，因此 v1 没有引入 Flow/SDE、GRPO 或 PPO。

## 当前边界

- 已实现模型、训练、DDP、AMP、checkpoint 和流式推理接口。
- `streaming_consistency_weight` 保留在配置合同中，但 v1 训练入口要求为 0；它需要先完成
  严格的 prefix target causality audit，再单独实现和验证。
- 正式训练前仍需运行 960h manifest audit、单卡/多卡显存 benchmark 和
  full/chunk/reset/flush acceptance。
