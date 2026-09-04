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

