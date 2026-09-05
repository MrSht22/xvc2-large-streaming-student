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

推荐使用 Python 3.10。先安装与目标 GPU 匹配的 PyTorch CUDA wheel，再安装其余固定依赖
和本仓库：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchaudio==2.4.1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
xvc2-student-smoke
xvc2-student-smoke --config configs/student_12x768.yaml
```

以上是已经通过 Teacher audit 的 CUDA 12.1 组合。若新 GPU 必须使用其他 CUDA wheel，先替换
第一条命令中的 PyTorch index，但必须保持 `torch` 和 `torchaudio` 的版本完全一致。系统还需要
可工作的 NVIDIA driver；Conda 环境不需要单独安装完整 CUDA Toolkit。

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

### LibriHeavy 仓库检查

在生成正式 manifest 前，先检查 LibriHeavy/LibriLight 的实际落盘结构：

```bash
xvc2-student-inspect-libriheavy \
  --root /path/to/libriheavy \
  --audio-root /path/to/librilight \
  --output-dir runs/libriheavy-inspection
```

脚本只读取目录、Lhotse `jsonl/jsonl.gz` 和音频路径，不解码或修改音频。默认每个 manifest
抽样 2,000 行，输出 `report.json` 与 `report.md`，用于决定 split、文本字段和音频路径映射。

### LibriSpeech 与 LibriLight 原始数据检查

只有两个原始数据集路径、尚不清楚目录和时长时，运行：

```bash
OUT=~/X-VC2/runs/speech-corpora-inspection
mkdir -p "$OUT"

PYTHONPATH=src python -m xvc2_student.inspect_audio_corpora \
  --librispeech-root /absolute/path/to/LibriSpeech \
  --librilight-root /absolute/path/to/librilight \
  --output-dir "$OUT" \
  --max-files-per-corpus 2000000 \
  --metadata-samples-per-group 500 \
  2>&1 | tee "$OUT/run.log"
```

脚本完整遍历文件名以统计实际文件数、磁盘大小、split、speaker 和 chapter/book 分布，但每个
split 默认只随机抽样 500 个音频 header 来估算时长，不解码波形。`report.json` 保留路径和文本
示例，`report.md` 给出紧凑汇总；若某个 split 的全部音频都被抽样，其时长会标记为精确 metadata
求和，否则明确标记为抽样估算。把 `report.json`、`report.md` 和 `run.log` 发回后，再据此确定
正式 manifest 的字段、split、文本来源和采样比例。

若 LibriLight 同时包含 `raw/` 长录音和 `vad/` 切段，报告会把它们视为同一语料的两种
representation，分别统计但不会相加为总时长；伴随 raw 音频的 JSON 会抽样解析字段结构。

### 构建 5,000 小时 Codec 音频选择清单

结构检查通过后，使用 LibriSpeech 三个 train split，加上经过 raw JSON SNR 映射的
LibriLight `vad/small` 与 `vad/medium`，累计到至少 5,000 小时：

```bash
OUT="$PWD/runs/codec-audio-5000h-v1"
mkdir -p "$OUT"

PYTHONPATH=src python -m xvc2_student.build_audio_manifest \
  --librispeech-root /absolute/path/to/LibriSpeech \
  --librilight-root /absolute/path/to/LibriLight \
  --output-dir "$OUT" \
  --target-train-hours 5000 \
  --librilight-subset small \
  --librilight-subset medium \
  --min-duration-seconds 3.2 \
  --max-duration-seconds 120 \
  --min-snr 8 \
  --max-librilight-hours-per-speaker 20 \
  --seed 1 \
  --num-workers 8 \
  2>&1 | tee "$OUT/run.log"
```

构建器读取所有入选候选的音频 header，因此这里的小时数是精确求和，不是检查器的抽样估算；
最终总量最多比目标多一个 VAD 切段，超出秒数写入 `report.json`。
`--num-workers` 并行读取音频 header；10 核节点建议从 `8` 开始，共享存储压力较大时降到 `4`。
它输出 `train_audio.jsonl`、`validation_audio.jsonl`、`test_audio.jsonl`、`report.json` 和
`report.md`。LibriSpeech dev/test speaker 会从 LibriLight train 中排除。输出是后续提取
Student hidden、speaker target 和可选 anchor 的 source-audio selection manifest；尚不是
可直接交给 Codec DataLoader 的 cache manifest。

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
