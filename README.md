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

仓库固定了一份与 `checkpoint-8000/vocab.json` 字节一致的词表：
`assets/ctc_gop_teacher_vocab.json`。

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

### 构建 Codec 音频选择清单

结构检查通过后，使用 LibriSpeech 三个 train split，加上经过 raw JSON SNR 映射的
LibriLight `vad/small` 与 `vad/medium`。默认取出全部合格音频，每个 LibriLight speaker
最多保留 30 小时，不要求最终总时长达到固定值：

```bash
OUT="$PWD/runs/codec-audio-all-cap30h-v1"
mkdir -p "$OUT"

PYTHONPATH=src python -m xvc2_student.build_audio_manifest \
  --librispeech-root /absolute/path/to/LibriSpeech \
  --librilight-root /absolute/path/to/LibriLight \
  --output-dir "$OUT" \
  --librilight-subset small \
  --librilight-subset medium \
  --min-duration-seconds 3.2 \
  --max-duration-seconds 120 \
  --min-snr 8 \
  --max-librilight-hours-per-speaker 30 \
  --seed 1 \
  --num-workers 8 \
  2>&1 | tee "$OUT/run.log"
```

构建器读取所有入选候选的音频 header，因此最终小时数是精确求和，不是检查器的抽样估算。
如果未来需要恢复固定目标，可以显式增加 `--target-train-hours HOURS`；不传该参数时即使
最终不足 5,000 小时也会正常生成 manifest。
`--num-workers` 并行读取音频 header；10 核节点建议从 `8` 开始，共享存储压力较大时降到 `4`。
运行期间约每 5 秒打印当前子阶段、处理数、接受数、耗时和平均文件速度，并由 `tee` 保存到日志。
它输出 `train_audio.jsonl`、`validation_audio.jsonl`、`test_audio.jsonl`、`report.json` 和
`report.md`。LibriSpeech dev/test speaker 会从 LibriLight train 中排除。输出是后续提取
Student hidden、speaker target 和可选 anchor 的 source-audio selection manifest；尚不是
可直接交给 Codec DataLoader 的 cache manifest。

### 构建 Student 蒸馏 Manifest

Codec manifest 中的 LibriSpeech 已有 transcript，可以直接生成 Teacher 固定 40 类词表的
`phone_ids`。LibriLight VAD 行没有 transcript，因此构建器先读取 Codec 选中的
`subset/speaker/raw_recording_id`，再从 LibriHeavy cuts 中提取相同 raw recording 的文本切段：

```bash
CODEC_MANIFEST_DIR="$PWD/runs/codec-audio-all-cap30h-v1"
OUT="$PWD/runs/student-manifest-5000h-v1"
mkdir -p "$OUT"

PYTHONPATH=src python -m xvc2_student.build_student_manifest \
  --codec-manifest-dir "$CODEC_MANIFEST_DIR" \
  --libriheavy-root /absolute/path/to/libriheavy \
  --librilight-root /absolute/path/to/LibriLight \
  --output-dir "$OUT" \
  --target-train-hours 5000 \
  --max-librilight-hours-per-speaker 30 \
  --text-source book \
  --seed 1 \
  --num-workers 8 \
  2>&1 | tee "$OUT/run.log"
```

默认自动寻找 LibriHeavy root 下的 `libriheavy_cuts_small.jsonl.gz` 和
`libriheavy_cuts_medium.jsonl.gz`；也可以重复传入 `--libriheavy-manifest PATH` 覆盖自动发现。
候选通过临时 SQLite 和稳定哈希排序，避免把百万级 cuts 全部放入内存。输出为
`train.jsonl`、`validation.jsonl`、`test.jsonl`、`report.json` 与 `report.md`。若有效匹配
不足 5000 小时，已有结果仍会写出，但状态为 `NEEDS_ATTENTION`；使用 `--all-matched` 可取消
固定总时长目标。

`--num-workers` 使用多进程并行生成 CTC phone IDs；10 核节点建议设为 `8`，为 gzip 解压、
SQLite 和主进程保留 2 核。LibriHeavy 扫描和最终写入保持单一确定顺序，因此 worker 数变化不会
改变固定 seed 下的样本选择与 manifest 顺序。

LibriHeavy 行指向 LibriLight raw 长录音，并保留 `start_seconds` 与 `duration_seconds`。
`PhoneManifestDataset` 使用 JSONL 字节偏移进行懒加载，并按源音频采样率将这两个字段换算成
frame offset 后只读取对应 cut；不会将 65 万行 manifest 或 LibriLight raw 长录音整体载入内存。
每个 DataLoader worker 会独立打开 manifest 文件句柄。

```bash
xvc2-student-audit manifest \
  --manifest train=/path/train.jsonl \
  --manifest validation=/path/validation.jsonl

xvc2-student-audit teacher \
  --teacher /path/is24/models/checkpoint-8000 \
  --config configs/student_12x768.yaml \
  --device cuda:0

xvc2-student-audit loader \
  --manifest /path/student-manifest-2500h-v1/train.jsonl \
  --batch-size 2 \
  --num-workers 2
```

第一条检查音频可读性、采样率、时长、重复 ID、phone ID 范围和 speaker/chapter split
leakage。第二条拒绝 Git LFS pointer，并验证 Layer 20 为 1024 维、词表为 40 类以及
Teacher/Student 50 Hz 帧长一致。第三条从 manifest 开头和首个 LibriHeavy 分段附近各读取
少量样本，用于确认 JSONL 随机访问、raw 音频切片和多 worker DataLoader 均可运行。

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

四卡推荐使用按时长动态组 batch。下面的 `60` 是每张卡每个 micro-batch 的 padded audio
预算（秒），不是四卡总预算；单条音频超过预算时仍会作为单样本 batch 读取：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 NCCL_DEBUG=WARN \
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
  -m xvc2_student.train \
  --config configs/student_12x768.yaml \
  --manifest /path/student_train_manifest.jsonl \
  --teacher /path/is24/models/checkpoint-8000 \
  --output-dir runs/student-12x768-v1 \
  --max-batch-audio-seconds 60 \
  --max-batch-items 8 \
  --duration-bucket-size 2048 \
  --grad-accum 1 \
  --num-workers 1 \
  --prefetch-factor 2 \
  --teacher-attention sdpa
```

继续训练时传入 `--resume runs/.../step-XXXXXX.pt`。Checkpoint 包含模型、optimizer、
scheduler、sampler epoch/position 和 Python/Torch/CUDA RNG state。

四卡训练会对 Teacher 和 Student 同时使用 BF16 autocast，CUDA 上使用 fused AdamW，并在梯度
累积的非末尾 micro-step 通过 DDP `no_sync()` 跳过冗余 all-reduce。动态 batch 先在相近时长的
bucket 内打乱，再按 padded audio 预算组 batch，以减少随机长短样本混合造成的 padding。四个 rank
会得到相同 batch 数，sampler 的 epoch 和 batch 位置也会写入 checkpoint。

默认训练路径还会：只执行 Teacher Layer 1--20，跳过 Layer 21--24 和 Teacher CTC head；复用一次
Teacher frozen convolution 结果作为 Student frontend 输入；并允许 Teacher 使用 PyTorch SDPA。
启动时每个 rank 会用一段 0.5 秒零波形比较完整 Teacher 与 early-exit target，只有数值一致才进入
训练。以下开关分别用于回退和做消融：

```text
--teacher-attention eager          使用原始 eager attention
--full-teacher-forward             恢复完整 24-layer Teacher 和独立 Student frontend
--no-shared-frontend               保留 early exit，但让 Student 再计算一次 convolution
--skip-teacher-optimization-check  跳过启动时的完整/优化 Teacher 一致性检查
```

`--num-workers` 按 rank 计数；10 核 CPU 配合 4 个训练 rank 时从每 rank 1 个 worker 开始。日志包含
实际全局样本数、平均 global batch size、全局音频吞吐、动态 batch 参数和每个 rank 的峰值
allocated/reserved 显存。

正式训练前的 20-step 四卡门槛可直接覆盖配置中的 200000 step，不修改正式配置文件：

```bash
MANIFEST="$PWD/runs/student-manifest-2500h-v1/train.jsonl"
TEACHER=/absolute/path/to/checkpoint-8000
OUT="$PWD/runs/student-4xh100-optimized-preflight"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 NCCL_DEBUG=WARN \
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
  -m xvc2_student.train \
  --config configs/student_12x768.yaml \
  --manifest "$MANIFEST" \
  --teacher "$TEACHER" \
  --output-dir "$OUT" \
  --max-batch-audio-seconds 60 \
  --max-batch-items 8 \
  --duration-bucket-size 2048 \
  --grad-accum 1 \
  --num-workers 1 \
  --prefetch-factor 2 \
  --teacher-attention sdpa \
  --max-steps 20 \
  2>&1 | tee "$OUT/train.log"
```

动态 batch 的实际 global batch size 随时长变化，不再固定为 16；日志中的
`mean_global_batch_size` 给出每个统计窗口的实际均值。20-step 只用于速度和显存检查，因为
`--max-steps 20` 也会让学习率调度器在第 20 步降到零，不能据此判断正式训练收敛。

## Checkpoint Validation

训练完成后，用四卡一次验证目录下全部 `step-*.pt`。Teacher target 对每个 validation batch
只计算一次，随后依次运行全部 Student checkpoint；validation 样本按 rank 无重复切分：

```bash
BASE=/inspire/hdd2/project/multilingualspeechrecognition/chenxie-25019/qixiangxu
CHECKPOINT_DIR="$PWD/runs/student-12x768-2500h-3epoch-v1"
MANIFEST="$PWD/runs/student-manifest-2500h-v1/validation.jsonl"
TEACHER="$BASE/models/checkpoint-8000"
OUT="$CHECKPOINT_DIR/validation"

mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=1 \
NCCL_DEBUG=WARN \
PYTHONUNBUFFERED=1 \
PYTHONPATH=src \
torchrun --standalone --nproc_per_node=4 \
  -m xvc2_student.validate \
  --config configs/student_12x768.yaml \
  --manifest "$MANIFEST" \
  --teacher "$TEACHER" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output-dir "$OUT" \
  --max-batch-audio-seconds 180 \
  --max-batch-items 32 \
  --num-workers 2 \
  --prefetch-factor 2 \
  --teacher-attention sdpa \
  2>&1 | tee "$OUT/validate.log"
```

输出 `report.json` 和 `report.md`，先用完整 Teacher forward 报告 Teacher CTC loss、greedy CTC
phone error rate (PER) 和整句 phone exact-match，再逐 checkpoint 报告：Teacher Layer 20 feature
loss、phone CTC loss、`feature + 0.1 * CTC`、PER 和整句 phone exact-match。
其中 feature loss 按全部有效帧聚合，CTC loss 按全部 utterance 聚合，PER 按全部 reference
phone 聚合，不对 batch 均值做二次平均。报告分别标记 weighted total loss 最低和 PER 最低的
checkpoint；两者不一致时应保留两者进入后续 Codec/streaming downstream 验证，而不是只凭一个
指标删除 checkpoint。

## Flow-OPD 结论

详见 `docs/FLOW_OPD_ASSESSMENT.md`。Flow-OPD 不直接适用于当前异构的 Wav2Vec2
Teacher -> Emformer Student 表征蒸馏，因此 v1 没有引入 Flow/SDE、GRPO 或 PPO。

## 当前边界

- 已实现模型、训练、DDP、AMP、checkpoint 和流式推理接口。
- `streaming_consistency_weight` 保留在配置合同中，但 v1 训练入口要求为 0；它需要先完成
  严格的 prefix target causality audit，再单独实现和验证。
- 正式训练前仍需运行 960h manifest audit、单卡/多卡显存 benchmark 和
  full/chunk/reset/flush acceptance。
