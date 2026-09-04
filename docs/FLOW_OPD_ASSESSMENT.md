# Flow-OPD 对 X-VC2 Student 蒸馏的适用性

日期：2026-09-04。

审阅材料：

- `2605_FlowOPD.pdf`，arXiv:2605.08063v5，2026-05-24；
- 官方仓库 `CostaliyA/Flow-OPD`，审阅 commit
  `434ab16911fc580f51e22107151915846222eb6a`。

## Flow-OPD 实际做什么

Flow-OPD 是 Flow Matching 文生图模型的多专家 post-training 方法：

1. 用单任务 GRPO 得到多个同构专家；
2. 从 SFT 或模型合并得到 Student cold start；
3. Student 通过 SDE 在自己的当前策略分布上生成轨迹；
4. 每个轨迹状态由任务路由到对应专家，专家返回 dense velocity field；
5. 在逐 timestep transition 上计算 Student/Teacher 高斯策略的 Reverse-KL；
6. 由于协方差相同，KL 化为带时间权重的 velocity-field MSE；
7. 使用额外 aesthetic teacher 的 MAR loss 防止专项能力训练破坏整体生成质量。

官方实现对应的是 SD3.5 base + 多个 LoRA expert，关键配置包括
`kl_ref_lora_path`、`reward_mode=kl_only/gkd`、`kl_reward_level=step_wise` 和
`mar_lora`。训练代码直接比较 Student 与 reference 在同一 SDE transition 上的
`prev_sample_mean`。

## 为什么不能直接套用

当前 X-VC2 蒸馏是：

```text
offline Wav2Vec2 Teacher Layer 20
-> deterministic 50 Hz target
-> limited-lookahead Emformer Student
```

它没有 Flow Matching velocity field、SDE rollout、可计算 transition probability 的生成
policy，也没有同构多专家。论文附录还明确把 Teacher/Student 架构同质性列为细粒度
step-wise supervision 的要求。因此：

- 把 Layer 20 SmoothL1 改名为 Flow-OPD 不成立；
- 为了使用其公式而额外建立 Flow/SDE policy 会改变任务本身；
- 直接引入 PPO/GRPO 会增加显存、方差和工程复杂度，却没有对应的 on-policy action；
- offline Teacher 对短 prefix 的 target 可能使用未来上下文，未经 causality audit 不能当作
  合法 streaming supervision。

## 可以迁移的思想

1. **Student-state coverage**：后续可以在 Student 真实 chunk/cache 状态上采样，减少只在
   full-utterance 输入上训练造成的 exposure gap。
2. **Dense expert supervision**：如果未来有 phone、prosody、prefix-stability 等多个同构
   streaming expert，可按样本类型路由，而不是把稀疏总分混成一个 reward。
3. **Manifold anchor**：在专项微调时保留旧 Student 的 phone posterior 或 hidden behavior
   anchor，防止改善一个指标时破坏内容与流式稳定性。
4. **Cold start**：先完成标准 Layer 20 + CTC 蒸馏，再从合格 checkpoint 开始任何
   on-policy streaming-state 实验，不从随机模型开始。

## 当前决定

Large Student v1 使用：

```text
Layer 20 normalized SmoothL1 + 0.1 * phone CTC
```

Flow-OPD 不进入正式训练代码。后续只有在以下前置条件全部满足时才建立单独分支：

- 定义可验证的 Student on-policy state；
- Teacher target 通过 prefix causality audit；
- 明确 anchor teacher 与专项 teacher；
- 与标准蒸馏在完全相同 manifest、训练量和 streaming gate 下比较。

