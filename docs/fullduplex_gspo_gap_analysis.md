# 全双工 GSPO 训练缺口分析：结合 vllm-omni 现状的插件应对方案

> 输入：《全双工推理与 GSPO 训练：概念、过程与缺口》（缺口分析文档，下称"缺口文档"）
> 本文结合本仓库三层分治架构与**源码验证**，回答：6 项缺口每一项怎么补、放哪一层、
> 我们的 Omni-Flow 回放式设计已经绕过了哪些。
>
> 关联：[feature_fullduplex_omniflow.md](feature_fullduplex_omniflow.md)（回放式设计）、
> [three_layer_strategy.md](three_layer_strategy.md)（L1/L2/L3 分层规则）

---

## 一、对缺口文档的三个修正（基于源码验证）

### 修正 1：GSPO 不需要实现——verl 已内置

缺口文档把 GSPO 当作要落地的算法。实际上 verl 主干已有完整实现：

```
verl/trainer/ppo/core_algos.py:1558
@register_policy_loss("gspo")
def compute_policy_loss_gspo(..., loss_agg_mode="seq-mean-token-mean", ...)
    # See https://arxiv.org/pdf/2507.18071 for more details.
```

配置即用（`loss_type` 默认 `ppo_clip`）：

```yaml
actor_rollout_ref:
  actor:
    ppo:
      loss_type: gspo            # token-level ratio → sequence-level
```

GSPO（Qwen，[arXiv 2507.18071](https://arxiv.org/pdf/2507.18071)）的核心差异：
`ratio = exp(Σ new_logprob - Σ old_logprob)`（**序列级**），而非逐 token。
插件侧**零代码**——这不是"实现 GSPO"，是"配置 GSPO + 构造正确的 mask"。

### 修正 2：缺口 1（per-token logprob）有 verl 标准解法——不必然改 vllm-omni

缺口文档假设 old_logprob 必须来自推理引擎。但 verl 的训练循环本来就有
**训练侧重算**的路径（`recompute_log_prob`，`verl/trainer/ppo/ray_trainer.py:630`）：
rollout 引擎只出 token，训练前用 actor 对重建的轨迹做 forward 重算 old_logprob。

```
选项 A（L1，verl 标准流程）：
    rollout 出 token 流（无需 logprob）
    → 轨迹重建（依赖缺口 2/3/4 的数据）
    → actor forward 重算 old_logprob
    代价：多一次训练侧 forward + 轨迹重建必须完整精确
    收益：vllm-omni 零改动

选项 B（L3 gate patch，GP-005 候选）：
    采样点直出 logprob。采样点已定位：
      vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py:753  # listen/speak 边界决策
      vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py:787  # 内容 token
      （已有 _record_minicpmo45_duplex_generation_token 钩子 :786/:897 可挂）
    代价：改 vllm-omni 源码（L3 台账）
    收益：省一次训练 forward；old_logprob 与采样分布严格一致
```

**推荐**：MVP 用 A。理由：零侵入优先（三层分治的 L1 ≥95% 原则）；
选项 B 的正确做法是**先给 vllm-omni 上游提 PR**（采样期概率记录是合理的
训练导向特性，与 #3907 的实验性对齐），合并前不值得自己背 L3。

### 修正 3：缺口 6（权重同步）的 MVP 方案 = 我们已有的 episode 边界设计

缺口文档给出的 MVP（"同步前关所有 session，加载后重开"）正是
[feature_fullduplex_omniflow.md](feature_fullduplex_omniflow.md) §七的设计：
回放式 episode 是有限窗口，`close_session` 释放全部 KV lease →
`update_weights` → 下个 episode 新权重 `open_session`，`weight_version`
随轨迹记录供训练侧校验。**结论一致，且插件侧已有实现位置**（duplex_rollout.py
的编排层）。

需要补验证的一点：verl 的 `llm_server` `update_weights` 链路是否覆盖
duplex engine 进程。若 duplex engine 独立于 verl 的 llm server 生命周期管理，
`_vllm_omni_bridge.py` 需扩展一个 `update_weights` 转发（仍是 L1）。

---

## 二、核心输出：6 缺口 × 三层分治归置矩阵

| 缺口 | 回放式 MVP（我们当前设计） | 交互式（后续） | 判定 |
|------|--------------------------|---------------|------|
| **1. per-token logprob** | `recompute_log_prob` 训练侧重算（L1，verl 配置） | engine 直出（L3 GP-005，先提上游 PR） | **L1 可解** |
| **2. event timestamps** | 逻辑时间够用：块序号 × chunk_size = 时间轴（L1，omniflow 的 `t_range` 已覆盖） | wall-clock 需改 vllm-omni 事件消息（L3） | **回放式无需补** |
| **3. action token IDs** | 输出流提取：轨迹已分 control / text_tokens / speech_tokens（L1，`DuplexTrajectory` 已覆盖） | engine 统一标记（L3） | **L1 可解** |
| **4. cancellation token 级精度** | barge-in 由编排层注入（`barge_in_ks`），模型停止位置可从轨迹观察（L1） | `barge_in_token_index` 需改 session buffer 快照（L3） | **回放式大幅缓解** |
| **5. stage identity** | 轨迹层逻辑标记（L1）；训练侧只训 thinker = response_mask 构造问题 | event 携带 stage_id（L3，小） | **L1 可解** |
| **6. weight sync** | episode 边界对齐（L1，已设计+实现位置） | 细粒度 invalidation / rollback（L3，RFC #3745 的深水区） | **L1 可解** |

**量化结论**：回放式设计把 6 项缺口中的 **5 项归约到 L1**（第 2/4 项直接缓解，
无需补）。这正是当初选回放式做 MVP 的架构理由——现在有了缺口文档的
系统对照，这个设计决策的收益可以被显式陈述：

> 交互式路线的 6 缺口里有 4 项要动 vllm-omni 源码（L3 集中在 2/4/6）；
> 回放式把它们全部吸收进插件编排层（L1）。L3 清单维持 4 条（GP-001~004）不增长。

---

## 三、GSPO vs GRPO：全双工轨迹为什么倾向 GSPO

### 轨迹特性

| 特性 | 全双工的形态 | 对 ratio 的影响 |
|------|------------|---------------|
| 长度 | 一个 episode 数百 chunk × 每 chunk 多 token | token-level ratio 方差随长度累积，尖峰概率高 |
| 分段 | barge-in 的 epoch 切分 | 被取消段的 token 若进 loss 会污染统计 |
| 多词表 | thinker 文本 ~2万 / talker codec ~6500 | 不同词表的 logprob 尺度不可直接比较 |
| 决策稀疏 | listen/speak 控制位占比小但价值密度高 | token-mean 聚合稀释决策信号的梯度 |

GSPO 的 sequence-level ratio 天然缓解前两条（序列内 logprob 求和后取 exp，
单 token 尖峰被平滑）；配合 `loss_agg_mode="seq-mean-token-mean"`（verl
实现注明的推荐模式）处理长序列的尺度。这与 Qwen3 在 agentic 长轨迹上
选 GSPO 的动机同构。

### epoch 段级 GSPO（插件可做的增强，L1）

全双工的天然延伸：把 GSPO 的 "sequence" 定义为 **epoch 段**（barge-in 切分的
子轨迹）而非整个 episode：

- 被打断的旧 epoch 段：单独成段，reward 独立计算（打断处理 reward 只看该段）
- 每个 epoch 段内 sequence-level ratio，段间独立聚合
- 实现位置：**response_mask / 段边界构造**在轨迹 → 训练 batch 的转换层
  （我们的 trajectory builder），loss 本身仍用 verl 的 `gspo`——不用改算法代码

MVP 可以先做 episode 级（整条轨迹一个 sequence），epoch 段级作为 P1 增强。

---

## 四、最深难点：三阶段流水线的 logprob 重算链（缺口文档未展开）

缺口文档指出了"不同 stage 的 log-prob 计算方式不同"，但没有展开**重算链的
依赖结构**——这是 recompute 路线（修正 2 的选项 A）的核心成本：

```
Stage 0 thinker：重算 = 标准（重建序列 → forward → logprob）
                   └─ 输出 text token + hidden states
Stage 1 talker：  输入 = thinker 的 hidden states（不只 token！）
                   └─ 重算 talker logprob 必须先重跑 thinker forward 拿 hidden
Stage 2 code2wav：flow-matching 生成式，无离散 logprob
                   └─ 不是 action 空间，是"渲染器"——不进 policy gradient
```

三条推论：

1. **MVP 只训 thinker**（与缺口文档所述 gspo_trainer 现状一致）。talker 冻结时，
   其 codec token 来自冻结模块的采样——对 policy gradient 无贡献，
   `response_mask=0` 直接排除，**old/new logprob 都不需要算**。
2. **stage identity 的实质是 response_mask 的构造语义**：
   thinker token mask=1（action），talker token mask=0（冻结执行），
   控制位 mask=1（决策是策略核心）——这正好落在我们的
   `omniflow_dataset.build_loss_mask` 已有的分层上（控制位+内容训练位）。
3. **credit assignment 是 episode/epoch 级的**：语音质量差可能是 talker 的锅，
   但 talker 冻结时这个信号无法回传——reward 只评估"thinker 决定说什么"
   （内容 + 时机），语音自然度交给 talker 的独立评估/训练。这是有意的
   关注点分离，不是缺陷。

---

## 五、回放式 vs 交互式：缺口敏感度总结

```
                 ┌─────────────────────────────────────────┐
                 │  回放式 MVP（当前）                        │
                 │  缺口 1 → L1 recompute                    │
                 │  缺口 2 → 已覆盖（逻辑时间）                │
                 │  缺口 3 → 已覆盖（轨迹结构）                │
                 │  缺口 4 → 已缓解（编排注入）                │
                 │  缺口 5 → L1（mask 构造）                  │
                 │  缺口 6 → 已设计（episode 边界）            │
                 │  vllm-omni 改动：0                       │
                 └─────────────────────────────────────────┘
                                │ 跑通后
                                ▼
                 ┌─────────────────────────────────────────┐
                 │  交互式（后续）                            │
                 │  新增需求：用户模拟器（L1，simulator 槽位）   │
                 │  缺口 2/4/6 升级为 wall-clock / token 级    │
                 │   快照 / 细粒度 invalidation → L3          │
                 │  缺口 1 可选 engine 直出（L3，先提上游 PR）  │
                 │  vllm-omni 改动：2~4 条 L3（GP-005 起台账）  │
                 └─────────────────────────────────────────┘
```

---

## 六、更新后的落地路径

对照缺口文档的 P0/P1/P2，映射到本仓库的里程碑：

### P0 — 单 episode、单 epoch 的闭环（全部 L1）

| 项 | 位置 | 状态 |
|----|------|------|
| GSPO loss | verl 配置 `loss_type: gspo` | ✅ verl 已有，配置即用 |
| old_logprob | verl 配置 `recompute_log_prob` | ✅ verl 已有，配置即用 |
| 轨迹重建 | trajectory → 训练 batch 转换 | 🔨 omniflow_dataset 已有大半（ChunkSpec + loss_mask），需补 thinker 输入序列的 token 化 |
| action mask | response_mask 构造（thinker=1 / talker=0 / 控制位=1） | 🔨 build_loss_mask 已有分层，需对齐 verl 的 mask 语义 |
| weight sync | episode 边界 | ✅ 已设计 + duplex_rollout 已有实现位置 |
| 验证标准 | old vs new logprob 的 ratio 分布合理（无尖峰爆炸） | 待跑 |

### P1 — barge-in 段级 + 全链路验证

- epoch 段级 response_mask（段边界 = epoch 切分点）
- 打断处理 reward 只作用于被打断段（rewards.py 的 interruption 已按段算，
  需接到段级 advantage）
- 多 episode 并发（vllm-omni `max_sessions=2` 的批量限制 → 回放式串行
  episode 缓解，吞吐靠并行 client）

### P2 — 交互式 + L3 评估

- 用户模拟器（simulator 槽位，L1）
- wall-clock timestamp / barge_in_token_index / 细粒度 invalidation：
  **集中评估一次 L3 清单**——这些点位的共同形态是"vllm-omni 训练导向
  输出接口"，正确的动作是**攒一个上游 PR**（对齐 #3745 RFC 的
  trajectory contract 一节），而不是插件各自背 patch
- 采样期 logprob 直出（GP-005 候选）：上游 PR 的头号内容——
  `minicpmo_4_5_omni.py:753/787` 的 multinomial 处保存 softmax 结果，
  钩子 `_record_minicpmo45_duplex_generation_token` 已存在，改动形态友好

---

## 七、术语对照（缺口文档 ↔ 本仓库实现）

| 缺口文档术语 | 本仓库对应 |
|-------------|-----------|
| Session / Fence / Lease / epoch | vllm-omni 侧概念；插件侧轨迹记录 `weight_version` + chunk 事件 |
| ordered session events | `DuplexTrajectory.chunks`（chunk 序号即逻辑时间） |
| action token IDs | 轨迹的 control / text_tokens / speech_tokens 三字段 |
| cancellation/epoch state | `barge_in_ks` 编排注入 + 事件 `{"type": "barge_in", "epoch"}` |
| stage identity | `build_loss_mask` 的分层（thinker 训练位 / talker 冻结） |
| policy version | 轨迹的 `weight_version` 戳（episode 边界同步的依据） |
| temporal reward aggregation | `rewards.py` 四通道（decision/timeliness/interruption/content） |
| weight sync + session invalidation | episode 边界对齐（feature_fullduplex_omniflow.md §七） |

---

## 参考

- 缺口分析文档（用户提供）：《全双工推理与 GSPO 训练：概念、过程与缺口》
- GSPO：[arXiv 2507.18071](https://arxiv.org/pdf/2507.18071)（Qwen）· verl 实现 `core_algos.py:1558`
- verl `recompute_log_prob`：`verl/trainer/ppo/ray_trainer.py:630`
- vllm-omni 采样点：`minicpmo_4_5_omni.py:753/787`（决策/内容）、`higgs_audio_v3_talker.py:1487`（talker codec）
- [RFC #3745](https://github.com/vllm-project/vllm-omni/issues/3745) trajectory contract 一节（缺口 2 的事件 schema 出处）
- MiniCPM-o 4.5 论文：[arXiv 2604.27393](https://arxiv.org/abs/2604.27393)