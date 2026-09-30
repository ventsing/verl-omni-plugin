# 插件数据通路协议：六个接缝的契约全集

> 数据在 verl-omni-ext 里走完整条 RL 旅程要跨六个接缝，每个接缝都有
> 明确的输入/输出契约。本文档是这些契约的权威参考——字段级定义、
> 跨接缝不变量、已验证 vs 待接线（TODO）部分。
>
> 全部结构基于当前代码（`22c2bd7`），上游协议引用 vllm-omni 本地
> checkout 源码行号。
>
> 关联：[feature_fullduplex_omniflow.md](feature_fullduplex_omniflow.md)（设计）、
> [fullduplex_gspo_gap_analysis.md](fullduplex_gspo_gap_analysis.md)（缺口）、
> [vllm_omni_deployment_and_weight_transform.md](vllm_omni_deployment_and_weight_transform.md)（权重同步）

---

## 0. 总览：一条数据的旅程

```
┌─存储协议─┐   ┌─A 数据集协议─┐   ┌─C 会话协议─┐   ┌─D WS 线上协议─┐
│ parquet  │ → │ 槽位④collate │ → │ DuplexSession│ → │ /v1/duplex    │
│ episode行│   │ ChunkSpec序列│   │ Client 4方法 │   │ 事件流(桥接)  │
└──────────┘   └──────────────┘   └──────┬───────┘   └───────────────┘
                   B 逻辑视图              │    ↑ 帧协议常量
                                          ▼    └── vllm-omni 引擎
                                    ┌─E 轨迹协议─┐
                                    │ DuplexTraj │
                                    │ +GRPO group│
                                    └─────┬──────┘
                        ┌─F reward 协议───┤
                        │ custom_reward_  │   ┌─训练侧重放────────┐
                        │ function/manager│   │ recompute_log_prob│
                        └─────────────────┘   │ response_mask     │
                                              └───────────────────┘
```

两种运行模式共用 A/B 协议：
- **SFT 模式**：A 的 `loss_mask` 直接进训练（dataset → actor forward）
- **RL 模式**：C/D/E/F 走 rollout（环境回放 → 轨迹 → reward），
  训练侧重算 logprob（不依赖 D 的采样期输出）

---

## 1. 存储协议：episode parquet 行

**定义处**：`omniflow_dataset.make_episode_row` / `serialize_episode` 的 docstring

```python
{
  "episode_id": str,                      # 唯一标识
  "streams": {
    "visual": [ {"t_start": float, "t_end": float, "frames": [...]} ],
    "audio":  [ {"t_start": float, "t_end": float,
                 "samples": [..] | bytes→自动 base64 为 samples_b64} ],
    "text":   [ {"t_start": float, "t_end": float,
                 "tokens": [str], "token_times": [float]?} ],  # token_times 可选：逐 token 时间戳
    "speech": [ {"t_start": float, "t_end": float, "tokens": [...]} ],  # t_start = 对应文本的起始
  },
  "meta": {...},                          # 透传（reward weights 等配置可放这）
  # 可选：barge_in_ks: [int]              # 数据管道预计算的打断块（优于默认推导）
}
```

**时间语义**（易错点）：
- `t_end` 是**闭区间右端点**——时长 1.0s @0.2s chunk = 5 块，不是 6 块
- 块归属按 **start time**（`floor(t/chunk + 1e-9)`，epsilon 抵消浮点边界）
- `speech[].t_start` 语义特殊：指向**所属文本段**的起始时间（不是 speech 自身）——TAIL 规则依据

---

## 2. 接缝 A→B：数据集协议（槽位④ `data.custom_cls`）

**上游契约**（verl-omni `rl_dataset.py:66-81`）：`custom_cls.path`
（pkg:// 支持）+ `custom_cls.collate_fn`（可调用对象名，`load_extern_object` 加载）。

**config 声明**：
```yaml
data:
  custom_cls:
    path: pkg://verl_omni_ext.features.fullduplex.omniflow_dataset
    collate_fn: omniflow_collate_fn
```

**collate 输入**：parquet 行列表（原始 episode 行，未 token 化）

**collate 输出**（`omniflow_collate_fn`）：
```python
{
  "episodes": [
    {
      "episode_id": str,
      "group_id": str,            # GRPO group，默认 = episode_id
      "chunks": [ChunkSpec, ...], # B 逻辑视图（见下）
      "loss_mask": [bool, ...],   # 输出侧逐逻辑位掩码
      "meta": {...},
    }
  ],
  "batch_size": int,
  "chunk_size_s": float,
  "omniflow": True,               # 模式标记：下游识别这是全双工样本
}
```

**设计约束**：collate 在 dataloader worker 里跑——**tokenizer 无关**
（`<|listen|>` 保持字符串，token 化由 thinker_adapter 在模型侧完成）。

### B：ChunkSpec 逻辑视图（g_k = [v_k; a_k; o_k]）

| 字段 | 类型 | 语义 | 训练位 |
|------|------|------|--------|
| `k` | int | 块序号（0-based） | — |
| `t_start`/`t_end` | float | 块时间窗 | — |
| `visual_frames` | list | v_k：视觉帧（processor 消费） | 否（输入侧） |
| `audio_chunks` | list | a_k：音频段 | 否（输入侧） |
| `control` | str | `<|listen|>` / `<|speak|>`（LS 控制位） | **是** |
| `text_tokens` | list[str] | 本块文本 token | **是** |
| `speech_tokens` | list | TAIL 归属本块的 speech token | **是**（SFT） |
| `events` | list[dict] | rollout 期填充；SFT 数据为空 | — |

**序列化规则**（`serialize_episode`，论文 §3 的实现）：
1. 块数 = `ceil(t_max/chunk - 1e-9)`（覆盖全部流的最晚 t_end）
2. v_k/a_k/text：按 start time 归块
3. speech：**TAIL 有界 look-ahead**——文本段起点落在块尾 lookahead 窗口
   （相对位置 ≥ `1 - lookahead/len(block_tokens)`）时，speech 推迟到下一块
4. 控制位：块内有文本 → `<|speak|>`，否则 `<|listen|>`
   （speech 不触发控制位——它是 text 的播放形态）

**loss_mask 语义**（`build_loss_mask`）：对每块展开序列
`[控制位, text…, speech…]` 逐位给 True；输入侧（v_k/a_k token 展开）
不在其中——由 thinker_adapter 在构造模型输入时统一 mask。

---

## 3. 接缝 C：rollout 会话协议（DuplexSessionClient）

**定义处**：`duplex_rollout.py:47`（Python Protocol——结构化鸭子类型）

| 方法 | 签名 | 语义 |
|------|------|------|
| `open_session` | `(session_config: dict) → session_id: str` | 建会话（config 含 `model`、`chunk_period_ms`） |
| `push_chunk` | `(session_id, visual: list?, audio: Any?) → dict` | 推第 k 块环境输入，**返回模型本块输出增量** |
| `barge_in` | `(session_id, scope="current") → epoch: int` | 注入打断（epoch 递增，旧输出失效） |
| `close_session` | `(session_id) → None` | 关会话（**释放 KV lease——权重同步前提**） |

**push_chunk 返回结构**（协议核心）：
```python
{
  "control": "<|listen|>" | "<|speak|>",
  "text_tokens": [...],       # speak 时
  "speech_tokens": [...],     # TAIL 归属本块
  "events": [...],            # barge_in / playback_ack / cancel
  "latency_ms": float,
}
```

**时序契约**：`push_chunk(k)` 的输出 o_k 条件于 g_1..g_k 的全部输入
（先感知后输出，论文 §3.2 的因果结构）——天然匹配 vllm-omni 的
duplex-append（decode 时扩展 prompt 实现输入输出并发，RFC #3745）。

**实现矩阵**：
| 实现 | 用途 |
|------|------|
| `ReplayDuplexClient` | 离线单测（ground truth 模拟输出 + barge_at 注入） |
| `VllmOmniDuplexClient`（桥） | 生产：WS → /v1/duplex（**运行时接线 TODO**，见 §7） |

---

## 4. 接缝 D：WS 线上协议（vllm-omni /v1/duplex）

**上游权威定义**（vllm-omni `experimental/fullduplex/openai/websocket.py:14-78`）：

**入站事件**（INPUT_EVENTS）：
```
input.text.append          input_audio_buffer.append
input.commit               input_audio_buffer.commit
response.create
```
**别名表**（`_INPUT_EVENT_ALIASES`）：
```
push_chunk        → input_audio_buffer.append   （+默认 format=wav）
input.audio.append→ input_audio_buffer.append
input_text.append → input.text.append
close_session     → session.close
audio.playback_ack→ playback.ack
signal_turn       → turn.signal
```

**出站事件**（MODEL_OUTPUT_EVENTS，节选）：
```
response.created / response.listen / response.speak     ← 控制位可见！
response.output_text.delta / response.text.delta        ← 文本增量
response.output_audio.delta / response.audio.delta      ← 语音增量
response.done / response.output_item.done               ← 完结
runtime.control                                          ← 运行时控制
```

**终止事件**（DOMAIN_TERMINAL_EVENTS）：
`response.done`、`response.listen`、`audio.cancelled`、`input.cancelled`、`session.closed`

**桥的职责**（`_vllm_omni_bridge.py`）：
1. **帧协议换算**：`chunk_period_ms=200` → 3200 samples @16kHz = **2 个 audio
   embedding**（`SAMPLES_PER_AUDIO_TOKEN=1600`，100ms/embedding）；
   每视觉帧 `<image>` + 64 embeds + `</image>`（`VISION_EMBEDS_PER_FRAME=64`）
2. **事件流投影**：`response.speak/listen` → `control` 字段；
   `response.*.delta` 聚合 → `text_tokens`/`speech_tokens`；
   epoch 事件 → `events`
3. **能力探测**：初始化时确认 `vllm_omni.experimental.fullduplex` 可导入
   （版本无关的运行时探测，代替 import 期断言）

---

## 5. 接缝 E：轨迹协议（DuplexTrajectory + GRPO group）

**结构**（`duplex_rollout.py:145`）：
```python
DuplexTrajectory:
  episode_id: str              # 来源 episode
  group_id: str                # = f"grp_{episode_id}"（GRPO group 内共享）
  weight_version: str          # 权重版本戳——batch 内必须一致（校验点）
  chunk_size_s: float          # 与 dataset 的 chunk_size_s 同源
  session_id: str              # 关联的会话（排障用）
  chunks: [ {k, t_range, control, text_tokens,
             speech_tokens, latency_ms, events} ]   # push_chunk 返回的逐块归档
```

**GRPO group 语义**（`collect_group`）：同环境（同录制流）、**同打断时刻**
（barge_in_ks 一致）、不同采样的 N 条轨迹——group 内归一化算 advantage
（verl `grpo` 估计器或 `loss_type: gspo`）。

**编排协议**（`run_replay_episode`）：
```
open_session(cfg)
for k, env in episode_chunks:
    if k in barge_in_ks: barge_in(sid)      # 打断先于本块输入
    out = push_chunk(sid, env.visual, env.audio)
    traj.chunks.append(out)
close_session(sid)                           # KV 释放 → 允许 update_weights
```

**环境输入只用 v_k/a_k**：episode_chunks 的 text/speech 是标注（ground
truth），留给 reward——不推给模型（防泄漏）。

**barge_in_ks 来源**：数据管道标注优先；默认推导 = ground truth 控制位
speak→listen 的转换块（`_infer_barge_in_ks`）。

---

## 6. 接缝 F：reward 协议

### F1：custom_reward_function（verl 全局约定签名）

```python
def omniflow_content_score(
    data_source: str,           # 数据源名
    solution_str: str,          # 模型整个 episode 的文本输出拼接
    ground_truth: str,          # 标注文本
    extra_info: dict | None,    # 透传：chunks（轨迹）、content_weights 等
) -> float
```
verl reward manager（`__call__(data: DataProto) → Tensor|dict`）按
batch 维度调用；reward tensor 与 `data.batch["responses"]` 对齐。
纯 CPU、无外部依赖（dataloader worker 可跑）——judge model 由上层注入。

### F2：轨迹级 manager（槽位⑤ `@register("omniflow_duplex_reward")`）

**输入结构**：`trajectory_chunks`（= `DuplexTrajectory.chunks`，
经 `to_reward_input()` 直通）+ `ground_truth_chunks`（ChunkSpec 转 dict，同构）

**四通道**（`omniflow_trajectory_reward`，默认权重 0.35/0.25/0.15/0.25）：

| 通道 | 输入字段 | 语义 |
|------|---------|------|
| decision | 双方 `control` 按 `k` 对齐 | 混淆矩阵；抢话 ×1.5 罚 |
| timeliness | text/speech 与 TAIL 期望的比例偏差 | 及时性 |
| bargein_latency | barge_in 事件后的首个 listen 块（最早反应 = 下一块） | `-min(latency/400ms, 1)` 饱和罚 |
| interruption | barge_in 后 `1<=lag<=grace` 窗口内的 listen | 打断响应 |

**分项返回**：`(total, {"decision": …, "timeliness": …, …})`——训练侧可
按通道监控/调权。

---

## 7. 跨接缝不变量（违反 = 静默错位或行为破坏）

| # | 不变量 | 违反后果 |
|---|--------|---------|
| I1 | `chunk_size_s`（A）= `chunk_period_ms/1000`（C/D）= 帧换算基准 | 块数/token 数错位 |
| I2 | `SAMPLES_PER_AUDIO_TOKEN=1600` 等常量与 vllm-omni `minicpmo45/policy.py` 一致（**import 优先，本地 fallback**） | KV 里 pad embedding 破坏 listen/speak 行为 |
| I3 | TAIL 规则（B 序列化）与 rollout 期 speech token 归块（C 返回）同一实现 | reward 的 timeliness 通道失真 |
| I4 | `push_chunk(k)` 输出条件于 g_1..g_k（因果结构）| 训练/推理分布不一致 |
| I5 | GRPO group 内 `barge_in_ks` 一致 | group 比较失去同环境前提 |
| I6 | batch 内 `weight_version` 一致；权重更新只在 `close_session` 之后 | KV-version 冲突（缺口文档的最大技术风险） |
| I7 | 环境侧 text/speech 不进 `push_chunk`（防标注泄漏）| reward 信号失真（模型"背答案"） |
| I8 | `omniflow: True` 标记贯穿 collate → worker（识别样本模式） | 普通样本路径误处理全双工 batch |

---

## 8. 现状：已验证 vs 待接线

**已验证**（有单测）：
- A/B：serialize/loss_mask/collate（`tests/test_omniflow.py` 16 测试）
- C：DuplexSessionClient 协议 + ReplayDuplexClient + 编排 + GRPO group
- E：DuplexTrajectory 结构与 summary
- F：四通道 reward 数学（含修正后的 barge-in/打断窗口语义）

**待接线**（代码内 TODO）：
1. **D 桥运行时**：`VllmOmniDuplexClient.push_chunk/barge_in/close_session`
   返回桩结构——WS 事件泵未实装（`_open_raw` 已收敛到单点）
2. **E→训练**：轨迹 → verl DataProto 的 token 化（thinker 输入序列重构，
   gap 文档 P0 项）；`response_mask` 对齐 verl 语义
   （thinker=1 / talker=0 / 控制位=1）
3. **D 出站缺口**（上游限制，非插件可解）：逐 token logprobs（用
   recompute_log_prob 绕开）、wall-clock 时间戳、barge_in_token_index
   ——回放式路线已把 5/6 缺口归约到 L1，交互式才需上游 PR

---

## 9. 速查：一次 RL step 的协议调用链

```
dataloader:  parquet rows → omniflow_collate_fn → {"episodes": [...], "omniflow": True}
trainer:     batch → DuplexSessionRolloutWorker.generate_sequences
worker:      每 episode → collect_group(N 条)
session:     open_session → [barge_in? →] push_chunk × K → close_session
             （桥：input_audio_buffer.append ⇄ response.speak/delta 事件）
trajectory:  DuplexTrajectory × N（group_id/version 戳齐）
reward:      omniflow_trajectory_reward(traj.chunks, gt.chunks) → 分项分数
训练:        轨迹 token 化 → recompute_log_prob（verl 侧）→ GSPO/GRPO
权重同步:    close_session 之后 → update_weights（七环变换链，见部署文档 §四）
```
