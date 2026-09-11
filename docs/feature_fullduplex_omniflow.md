# 全双工交互训练（Omni-Flow）：基于 MiniCPM-o 4.5 论文的插件设计

> 论文：[MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction](https://arxiv.org/abs/2604.27393)
>
> 本文回答：如何把论文的全双工交互范式（Omni-Flow）落进 verl-omni-plugin 的三层分治架构，
> 做到训练侧零侵入、rollout 侧零新增 gate patch。
>
> **与 [feature_fullduplex.md](feature_fullduplex.md) 的关系**：那篇讲的是
> **训练系统级全双工**（训练与 rollout 并发执行）；本文讲的是**交互范式级全双工**
> （模型边听边说、可打断、可主动）。两者正交，可叠加使用。

---

## 一、论文核心机制（工程视角提炼）

### 1. Omni-Flow：把交互变成时间对齐的 token 流

论文 §3 的三个时间对齐流：

| 流 | 内容 | 载体 |
|----|------|------|
| `env-visual` | 环境视觉观测 | 每 chunk 的视觉 token `v_k` |
| `env-audio` | 声学场景（含用户语音） | 每 chunk 的音频 token `a_k` |
| `out-stream` | 助手的文本 + 语音输出 | 每 chunk 的输出 token `o_k` |

**统一序列化**（论文 §3.2）——第 k 个时间块的 token 组：

```
g_k = [v_k ; a_k ; o_k]          # 组内先感知后输出，每个输出都条件于最新观测
序列 = g_1 g_2 g_3 ...           # 标准 causal LM 直接消费
```

**用户不再是"特权对话角色"**：用户语音只是 `env-audio` 的一部分，模型自己决定
**是否说、何时说、说什么**——这是主动行为（proactive behavior）的来源。

### 2. LS 控制形式（论文 §3.3 ablation 的胜者）

- **LS（Listen-Speak）**：模型在每个输出位先预测二值控制 token（`<|listen|>` / `<|speak|>`），再生成内容
- **LT（Listen-Text）**：`<|listen|>` 和普通文本 token 共享一个输出空间直接预测

论文结论：**LS > LT**——"是否说"和"说什么"必须解耦，纠缠在一个预测步里全双工更难学。
vllm-omni 的 `minicpmo45/policy.py` 印证了这一点（`<|listen|>`/`<|speak|>` 独立 token）。

其他 ablation 结论：chunk size **0.2s** 最优（1.0s 太钝、0.1s 训练不稳）；组间显式边界 token。

### 3. TAIL：时间对齐交织（论文 §3.4）

难点：文本生成速度 ≠ 语音播放速度——文本跑在前面，播出来的语音就是"过期"的。

TAIL 策略：第 k 块生成的文本量自适应调节，使播放进度逼近当前时间边界 `k·t`；
落后了就少生成一点让语音追上来。**有界 look-ahead**：chunk k 最后几个文本 token 的
语音 token 推迟到 chunk k+1（给发音留局部上下文，如 "the apple" vs "the car"）。

### 4. 论文的 RL 配方（§5.4）

- **GRPO**（不是 PPO）+ answer accuracy（rule-based + judge model）+ format reward
- **smooth length reward**（Kimi-K1.5 式）：答对时奖励短、答错时罚短；前 480 步不加（先收敛）
- **RLAIF-V** 降幻觉（图文学到的能力迁移到流式全双工）

---

## 二、关键现状盘点（决定架构的三个事实）

### 事实 1：vllm-omni 主干已有完整全双工 runtime（无需新增 L3）

[PR #3907](https://github.com/vllm-project/vllm-omni/pull/3907)（已合并，142 文件 +48k 行）
按 [RFC #3745](https://github.com/vllm-project/vllm-omni/issues/3745) 落地：

```
vllm_omni/experimental/fullduplex/
├── core/        模型无关契约（DuplexAdapter / session / turn runtime）
├── engine/      AsyncOmni/orchestrator 调度数据面（session KV lease、barge-in epoch）
├── openai/      WS 传输、Realtime 投影、音频编解码（/v1/duplex、/v1/realtime?duplex=1）
├── minicpmo45/  MiniCPM-o 4.5 帧协议、listen/speak policy、Stage0 状态
└── personaplex/ Moshi 级 lockstep 语音到语音模型
```

MiniCPM-o 4.5 的激活路径：`openai` session controller + `engine` contracts +
标准 scheduler + 注入的 `minicpmo45/runtime.py` 扩展。

**新增全双工模型的 seam 是 `core.DuplexAdapter`**（`capabilities` / `on_input` / `respond`）。

### 事实 2：MiniCPM-o 4.5 的帧协议已固化（policy.py 的常量）

```
SAMPLE_RATE_HZ = 16000          # 1s 单元 @16kHz
SAMPLES_PER_AUDIO_TOKEN = 1600  # 100ms 一个 audio embedding
CHUNK = 10 embeddings + <unit>…</unit>
VISION：每帧 <image> + 64 resampler embeds + </image>
MAX_NEW_SPEAK_TOKENS_PER_CHUNK = 20  # 每 chunk 最多说 20 token（TAIL 的运行时形式）
```

### 事实 3：verl-omni 训练侧的扩展点足够（无需改源码）

| 槽位 | 全双工用法 |
|------|-----------|
| ① `@OmniModelBase.register` | Omni-Flow 格式模型的 thinker adapter（4 方法） |
| ② `@OmniRolloutPipelineBase.register` | rollout pipeline（全双工轨迹形态） |
| ③ `VERL_USE_EXTERNAL_MODULES` | 加载 verl_omni_ext |
| ④ `data.custom_cls`（pkg:// 路径） | **Omni-Flow 数据集**：时间索引样本 → chunk 序列 |
| ⑤ `@register_trainer` | `omni_fullduplex`（已有骨架）+ GRPO |
| ⑥ reward manager `register` + `custom_reward_function` | 全双工 reward 四件套 |

---

## 三、全双工 RL 的范式选择：回放式（replay-based）

全双工 RL 的根本难点：**训练需要有限 episode + 可复现环境**，而全双工交互是
连续无限流。论文数据形态给出了答案：

> "Each training sample contains the full visual input, audio input, output text and
> output speech, where each piece of information is tagged with a time index."（§4.3）

**用户流是录制的（不响应模型），模型流是策略**。这就是回放式环境：

```
环境  = 录制的 (env-visual, env-audio) 时间索引流     ← 固定，可复现
策略  = 模型在每个 chunk 的 o_k（LS 控制 token + 内容）← 待训练
episode = 一条录制流的 [0, T) 窗口（如 30s~5min）
```

为什么不建交互式用户模拟器（TTS/LLM-driven）：
1. 论文数据天然是回放形态，SFT 和 RL 用同一套数据管道
2. 回放环境下 GRPO group 天然成立：**同一录制流上 N 次采样**（同上下文不同输出），
   group 内对比产生优势——正是 GRPO 的形态，与论文用 GRPO 一致
3. 交互式模拟器是独立的开放问题（语气/打断行为建模），不应耦合在训练插件里

代价与对策：模型输出不会影响用户后续行为（无真实闭环）。论文的做法是
SFT 学行为模式（被动部分）+ RL 学决策质量（chunk 级判别），实践上够用；
后续如需闭环可加 simulator 槽位（见 §九）。

---

## 四、三层分治映射

```
L1 插件（本仓库，~95%）
├── verl_omni_ext/models/minicpmo_45/          # Omni-Flow 模型适配
│   ├── thinker_adapter.py                     #   槽位①：strip/configure_*4 方法
│   ├── rollout_adapter.py                     #   槽位②：全双工轨迹 pipeline
│   └── dataset.py → 由 features/fullduplex/omniflow_dataset.py 复用
├── verl_omni_ext/features/fullduplex/
│   ├── omniflow_dataset.py                    #   槽位④：时间索引→g_k 序列化
│   ├── rewards.py                             #   槽位⑥：四类 reward
│   ├── duplex_rollout.py                      #   rollout worker：duplex session 驱动
│   └── trainer.py                             #   槽位⑤：omni_fullduplex（GRPO）
└── examples/minicpmo_45/run_*.sh + config/

L2 monkey patch（~4%，仅在 remote code 不兼容时）
└── minicpmo_45/patches.py                     # forward 签名/processor（复用 _patchkit
                                               #   probe_signature，多进程传播证明）

L3 gate patch（0 条新增）
└── rollout 侧复用 vllm-omni 主干已有 minicpmo45 runtime（PR #3907 已合并）
    GP-004 照旧：VLLM_OMNI_EXTERNAL_MODULES 注册 pipeline 定义
```

**为什么 rollout 侧零新增 L3**：训练 rollout 复用 vllm-omni 的
`/v1/duplex` WS API（`request_client.py` 已有异步客户端），会话生命周期、
barge-in epoch、playback cursor 都由主干管理；我们只在插件里做
"环境回放 + 轨迹收集"的编排。

---

## 五、数据管道（omniflow_dataset.py，槽位④）

### 输入格式（parquet 每行一个 episode）

```jsonc
{
  "episode_id": "ep_000123",
  "duration_s": 42.0,
  "streams": {
    "visual": [{"t_start": 0.0, "t_end": 0.2, "frames": ["frame_0000.jpg"]}, ...],
    "audio":  [{"t_start": 0.0, "t_end": 0.2, "samples_b64": "...", "sr": 16000}, ...],
    "text":   [{"t_start": 3.4, "t_end": 4.1, "tokens": ["好", "的", "，", "马上"]}],
    "speech": [{"t_start": 3.6, "t_end": 4.4, "tokens_b64": "..."}]   // 参考 speech token
  },
  "meta": {"source": "web_av" | "task_data", "has_ref_audio": true}
}
```

### 序列化规则（论文 §3.2 + §3.4 TAIL）

1. 按 chunk size `t=0.2s` 切时间轴，第 k 块产出 `g_k = [v_k ; a_k ; o_k]`
2. `v_k`：t 落在块内的帧 → `<image>` + 64 embeds + `</image>`
3. `a_k`：音频样本按 `SAMPLES_PER_AUDIO_TOKEN=1600` 池化 → 10 embeddings + `<unit>` 包装
4. `o_k`（监督信号来自标注流）：
   - 该块无输出 → `[<|listen|>]`（LS 控制位）
   - 该块有输出 → `[<|speak|>]` + text token（按 **start time** 归块）
5. **TAIL 有界 look-ahead**：块内最后 `lookahead_n=3` 个文本 token 的 speech token
   推迟到 k+1 块（`t_start` 用 text 的，`t_end` 用 speech 的判归属）
6. loss mask：`v_k/a_k` 全 mask（不训），`o_k` 训练位 = LS 控制位 + 内容位
7. GRPO group：`group_size` 个样本共享同一 `episode_id`（同环境不同采样）

### 对 vllm-omni 帧协议的一致性

`SAMPLES_PER_AUDIO_TOKEN`、`VISION_TOKENS_PER_FRAME` 等常量**必须与
vllm-omni `minicpmo45/policy.py` 完全一致**（否则训练与推理的 token 数错位，
KV 里出现 pad embedding 会"measurably corrupt listen/speak behavior"——
policy.py 原注释）。实现时**直接 import vllm-omni 的 policy 常量**，
不复制字面量。

---

## 六、Rollout 编排（duplex_rollout.py）

### Episode 生命周期

```
open_session(session.config{model, voice, chunk_period_ms=200})
  │
  ├─ for k in 0..K-1:                     # 回放录制流
  │    push_chunk(env_visual[k], env_audio[k])   # /v1/duplex input append
  │    collect o_k: control_token + content + speech delta + barge_in events
  │
  ├─ （episode 内用户流固定 → 模型在"实时"压力下做 listen/speak 决策）
  │
  └─ close_session() → trajectory 归档
```

### 轨迹（trajectory）结构

```jsonc
{
  "episode_id": "ep_000123",
  "group_id": "grp_0007",            // GRPO group：同 episode 同环境
  "weight_version": "step_120",      // 权重版本戳（episode 边界同步的依据，见 §七）
  "chunks": [
    {
      "k": 0, "t_range": [0.0, 0.2],
      "control": "<|listen|>",       // 或 <|speak|>
      "content_tokens": [],          // text token ids
      "speech_tokens": [],           // TAIL 归属本块的 speech token
      "latency_ms": 0,               // 控制决策延迟
      "events": []                   // barge_in / playback_ack / cancel
    }, ...
  ]
}
```

### worker 形态（接入 verl 的两种方式）

1. **`worker_cls` 配置**（`actor_rollout_ref.rollout.worker_cls`）——本插件的
   `DuplexSessionRolloutWorker` 包装 vllm-omni 的
   `experimental/fullduplex/request_client.py`
2. 复用 `FullyAsyncLLMServerClient` 时（简单路径）：duplex session 作为
   agent loop 的一个"环境步"，`agent_loop_tq` 传轨迹

---

## 七、权重同步 × session KV：episode 边界对齐

**冲突**：异步训练 `update_weights` 时，活跃 duplex session 的 KV cache 是
**旧权重**算的——新权重续写旧 KV 语义错位（RFC #3745 的 KV lease 与
verl 权重同步的正面碰撞）。

**方案（episode 边界同步，默认）**：
- episode 是有限窗口（配置 `fullduplex.episode_max_chunks`，如 150 块=30s）
- rollout 引擎在 episode 边界才应用新权重：`close_session` 释放全部 KV lease →
  `update_weights` → 下个 episode 用新权重 `open_session`
- `weight_version` 戳随轨迹记录，训练侧校验 batch 内版本一致（防混版本优势估计）

**不采用的方案**：RFC 的 `rollback_to_checkpoint` / re-prefill 续session——
训练场景 episode 短，重建成本低于复杂度成本；那条路留给分钟级长会话 serving。

---

## 八、Reward 设计（rewards.py，槽位⑥）

四类信号，对应论文的 reward 组合方式（加权求和，权重 config 暴露）：

| 类 | 信号 | 计算 | 来源 |
|----|------|------|------|
| **决策恰当性** | 该说才说，该听才听 | 录制流的标注 text/speech 时间戳为 ground truth：标注有输出而模型 `<|listen|>`（漏说）→ 罚；标注静默而模型 `<|speak|>`（抢话）→ 罚 | 论文 LS 范式的直接监督化 |
| **及时性** | 说的内容和当前时刻对齐 | TAIL 对齐度：chunk k 的 text token 量 vs 播放进度偏差的负值；barge-in 后停止延迟（目标 <400ms v2v，论文/RFC） | 论文 §3.4 |
| **打断处理** | 被打断后正确切换 | barge-in epoch 事件后：模型在 ≤N chunk 内转 `<|listen|>` → 奖；继续说旧 epoch 内容 → 罚 | RFC #3745 use case |
| **内容质量** | 说得对、不啰嗦 | 论文配方：rule-based accuracy + judge model + format + Kimi-K1.5 smooth length（**前 480 步不加 length**，先收敛） | 论文 §5.4 |

注册方式（全部走扩展点）：

```python
# verl_omni_ext/features/fullduplex/rewards.py
from verl.experimental.reward_loop.reward_manager.registry import register

@register("omniflow_duplex_reward")
class OmniFlowDuplexRewardManager(RewardManager):   # 决策+及时性+打断（轨迹级，快）
    ...

# 内容质量走 custom_reward_function（config: custom_reward_function.path）
def omniflow_content_score(data_source, solution_str, ground_truth, extra_info): ...
```

RL 算法：`@register_adv_est("grpo")` verl 已有 GRPO——trainer 直接用
`trainer.v1.adv_estimator=grpo`，group = 同 episode_id 的 N 条轨迹。

---

## 九、实现清单与后续扩展

### 本仓库新增/改动（本次）

| 文件 | 内容 |
|------|------|
| `docs/feature_fullduplex_omniflow.md` | 本文 |
| `verl_omni_ext/features/fullduplex/omniflow_dataset.py` | 时间索引→g_k 序列化（slot ④） |
| `verl_omni_ext/features/fullduplex/rewards.py` | 四类 reward + 注册 |
| `verl_omni_ext/features/fullduplex/duplex_rollout.py` | episode 回放 worker + 轨迹收集 |
| `verl_omni_ext/features/fullduplex/trainer.py` | 补 GRPO 配置 + episode 边界同步说明 |
| `tests/test_omniflow.py` | 序列化单测（chunk 归属/TAIL/loss mask） |

### 模型适配（下一个 PR，仿 minicpmo_5_0 三步走）

`verl_omni_ext/models/minicpmo_45/`：thinker_adapter（槽位①，4 方法）+
rollout_adapter（槽位②）+ 探针先跑
（`python -m verl_omni_ext.probes.forward_signature`——适配决策是测量出来的）。

### 后续扩展（不在本次范围）

- **交互式用户模拟器**：simulator 槽位（TTS/LLM-driven），回放式跑通后做闭环
- **语音 token 重建 loss**：flow-matching decoder 的训练（论文 speech pretraining
  阶段）——verl-omni 主干若加流式 decoder 训练支持，走 L1/L3 评估
- **RLAIF-V**：幻觉抑制，复用 verl 的 reward model 循环

---

## 参考

- 论文：[MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction](https://arxiv.org/abs/2604.27393)（Omni-Flow §3、TAIL §3.4、数据 §4.3、RL §5.4）
- [RFC #3745: Full-Duplex Session Architecture for vLLM-OMNI](https://github.com/vllm-project/vllm-omni/issues/3745)（DuplexSession/KV lease/barge-in epoch 设计）
- [PR #3907: Full-Duplex realtime runtime & MiniCPM-o 4.5 demo](https://github.com/vllm-project/vllm-omni/pull/3907)（已合并，`vllm_omni/experimental/fullduplex/`）
- vllm-omni 本地 checkout：`vllm_omni/experimental/fullduplex/{core,engine,openai,minicpmo45}/`
- 本仓库：[three_layer_strategy.md](three_layer_strategy.md) · [feature_fullduplex.md](feature_fullduplex.md)（训练系统级全双工）· [inject_new_model.md](inject_new_model.md)