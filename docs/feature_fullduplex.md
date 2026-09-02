# 特性添加指南：全双工 Omni RL 训练

> 全双工不是"加一个模型"，而是"加一个训练范式"。
> 它跨多个层面，但训练侧可以完全零侵入。

---

## 一、全双工 vs 传统 RL

```
传统 RL（串行）：
  训练 → 停 → 推理生成 rollout → 停 → 训练 → 停 → 推理 → ...
  GPU 利用率低（推理时训练卡空闲，训练时推理卡空闲）

全双工 RL（并发）：
  训练 Worker ──────────────────────────────────►  一直在跑
                  ↑ 权重同步 ↓
  推理 Worker ──────────────────────────────────►  一直在跑
                  ↑ 推理结果反馈 ↓
  GPU 利用率高（训练和推理同时进行）
```

---

## 二、verl 已有的异步基础（不需要重新发明）

| 机制 | 文件 | 作用 |
|------|------|------|
| `@register_trainer("separate_async")` | `verl/trainer/ppo/v1/trainer_separate_async.py` | 训练和 rollout 分离，可部分重叠 |
| `@register_trainer("colocate_async")` | `verl/trainer/ppo/v1/trainer_colocate_async.py` | 训练和 rollout 共置，partial rollout |
| `FullyAsyncLLMServerClient` | `verl/workers/rollout/llm_server.py` | 异步推理服务客户端 |
| `agent_loop_tq` | `verl/trainer/ppo/v1/agent_loop_tq.py` | TransferQueue 异步数据流 |
| `checkpoint_manager.update_weights()` | `verl/checkpoint_engine` | 权重同步 |
| `@register_trainer` 注册表 | `verl/trainer/ppo/v1/trainer_base.py:1900` | trainer 注册扩展点 |

**全双工 trainer = 继承 omni_sync + 改异步 + 加权重同步。**

---

## 三、添加方式：按层面拆解

### 层面 1：训练 trainer（L1 插件，零侵入 ✅）

```python
# verl_omni_ext/trainer/fullduplex_trainer.py
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

@register_trainer("omni_fullduplex")
class OmniPPOTrainerFullDuplex(OmniPPOTrainerSync):
    """全双工：训练和推理并发"""

    def get_llm_client(self):
        return self.llm_server_manager.get_client(
            client_cls=FullyAsyncLLMServerClient
        )

    def on_train_begin(self):
        # 预热：往推理队列放 warmup batches
        for _ in range(num_warmup):
            self._add_batch_to_generate()

    def on_step_end(self):
        # 每 N 步同步权重
        if self.global_steps % sync_step == 0:
            self.checkpoint_manager.update_weights(self.global_steps)
```

**为什么零侵入**：`@register_trainer` 是 verl 的公开扩展点（注册表字典），通过 `VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext` 触发 import → 注册。

### 层面 2：配置选择（L1，零侵入 ✅）

```yaml
# config.yaml
trainer:
  v1:
    trainer_name: omni_fullduplex    # ← @register_trainer 注册的名字
    fullduplex:
      num_warmup_batches: 2           # 预热 batch 数
      parameter_sync_step: 4          # 权重同步频率

actor_rollout_ref:
  model:
    external_lib: verl_omni_ext      # ← 触发 trainer 注册
    model_type: omni_model
    model_stage: thinker
```

### 层面 3：trainer 选择逻辑（零侵入 ✅）

```
main_omni.py 的链路：
  uses_v1_trainer() → True（trainer_type=policy_gradient）
  → run_ppo() 加载 V1 trainer 系统
  → 从 config.trainer.v1.trainer_name 匹配 @register_trainer("omni_fullduplex")
  → 你的 trainer 被实例化

全程不需要改 main_omni.py——它只检查 trainer_type，
trainer_name 的匹配在 verl 的 V1 系统里完成。
```

### 层面 4：推理侧（⚠ 需要改 vllm-omni 源码树）

vllm-omni 已有实验性全双工代码：

```
vllm_omni/experimental/fullduplex/
  ├── core/runtime.py      DuplexRuntime — 全双工推理运行时
  ├── core/session.py      DuplexSession — 会话管理
  ├── core/adapter.py      DuplexAdapter — 输入/输出适配
  ├── personaplex/          全双工会话管理实验
  └── request_client.py    异步请求客户端
```

但：
1. 这些是**实验性**代码，不是稳定 API
2. vllm-omni **没有 plugin 机制**——必须改源码树（见 [rollout 适配分析](rollout_adaptation.md)）
3. 需要适配到你的具体模型

**重要判断**：如果你的全双工只做"训练和推理并发"（不做真正的流式推理），可能不需要 vllm-omni 的 fullduplex——verl 的 `FullyAsyncLLMServerClient` 已经足够支持"训练和推理并发 + 权重同步"。

### 层面 5：数据流（复用 verl 机制 ✅）

verl 的 `agent_loop_tq.py` 提供了 TransferQueue 异步数据流：
- 训练数据 → 训练队列 → 训练 Worker
- 推理提示 → 推理队列 → 推理 Worker
- 推理结果 → 训练队列（用于 RLHF）

不需要重新发明——继承现有机制即可。

---

## 四、改动矩阵

| 层面 | 改什么 | 怎么 add-on | 侵入性 | 文件 |
|------|--------|-------------|--------|------|
| 训练 trainer | `@register_trainer("omni_fullduplex")` | ext 包新建 trainer | ✅ L1 | `verl_omni_ext/trainer/fullduplex_trainer.py` |
| 配置 | config.yaml 设 trainer_name | 纯配置 | ✅ L1 | config.yaml |
| 权重同步 | `on_step_end` → `update_weights` | 复用 verl 机制 | ✅ L1 | 同上 |
| 数据流 | agent_loop_tq | 复用 verl 机制 | ✅ L1 | 不需要新文件 |
| trainer 选择 | `uses_v1_trainer` | 不需要改 | ✅ L1 | — |
| **推理侧** | vllm-omni fullduplex 适配 | **需要改 vllm-omni 源码树** | ⚠ 侵入 | vllm-omni 仓 |

---

## 五、与模型适配的关系

全双工是一个**正交特性**——可以和任何模型适配组合：

| 组合 | 训练 trainer | 模型适配 |
|------|-------------|---------|
| Qwen3.5 + 同步 | `omni_sync` | `qwen3_5_moe` |
| Qwen3.5 + 全双工 | `omni_fullduplex` | `qwen3_5_moe` |
| MiniCPM-o + 全双工 | `omni_fullduplex` | `minicpmo_5_0` |

模型适配（槽位①②）和训练范式（trainer 注册表）是独立的扩展点，互不干扰。

---

## 六、骨架代码

见 [`verl_omni_ext/trainer/fullduplex_trainer.py`](../verl_omni_ext/trainer/fullduplex_trainer.py)

---

## 七、验证清单

- [ ] `@register_trainer("omni_fullduplex")` 被 import 触发（检查 `OmniPPOTrainerFullDuplex` 在注册表中）
- [ ] config.yaml 的 `trainer_name: omni_fullduplex` 能被 verl V1 系统匹配
- [ ] `FullyAsyncLLMServerClient` 能连接到推理引擎
- [ ] warmup batches 能正确放入 agent_loop 队列
- [ ] `update_weights` 能正确同步到推理引擎
- [ ] 推理结果能正确反馈到训练队列
- [ ] GPU 利用率确实提升了（训练和推理同时跑）
