# vllm-omni 部署拓扑与 FSDP 参数重排：三阶段模型的变换链

> 回答两个问题：① vllm-omni 怎么部署 thinker/talker/code2wav；② 训练侧 FSDP 到
> 推理侧多 stage 引擎之间，参数经历了哪些重排与变换。
> 全部结论基于本地 checkout 源码验证（行号可查）。
>
> 关联：[fullduplex_gspo_gap_analysis.md](fullduplex_gspo_gap_analysis.md)（缺口 6 权重同步）、
> [feature_fullduplex_omniflow.md](feature_fullduplex_omniflow.md)（episode 边界同步）

---

## 一、推理侧：vllm-omni 的部署拓扑

### 1.1 拓扑是"冻结"在 pipeline 定义里的

每个模型家族在 `vllm_omni/model_executor/models/<model>/pipeline.py` 声明
不可变的 stage 拓扑，注册进 `OMNI_PIPELINES`（`config/pipeline_registry.py`，
即 GP-004 所扩展的注册表）。MiniCPM-o 4.5 的定义
（`minicpmo_4_5/pipeline.py`）：

```
Stage 0: Thinker   model_stage="llm"      LLM_AR        owns_tokenizer
                     → 文本输出 + latent 交给 stage 1
Stage 1: Talker    model_stage="tts"      LLM_AR        hf_config_name="tts_config"
                     → codec token，经 llm2tts 桥接收 thinker latent
Stage 2: Code2Wav  model_stage="code2wav" LLM_GENERATION model_arch="MiniCPMO45Code2Wav"
                     → 波形，经 SharedMemoryConnector 流式收 codec chunk
```

每个 stage 独立声明：`execution_type`（调度器类型）、`input_sources`（数据依赖
DAG）、采样约束（如 talker 的 `stop_token_ids: [6561]` = codec EOS）、
`final_output_type`。stage 间数据面：thinker→talker 走 `llm2tts` 前处理 +
latent 传递；talker→code2wav 走 **SharedMemoryConnector**（codec chunk 流，
`codec_chunk_frames: 25`）。

### 1.2 进程模型：每 stage 一个（组）引擎子进程

`engine/stage_engine_core_proc_manager.py`：`StageEngineCoreProcManager` 为
**每个 replica** 孵化一个子进程（`context.Process(target=StageEngineCoreProc.
run_stage_core)`）。即：3-stage pipeline = 3 个引擎进程（stage 内 TP 再扩
worker），由 `Orchestrator` 统一路由请求与输出。

### 1.3 设备拓扑：deploy yaml 是"配方"

`deploy/minicpmo_4_5.yaml`（默认配置）——**三 stage 同卡共存**：

| Stage | gpu_memory_utilization | 角色 |
|-------|----------------------|------|
| 0 thinker | **0.55** | 大头：多模态 encoder + LLM + KV |
| 1 talker | **0.15** | MiniCPMTTS AR + 2GiB KV（cuda platform 覆盖） |
| 2 code2wav | **0.18** | enforce_eager + CFM DiT |

（三者合计 0.88，留 12% 余量给 HiFi-GAN cuDNN workspace 等并发开销）

变体 `minicpmo_4_5_3gpu_stage1_replicas.yaml`：**stage 1 双副本**——
stage 0 在 GPU 0，stage 1 `devices: "1,2"` + `num_replicas: 2`（talker 是
吞吐瓶颈时横向扩）。平台差异（cuda/npu）也在 yaml 里分 stage 覆盖。

duplex 相关的部署参数：`session_mode: duplex`、`max_sessions: 4`、
`active_stream_window: 4`——注意 `max_sessions` 实际可到 4（缺口文档说的
2 是旧值），但仍远小于 RL batch 需要的并发（见第五节）。

### 1.4 模型实例化：一个类，按 stage 分身

`MiniCPMO45OmniForConditionalGeneration`（`minicpmo_4_5_omni.py:57`）是
**单类双 stage**：`model_stage="llm"` 时只建 `self.thinker = 
init_vllm_registered_model(prefix="thinker")`；`"tts"` 时只建 `self.talker`。
code2wav 是独立 arch。**推理引擎里从不存在"完整的 thinker+talker 模型"**——
每个 stage 进程只持有自己的分身。

---

## 二、权重路由：同一 checkpoint，前缀分拣

`minicpmo_4_5_omni.py` 的 `load_weights` 是阶段分拣器：

```python
# MiniCPM-o checkpoint prefixes → stage mapping:
#   thinker: vpm, resampler, llm, apm, audio_projection_layer
#   talker:  tts (native MiniCPMTTS AR codec producer)
for k, v in weights:
    if k.startswith(("vpm.", "resampler.", "llm.", "apm.", "audio_projection_layer.")):
        thinker_weights.append((k, v))       # → self.thinker.load_weights
    elif k.startswith("tts."):
        talker_weights.append((k, v))        # → self.talker.load_weights
```

要点：**三个 stage 进程从同一份 checkpoint 各取所需**——stage 0 进程把
`tts.*` 当 unknown 跳过，stage 1 进程只认 `tts.*`。加载完成后再
`add_prefix_to_loaded_weights(loaded, "thinker"/"talker")` 对齐内部命名。
这个前缀分拣就是"部署变换"的第一环：checkpoint 是全家的，进程是单stage的。

---

## 三、训练侧：verl-omni 的 FSDP 结构

### 3.1 也是单 stage 加载 + 裁剪（strip-before-shard）

`verl_omni/workers/engine/fsdp/omni_impl.py`：

```python
module = AutoModelForMultimodalLM.from_pretrained(...)   # HF remote code 全模型
adapter_cls = OmniModelBase.get_class_by_name(architecture, model_stage, ...)
module = adapter_cls.configure_model(module, self.model_config)  # ← 裁剪点
```

`configure_model` 的默认实现（`pipelines/model_base.py:624`）执行
`get_strip_modules`（**我们的槽位①**）：

> "Stripping them before FSDP wrapping saves memory and avoids sharding
> unused parameters" —— thinker-only 训练典型值 `["talker", "code2wav",
> "code_predictor"]`

即：**训练 FSDP worker 也只持有一个 stage**（加载全模型→立刻删除不训练的
子模块→再分片）。裁剪发生在 FSDP 包装**之前**，所以：
- flat param 内存里根本没有 talker/code2wav 的参数
- 梯度、优化器状态、checkpoint 都天然只有被训练 stage

### 3.2 FSDP2 分片与"去展平"取回

训练中权重同步的取数路径（`verl/utils/fsdp_utils.py:477`）：

```python
# fsdp_version == 2:
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
state_dict = get_model_state_dict(model, options=StateDictOptions(
    full_state_dict=True, cpu_offload=..., broadcast_from_rank0=...))
```

FSDP2 的 `fully_shard` 把参数展平成 flat shard（DTensor），但
`get_model_state_dict` 负责**去展平并恢复原始 HF 命名**——这是整个变换链
里最重要的一环：**FSDP 分片结构不泄漏到权重同步协议里**，出去的就是
HF 原名的全量 state dict（且只剩被训练 stage 的前缀）。

---

## 四、完整变换链：从 checkpoint 到三个引擎进程

```
┌──────────────────────────────────────────────────────────────────┐
│ HF checkpoint（一份，全家前缀混合）                                  │
│   vpm.*  resampler.*  llm.*  apm.*  audio_projection_layer.*      │
│   tts.*  (code2wav 权重)                                           │
└──────────────┬───────────────────────────────────────────────────┘
               │ ① AutoModelForMultimodalLM.from_pretrained（HF remote code）
               │    训练侧：model_stage + {stage}_config 子配置
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 全模型（内存中）                                                    │
└──────────────┬───────────────────────────────────────────────────┘
               │ ② get_strip_modules 裁剪（configure_model，FSDP 前）
               │    thinker-only: 删 talker/code2wav/code_predictor
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 单 stage 模型 → ③ FSDP2 fully_shard（展平分片 + DTensor）           │
│    （flat 内存无冻结 stage 参数）                                    │
└──────────────┬───────────────────────────────────────────────────┘
               ════════ 训练循环 ════════
               │ ④ get_model_state_dict(full_state_dict=True)
               │    去展平 → 恢复 HF 原名（vpm.*…，无 tts.*）
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ ⑤ BucketedWeightSender：分桶 IPC（ZMQ/SHM）                        │
│    handle: ipc:///tmp/rl-colocate-zmq-{job}-replica-{r}-rank-{k}  │
└──────────────┬───────────────────────────────────────────────────┘
               │ ⑥ rollout worker.update_weights_from_ipc
               │    每桶 model.load_weights(weights)
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ ⑦ vllm-omni 各 stage 进程的前缀分拣 load_weights                    │
│    stage 0 进程: vpm./resampler./llm./apm. → self.thinker          │
│    stage 1 进程: tts. → self.talker（thinker-only 训练时收不到更新）│
│    → process_weights_after_loading（量化/后处理）                    │
└──────────────────────────────────────────────────────────────────┘
```

### 变换清单（七环）

| # | 变换 | 位置 | 性质 |
|---|------|------|------|
| ① | 全模型加载（stage 子配置） | `omni_impl.py` from_pretrained | 拓扑：checkpoint→内存全模型 |
| ② | strip 裁剪 | `configure_model` + `get_strip_modules`（槽位①） | **空间**：FSDP 分片不含冻结 stage |
| ③ | FSDP2 展平分片 | verl fsdp worker | 空间：参数→flat shard/DTensor |
| ④ | 去展平+恢复原名 | `get_model_state_dict` | **命名**：分片结构不泄漏 |
| ⑤ | 分桶 IPC | `BucketedWeightSender`（ZMQ/SHM；separate 模式 NCCL） | 传输：分桶限制 GPU 内存峰值 |
| ⑥ | 逐桶装载 | `update_weights_from_ipc` → `load_weights` | 协议：流式，无全量驻留 |
| ⑦ | 前缀分拣 | vllm-omni 各 stage 的 `load_weights` | **路由**：同名 checkpoint 喂多个分身 |

### 关键推论

1. **选择性同步是天然的**：thinker-only 训练的 state dict 里没有 `tts.*`
   ——stage 1/2 进程在权重同步时**收不到任何张量**，保持冻结。不需要显式
   的"跳过 talker"逻辑，前缀分拣自动完成。（这正是缺口文档
   "只训 Thinker 时只同步 Thinker 权重"的实现机制。）
2. **LoRA 有独立快路**：`update_weights_from_ipc` 的 LoRA 分支累积全桶后
   原子 `add_lora`（AR 模型走 `TensorLoRARequest`，diffusion worker 走
   `OmniTensorLoRARequest`）——LoRA swap 比 full sync 快一个量级，
   对 episode 边界同步的吞吐友好。
3. **命名对齐的契约点只有一个**：④输出 HF 原名 ↔ ⑦按 HF 前缀分拣。
   我们插件侧新增模型时，`get_strip_modules` 返回的属性名必须与 HF
   checkpoint 的顶层前缀一致（MiniCPM-o 4.5 是 `vpm./llm/.../tts.`，
   Qwen3-Omni 风格是 `thinker./talker./token2wav.`）——**属性名错了
   strip 不掉，FSDP 就会分片并同步不该训的参数**。这一步应该探针先行
   （`python -m verl_omni_ext.probes.forward_signature`）。
4. **dtype 变换**在两端独立决策：训练 `torch_dtype`（非 forward_only 默认
   bf16，forward_only fp32）；推理在 deploy yaml 每stage配置，装载后
   `process_weights_after_loading` 做最终量化/编译处理。

---

## 五、对 verl-omni-plugin 的部署决策影响

### 5.1 rollout 引擎的两种挂载方式

| 方式 | 机制 | 权重同步路径 | 适用 |
|------|------|-------------|------|
| **colocate** | vllm-omni worker 以 `vLLMOmniColocateWorkerExtension` 挂进 actor 进程树 | ZMQ/SHM IPC（⑤⑥） | 训推同卡，主流 RL |
| **separate** | vllm-omni 独立 HTTP server | NCCL broadcast（`update_weights_from_ipc` 的 ipc 分支） | 训推分卡/分机 |

全双工 duplex engine 是独立进程组（Orchestrator + 3 stage 子进程），
separate 形态更接近它的自然部署；episode 边界同步时 `update_weights`
要能触达 duplex engine 的三个 stage 进程——colocate 扩展类是否覆盖
duplex 进程树需要实测（gap 文档缺口 6 的验证点，桥接位置在
`_vllm_omni_bridge.py`）。

### 5.2 并发限制

deploy yaml 的 `max_sessions: 4` + `active_stream_window: 4`：duplex 并发
会话数上限 4。RL batch 的 group（如同 episode 4 条采样）可以塞进 4 个
session，但**多 episode 并行**需要：多 replica（`num_replicas`，参照
3gpu_stage1_replicas 变体）或串行 episode（回放式 MVP 的选择——
吞吐换正确性，与 episode 边界权重同步天然契合）。

### 5.3 插件侧的三个落点

1. **GP-004 注册 pipeline**：`models/<m>/vllm_omni/pipeline.py` 产出的
   `PipelineConfig` 要与训练侧 `architecture`/`pipeline_name` 咬合
   （三键咬合规则），stage 拓扑里 `model_stage` 值要与 HF 前缀分拣规则一致。
2. **thinker adapter 的 `get_strip_modules`**：新模型探针的第一件事——
   确认 HF 类的顶层属性名与 checkpoint 前缀，strip 错了整条同步链错。
3. **episode 边界同步**：复用 ⑤⑥⑦ 链路，编排层（duplex_rollout.py）控制
   同步时机在 `close_session` 之后；`weight_version` 戳沿轨迹传递。

---

## 六、速查：一图流

```
训练（每 FSDP worker）                  推理（每 stage 进程）
─────────────────────                  ─────────────────────
AutoModelForMultimodalLM ←─ 同一 checkpoint ─→ load_weights 前缀分拣
        │ strip（槽位①）                        ├─ stage0: vpm/llm/apm… → thinker
        ▼                                        ├─ stage1: tts… → talker
   FSDP2 flat shard                              └─ stage2: code2wav arch
        │ get_model_state_dict
        ▼                                   权重同步（episode 边界）
   HF 原名 state dict ──分桶 IPC──→ update_weights_from_ipc ──→ 各 stage 分拣
   （无被 strip 的前缀）                  （冻结 stage 天然零更新）
```

---

## 参考（源码行号）

- 拓扑冻结：`vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py`（MINICPMO_4_5_PIPELINE）
- 注册表：`vllm_omni/config/pipeline_registry.py:182`（GP-004 扩展点）
- 进程孵化：`vllm_omni/engine/stage_engine_core_proc_manager.py:48`
- 部署配方：`vllm_omni/deploy/minicpmo_4_5.yaml` / `minicpmo_4_5_3gpu_stage1_replicas.yaml`
- 单类双 stage：`minicpmo_4_5_omni.py:57,94-123`
- 前缀分拣：`minicpmo_4_5_omni.py` load_weights（vpm/resampler/llm/apm ↔ tts）
- 训练侧加载：`verl_omni/workers/engine/fsdp/omni_impl.py:150-210`
- strip 语义：`verl_omni/pipelines/model_base.py:563-653`（槽位①）
- FSDP2 取数：`verl/utils/fsdp_utils.py:477-485`
- 权重同步 worker：`verl_omni/workers/rollout/vllm_rollout/utils.py:29-230`
  （LoRA 原子 add_lora / 全量分桶 load_weights / ZMQ handle 规则）