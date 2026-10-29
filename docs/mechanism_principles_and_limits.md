# 插件与 Patch 的原理及限制

> 三层分治（L1 插件 / L2 monkey patch / L3 gate patch）的机制本质、
> 工程化包装、以及——同样重要的——每一层解不了什么。
> 全部结论基于源码验证（行号可查）。
>
> 关联：[three_layer_strategy.md](three_layer_strategy.md)（分层决策）、
> [plugin_architecture_design.md](plugin_architecture_design.md)（扩展点清单）、
> [data_pathway_protocol.md](data_pathway_protocol.md)（数据面契约）

---

## 一、原理

### 1.1 L1 插件：注册表注入 + "上游拉起"

**机制本质**：所有 L1 扩展都是往上游的**内存 dict 注册表**塞条目——
插件类和上游原生类在注册表里无差别：

| 注册表 | 塞什么 | 查表点 |
|--------|--------|--------|
| `OmniModelBase._registry[(arch, stage)]` | 模型适配器类 | fsdp worker `get_class_by_name` |
| `@register_trainer` 表 | trainer 类 | verl main 按名实例化 |
| reward manager 表 | manager 类 | reward 阶段 |
| `OMNI_PIPELINES[model_type]` | PipelineConfig | `resolve_pipeline_config` |
| `OmniModelRegistry.models[arch]` | 惰性模型类 | `ModelConfig.registry` |

**触发链（训练侧）**——关键设计是**让上游"拉"而不是插件"推"**：

```
config: external_lib: verl_omni_ext
  → verl-omni worker 解析 config 时调 import_external_libs("verl_omni_ext")
     （verl/utils/import_utils.py:81——本体就是 8 行 importlib.import_module）
  → 执行 verl_omni_ext/__init__.py
  → _load_all(): entry_points() 遍历 verl_omni.{models,trainers,reward}
     （importlib.metadata 只看已安装的 distribution）
  → ep.load() import 每个模型模块
  → 模块级执行：L2 补丁应用 + @OmniModelBase.register() 塞注册表
  → 之后上游任意查表点命中插件条目
```

"拉"的好处：加载发生在**每个需要它的进程**里（每个 Ray worker 解析
config 时各拉一次），注册表条目的进程覆盖面由上游调用拓扑保证，
插件不需要自己操心"哪些进程要装"。

**触发链（推理侧）**：pip install 后由 vllm-omni 原生插件组拉起——
`vllm_omni.general_plugins` 入口点 → `load_omni_general_plugins()`
（`OmniEngineArgs.__post_init__`，所有进程）→ `register()` →
`register_pipeline()` / `OmniModelRegistry.register_model()`。

**槽位④⑤等非注册表扩展点**：`data.custom_cls`（pkg:// 路径动态加载
collate_fn）、`custom_reward_function`（函数路径配置）、`worker_cls`
（动态加载）——共同模式是**配置里写插件模块路径，上游在用时加载**。

### 1.2 L2 monkey patch：import 期对象属性替换 + 工程化包装

**机制本质**：`setattr(module, attr, wrapper)`——在第三方对象被使用前
替换其属性。`_patchkit.py` 在这个原语上包了四层工程保障：

| 原语 | 解决的问题 |
|------|-----------|
| `idempotent_patch(...)` | 重复 import 不重复打（`__patch_state__` 标记） |
| `probe_signature=True` | 记录目标源码的 sha256 指纹——上游版本漂移可检测 |
| `assert_patch_consensus(states)` | 多进程补丁状态一致性证明（Ray worker 各自 import，状态应全等；不等 = 有进程漏打或版本倾斜） |
| `verify_patches_alive()` | 运行时复查（watchdog——对象被第三方二次替换时报警） |

`VERL_OMNI_EXT_PATCH_STRICT=1`（运行时动态读取）把所有检测从 warning
升级为 fail-fast。

**时序契约**（L2 的生命线）：
- `from_pretrained`（`fsdp/omni_impl.py:185`）发生在 `configure_model`
  **之前** → 改类/模块级行为的补丁必须放包 import 期（patches.py 模块级）
- 实例级补丁（打在加载后的模型对象上）放 `configure_model`——
  补丁放哪一层由**作用对象的生命周期**决定

**边界纪律**：只打第三方（transformers / vllm / checkpoint remote code）。
verl-omni 自己的问题不允许 L2——那是 L3 或上游 PR 的领地（否则
"零侵入 verl-omni"就名存实亡）。

### 1.3 L3 gate patch：源码树 diff + 环境变量门

**机制本质**：`git apply` 到上游源码树；patch 内部用环境变量做门——
未设 = 不执行任何额外代码 = 与上游逐字相同。台账（ledger.md）追踪
每一条，硬上限 ≤5 条。现状：GP-004 已退役（vllm-omni 原生插件组
取代），剩 GP-001~003，全部在 verl / verl-omni 侧。

L3 存在的唯一理由：**上游没有扩展点、且 L2 够不着**（例如需要改
模块 import 期之前的行为，或需要上游模块协作的改动）。

### 1.4 为什么是 entry-point（L1 的插件发现问题）

L1 的底层矛盾：**注册是 import 副作用**（`@register` 往每个进程的内存
dict 塞条目），而查表发生在之后的任意时刻——所以必须有人在**每个需要
查表的进程**里、**在查表之前** import 插件模块。问题变成：*"谁来发现
要 import 什么？"* 四个候选通道的对比：

| 通道 | 机制 | 问题 |
|------|------|------|
| ① 改上游 `__init__.py` | 手动加 `from .x import ...` | 违反零侵入（本仓的起源问题：42 行冲突面） |
| ② env var + import_module | `VERL_USE_EXTERNAL_MODULES` 式 | 每个启动脚本都要设；vllm-omni 侧无此通道，只能 GP-004 式 L3 |
| ③ config 路径 | `data.custom_cls: pkg://...` 式 | 逐项配置，用户要枚举每个模块——模型多了不可维护 |
| ④ **entry-point** | `importlib.metadata` 读已安装包元数据 | **pip install 是硬前提**（见下） |

选 ④ 的六个理由：

1. **消灭上游源码编辑**（原始动机）：加模型 = 加目录 + pyproject 一行。
2. **安装即生效、卸载即消失**：`pip install` 一次覆盖全部四个组
   （models/trainers/reward/vllm_omni），无需任何启动脚本配合。
3. **vllm-omni 侧的唯一零 patch 通道**：`load_omni_general_plugins()`
   （`vllm_omni/plugins/__init__.py:61`）**只认** entry points——它没有
   训练侧那样的 `external_lib` config 通道。不用 entry point 就只能
   回到已退役的 GP-004（L3）。
4. **降级隔离**：`_load_all()` 对每个 `ep.load()` 单独 try/except——
   一个模型加载失败只 warning，不拖垮整包。硬编码 import 链是
   一个失败全崩。
5. **可组合**：入口点组是命名空间，**任何包**都能往 `verl_omni.models`
   注册——第三方不必 fork verl_omni_ext 就能加模型。
6. **声明式清单**：pyproject.toml 是唯一事实来源；`_load_all` 的日志
   （`Loaded qwen3_5_moe -> ...`）即装即审。

**机制细节**（解释为什么必须 pip install）：`entry_points()` 读的是
已安装 distribution 的**元数据**（dist-info/EGG-INFO），不走 `sys.path`
的 import 机制——`ep.load()` 才真正 import。PYTHONPATH 挂上的包元数据
不可见，所以 entry point 通道的前提就是安装（这正是 L1-3 限制的由来）。

**代价**（诚实清单）：
- 训练/推理**两个环境**都要装本包（推理侧 vllm-omni 引擎进程独立环境时）
- editable 安装下新增入口点条目要重新 `pip install -e .`（元数据不热刷新）
- 静默降级：加载失败只 warning——部署后要看一眼日志确认全部 Loaded
- 组内重名：两个 distribution 声明同名入口点会都加载，注册表后写者胜
  （`register_pipeline` 对 clobber 有告警）

---

## 二、限制

### 2.1 L1 的限制

| # | 限制 | 表现 | 缓解 |
|---|------|------|------|
| L1-1 | **只能扩展上游设计了注册表的地方** | 上游没注册表的角落够不着（verl `main_ppo.py:140` 不认 omni trainer = GP-001 的由来） | 缺扩展点 → 上游 PR；L3 只是过渡 |
| L1-2 | **注册是 import 副作用，依赖拉起路径** | 某进程没走 config external_lib / engine init 路径 → 注册表空 → 查表静默失败 | 上游拉起保证覆盖面；排障用注册表 dump 探针 |
| L1-3 | **entry points 只发现已安装的包** | pip install 是硬前提，PYTHONPATH 不够；训练/推理**两个环境**都要装 | 文档明示；部署 checklist |
| L1-4 | **版本耦合是隐式 ABI** | 注册键（architecture/model_type）、config schema、adapter 方法签名随上游变 | 探针方法论（先测量再适配）；指纹检测 |
| L1-5 | **时序窗口** | 加载前必须生效的行为（from_pretrained 之前）进不了 L1 | 那是 L2 的领地 |

### 2.2 L2 的限制

| # | 限制 | 表现 | 缓解 |
|---|------|------|------|
| L2-1 | **上游重构即碎** | 签名/结构变化，wrapper 悄悄失效或崩 | probe_signature 指纹 + verify_alive watchdog——**检测是事后的，不能预防** |
| L2-2 | **import 期全局副作用** | 补丁进程生命周期内永久，无法干净卸载；多补丁叠加顺序敏感 | 补丁少而小（每模型 1-4 个）；PATCH_STATE 单行日志可审计 |
| L2-3 | **进程覆盖面 = import 链覆盖面** | 漏走 import 链的进程用原生行为 | assert_patch_consensus 多进程证明 + strict 模式 |
| L2-4 | **可调试性差** | 栈里全是 wrapper；类型检查器不可见 | 补丁体保持薄（逻辑进插件自己的函数，wrapper 只转发） |
| L2-5 | **打不了不存在的对象** | 惰性 import、C 扩展内部、运行时生成的对象 | 无——这类需求直接 L3/上游 PR |
| L2-6 | **边界自律是约定不是机制** | "只打第三方"靠 review 维持 | guardrails skill + 台账纪律 |

### 2.3 L3 的限制

| # | 限制 | 表现 |
|---|------|------|
| L3-1 | **升级脆弱** | git apply 对上下文漂移敏感——上游小重构 patch 就 apply 失败 |
| L3-2 | **部署负担** | 每台机器/容器要打补丁；不可 pip 分发；容器重建即丢 |
| L3-3 | **审查盲区** | 绕过包管理——依赖扫描/code review 看不见 |
| L3-4 | **上限是自律** | ≤5 条是纪律约定，没有技术强制 |

GP-004 的教训是 L3 限制的活案例：patch 针对的文件路径后来被上游
移动（`vllm_omni/registry.py` → `model_executor/models/registry.py`），
且补丁本身的机制设计有 bug（元组路径拼接）——**如果当时发现了原生
插件组，这条 L3 根本不该存在**。推论：**每一条 L3 都应该先穷尽
"上游是否已有原生机制"再立项**（这次是复查 `vllm_omni/plugins/`
目录才发现的，100 行小文件最容易漏）。

### 2.4 三层都解不了的（本质边界）

这些不是实现缺陷，是架构边界——需要不同的工具：

1. **上游没有的输出接口**（全双工 6 缺口）：per-token logprobs、
   wall-clock 时间戳、barge_in_token_index……回放式范式把 5/6 归约到
   L1（训练侧重算 + 环境注入），但**交互式 RL** 最终需要上游 PR
   （GP-005 候选清单已在缺口文档）。插件不能凭空让引擎输出它不产出的东西。
2. **权重同步的 KV-version 一致性**：episode 边界对齐（close_session →
   update_weights）是**协议纪律**，上游没有强制的版本栅栏——违反了
   没有机制拦截，只有轨迹里的 weight_version 戳事后审计。
3. **性能边界**：episode 边界同步的吞吐代价、`max_sessions=4` 的会话
   并发（部署配置可调但受显存约束）、三阶段流水的延迟下限。
4. **多阶段 logprob 重算的结构复杂度**：talker 依赖 thinker hidden
   states（不只 token）——三阶段全量重算链是模型结构决定的，插件层
   无法简化，只能战略选择（MVP：只训 thinker，talker mask=0，
   code2wav 当渲染器）。
5. **跨进程一致性的物理下限**：consensus 检查能*发现*不一致，不能
   *防止*——防止靠上游拉起拓扑的正确性，这不在插件控制范围内。

---

## 三、一张图总结

```
           原理                          限制的本质
┌─────────────────────────┐   ┌──────────────────────────────┐
│ L1 注册表注入 + 上游拉起  │   │ 扩展点清单由上游决定           │
│   import 副作用塞 dict    │   │ pip install 硬前提            │
│   查表点与原生无差别      │   │ 版本耦合是隐式 ABI            │
├─────────────────────────┤   ├──────────────────────────────┤
│ L2 setattr + 工程包装     │   │ 上游重构即碎（检测是事后的）    │
│   幂等/指纹/consensus/    │   │ import 期全局副作用           │
│   watchdog/strict        │   │ 进程覆盖面 = import 链        │
├─────────────────────────┤   ├──────────────────────────────┤
│ L3 git apply + env 门    │   │ 升级脆弱/部署负担/审查盲区     │
│   台账 ≤5，穷尽 L1 先     │   │ （GP-004 教训：先查原生机制）  │
└─────────────────────────┘   └──────────────────────────────┘
        三层都解不了：上游缺的输出接口 / KV-version 纪律 / 性能下限 /
        多阶段重算复杂度 / 跨进程一致性的物理下限
```

---

## 参考（源码行号）

- `import_external_libs` 本体：`verl/utils/import_utils.py:81-88`
- worker 侧拉起点：`verl_omni/workers/config/omni/model.py:143`（config 解析时）
- 注册表查询：`verl_omni/pipelines/model_base.py:545-552`（get_class_by_name）
- `_load_all` 入口点遍历：`verl_omni_ext/__init__.py`（降级策略：单模块失败不拖垮）
- patchkit 原语：`verl_omni_ext/_patchkit.py`（idempotent_patch:151 /
  consensus:335 / watchdog:393 / strict:41）
- 补丁实例（类级 + 实例级的分层）：`models/qwen3_5_moe/patches.py`
- L3 台账：`verl_omni_ext/gates/ledger.md`（GP-004 退役记录）
