# verl-omni-ext

verl-omni 的 out-of-tree 模型适配包。**三层分治**架构：插件 / monkey-patch / gate-patch 按对象所有权分层组合。

## 核心判断

> 你的适配代码已经是插件了，只差最后 42 行没有插件化。

实际开发的两次适配（Qwen3.5-MoE、MiniCPM-o）中，**98% 是新增独立文件**，只碰到 2 个上游 `__init__.py`（共 42 行 import 拼接）。这 42 行是唯一需要消灭的冲突面——用**入口点自动发现**归零它。

## 三层分治

| 层次 | 对象 | 手段 | 目标占比 |
|------|------|------|---------|
| **L1 插件** | 自己的模型适配代码 | out-of-tree 包 + 注册表（5 个扩展点） | ≥95% |
| **L2 monkey patch** | 第三方库 / checkpoint remote code | 幂等 + 前置断言 + 版本指纹 + 返回 bool | ~4% |
| **L3 gate patch** | verl-omni 自身、扩展点够不着 | 开关默认 off + 台账 + 上游 PR | ≤1%，每条有销账计划 |

**不是三选一，是按对象所有权分层。** 三者像螺丝刀和扳手——作用对象不同，组合使用。

## verl-omni 的 5 个扩展点

| 槽位 | 注册机制 | 用途 |
|------|---------|------|
| ① OmniModelBase | `@register(architecture, stage)` | 训练侧适配器（thinker/talker） |
| ② OmniRolloutPipelineBase | `@register(model_type)` | 推理侧拓扑适配器 |
| ③ VERL_USE_EXTERNAL_MODULES | 环境变量 → `importlib.import_module` | 注册触发器（接受逗号分隔多模块） |
| ④ data.custom_cls | `pkg://path:ClassName` | 数据集注入 |
| ⑤ examples 脚本 | 启动脚本 | 配置载体 + 前置自检 |

## 加载方式

```bash
# 启动脚本里设置
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext
```

verl-omni 在每个 Ray worker 进程里 `import_external_libs("verl_omni_ext")` → 执行 `verl_omni_ext/__init__.py` 的 `_load_all()` → 遍历 `entry_points("verl_omni.models")` 逐个 import → 触发 `@OmniModelBase.register()`。

## 零侵入边界

| 层 | 零侵入? | 说明 |
|----|---------|------|
| verl-omni 训练侧 | ✅ | 三层分治完全覆盖 |
| verl-omni rollout adapter | ✅ | 槽位②只是转发 vllm-omni 定义 |
| **vllm-omni pipeline/模型** | **✅ gate patch** | **GP-004: 5 行补丁加 VLLM_OMNI_EXTERNAL_MODULES** |
| vllm 平台适配 | ✅ | `vllm.platform_plugins` entry_points |
| vllm 模型加载 | ✅ | `trust_remote_code` 动态加载 |
| vllm weight_loader | ⚠ L2 | monkey patch 打 vllm 对象 |
| 数据处理 | ✅ | 槽位①+④完全覆盖 |

详见 [Rollout 侧适配分析](docs/rollout_adaptation.md) 和 [数据处理 Add-on](docs/data_pipeline.md)。

## 目录结构

```
verl-omni-ext/                        # = 本仓
├── pyproject.toml                     #   入口点声明（models/trainers/reward 三组）
├── verl_omni_ext/
│   ├── __init__.py                    #   _load_all() 多组自动发现
│   ├── _patchkit.py                   #   L2 monkey patch 公共基建
│   ├── models/                        #   【新增模型区域】一模型一独立目录
│   │   ├── qwen3_5_moe/
│   │   │   ├── thinker_adapter.py     #     槽位①: @OmniModelBase.register
│   │   │   ├── rollout_adapter.py     #     槽位②: @OmniRolloutPipelineBase.register
│   │   │   ├── patches.py             #     L2: vision device fix
│   │   │   ├── dataset.py             #     槽位④（按需）
│   │   │   └── vllm_omni/             #     vllm-omni 侧 pipeline（GP-004）
│   │   └── minicpmo_5_0/              #     同上 + 4 个 L2 补丁 + 模块级补丁
│   ├── features/                      #   【跨域特性区域】按功能域组织
│   │   └── fullduplex/                #     全双工 = trainer + worker 协同
│   │       ├── trainer.py             #       @register_trainer("omni_fullduplex")
│   │       └── async_worker.py        #       异步推理 worker
│   ├── reward/                       #   reward 扩展
│   │   ├── managers.py                #     @register reward manager
│   │   └── functions.py               #     custom_reward_function
│   ├── algos/                        #   自定义算法
│   │   ├── adv_est.py                 #     @register_adv_est
│   │   └── policy_loss.py             #     @register_policy_loss
│   ├── trainer/                      #   骨架（单注册表 trainer 可放这）
│   ├── workers/                      #   骨架（单注册表 worker 可放这）
│   ├── probes/                      #   探针（可 import 调用的测量工具）
│   │   ├── forward_signature.py       #     forward 签名探测
│   │   └── processor_whitelist.py     #     processor 白名单探测
│   └── gates/
│       ├── ledger.md                  #   L3 台账（≤5 条）
│       ├── vllm_omni_external_modules.patch  # GP-004
│       └── apply_patches.sh           #   自动 apply 脚本
├── examples/
│   ├── qwen3_5_moe/
│   │   ├── config/*.yaml              #     config 模板
│   │   ├── run_*.sh                   #     启动脚本 + 前置自检
│   │   └── probes/v0_*.sh             #     探针
│   ├── minicpmo_5_0/                  #     标注 6 处 [MiniCPM] diff
│   └── qwen35_whisper_plugin/         #     教学骨架
├── tests/                             #   适配器边界 CPU 测试
└── docs/
    ├── three_layer_strategy.md        #   三层分治详解
    ├── plugin_architecture_design.md  #   架构设计
    ├── rollout_adaptation.md          #   vllm-omni/vllm 适配（零侵入边界）
    ├── data_pipeline.md               #   数据处理 add-on
    ├── feature_fullduplex.md          #   全双工特性添加
    ├── vllm_omni_changes.md            #   跨仓适配记录
    ├── migration_guide.md             #   迁移路线
    └── gate_patch_ledger.md           #   L3 台账规范
```

## 加新模型只需 3 步

```bash
# 1. 加一行入口点声明
echo 'your_model = "verl_omni_ext.models.your_model"' >> pyproject.toml [entry-points]

# 2. 新建一个目录
mkdir verl_omni_ext/models/your_model/
# 写 adapter.py（槽位①）+ rollout.py（槽位②）+ patches.py（L2，如有）

# 3. 从最近的模型移植启动脚本
cp examples/qwen3_5_moe/run_*.sh examples/your_model/
# 改 6 处标注点
```

**不碰任何上游文件。**

## 扩展点覆盖

| 扩展点 | 注册表 | ext 包位置 | entry_points? |
|--------|--------|-----------|---------------|
| 模型 adapter | `@OmniModelBase.register` | `models/` | ✅ `verl_omni.models` |
| 推理 rollout | `@OmniRolloutPipelineBase.register` | `models/` | ✅ `verl_omni.models` |
| 训练范式 | `@register_trainer` | `trainer/` | ✅ `verl_omni.trainers` |
| reward 管理器 | `@register` (reward_loop) | `reward/managers.py` | ✅ `verl_omni.reward` |
| 优势估计器 | `@register_adv_est` | `algos/adv_est.py` | import 触发 |
| policy loss | `@register_policy_loss` | `algos/policy_loss.py` | import 触发 |
| 自定义 reward 函数 | `custom_reward_function` config | `reward/functions.py` | 路径指向 |
| 数据集 | `data.custom_cls` | `models/<m>/dataset.py` | 路径指向 |
| 自定义 worker | `worker_cls` config | `workers/` | 路径指向 |
| L2 monkey patch | `_patchkit.py` | `models/<m>/patches.py` | import 触发 |
| L3 gate patch | 台账 | `gates/ledger.md` | — |

## 文档

- [三层分治策略](docs/three_layer_strategy.md) — 什么放哪一层，为什么
- [架构设计](docs/plugin_architecture_design.md) — 5 扩展点 + 时序陷阱 + 入口点自动发现
- [迁移路线](docs/migration_guide.md) — 6 步从现状到目标
- [L3 gate patch 台账](docs/gate_patch_ledger.md) — 规范 + 当前 3 条隐性改动
- [Rollout 侧适配](docs/rollout_adaptation.md) — vllm-omni / vllm 需要改什么（零侵入边界）
- [数据处理 Add-on](docs/data_pipeline.md) — 6 个数据扩展点 + 静默失败陷阱
- [全双工特性添加](docs/feature_fullduplex.md) — 新训练范式怎么 add-on
- [跨仓适配记录](docs/vllm_omni_changes.md) — vllm-omni 侧每个模型的改动清单

## License

Apache-2.0
