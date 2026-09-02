# 迁移路线

> 从"适配代码混在 verl-omni 树里"到"独立 ext 仓 + 三层分治"的 6 步。
> 按优先级排序，可增量执行。1、2 是止血，必须先做。

---

## 步骤 1：补 L3 台账（半天）

把当前 3 条隐性改动登记、加 gate。

| ID | 文件:行 | 现象 | gate 变量 | 状态 |
|----|---------|------|-----------|------|
| GP-001 | verl/trainer/main_ppo.py:140 | omni trainer 未注册 | 无（无条件） | 最紧急 |
| GP-002 | verl_omni/workers/rollout/utils.py:263 | MoE weight_loader 丢失 | VERL_OMNI_MOE_LOADER_FIX | 应提上游 |
| GP-003 | verl_omni/workers/rollout/vllm_omni_async_server.py | additional_config 提升 | 无 | 应提上游 |

**收益**：立即消除最大的合并盲区。

---

## 步骤 2：仓库卫生（半天）

### 2.1 加 .gitignore，剔除二进制垃圾

d78a8b7 提交了 205 个文件，包含 `kernel_meta/` 下的 `.o`/`.json`（Ascend 算子编译产物）和几十个 1.6~3.3MB 的 `exception_info.*`。

```gitignore
# Ascend 算子编译产物
kernel_meta/
exception_info.*/
*. exception_info
```

用 `git filter-repo` 从历史里剔掉，或重整时起干净分支。

### 2.2 pipelines/__init__.py 注释改 try/except

```python
# 旧：注释掉 8 个上游 pipeline（每次 rebase 都冲突）
# from . import (
#     #bagel_flow_grpo,
#     #ltx2_flow_grpo,
#     minicpmo_5_0,
#     qwen35_moe,

# 新：try/except 降级（对上游文件改动归零）
for _pipeline in ["bagel_flow_grpo", "ltx2_flow_grpo", ...]:
    try:
        importlib.import_module(f".{_pipeline}", package=__name__)
    except ImportError as e:
        logger.warning("pipeline %r unavailable: %s", _pipeline, e)
```

### 2.3 修复 verl 仓 git 状态

`verl/trainer/main_ppo.py:140` 的手改未纳管（dubious ownership），升级 verl 即丢失。
修复 git 可用性 → 做成带说明的 patch 文件 → 启动脚本加自检。

**收益**：让后续任何策略可执行。

---

## 步骤 3：入口点自动发现（1 天）

见 [架构设计 §四](plugin_architecture_design.md#四入口点自动发现消灭-42-行核心编辑)。

```toml
# verl_omni_ext/pyproject.toml
[project.entry-points."verl_omni.models"]
qwen3_5_moe  = "verl_omni_ext.models.qwen3_5_moe"
minicpmo_5_0 = "verl_omni_ext.models.minicpmo_5_0"
```

```python
# verl_omni_ext/__init__.py
def _load_all():
    for ep in entry_points(group="verl_omni.models"):
        try:
            ep.load()
        except Exception as e:
            logger.warning("model plugin %r unavailable: %s", ep.name, e)
```

**收益**：对 verl-omni 上游文件的编辑归零。团队并行开发冲突面归零。

---

## 步骤 4：拆出独立仓（1~2 天）

把 A 类文件整体搬到 `verl-omni-ext` 仓：

```
verl_omni_ext/
├── __init__.py          # _load_all()
├── _patchkit.py         # L2 公共基建
├── models/
│   ├── qwen3_5_moe/     # adapter.py + rollout.py + patches.py + dataset.py
│   └── minicpmo_5_0/   # 同上
├── examples/
├── tests/
└── pyproject.toml
```

启动侧唯一变化：
```bash
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext
```

**收益**：verl-omni 本体保持零改动跟随上游；ext 仓只依赖公开扩展点。

---

## 步骤 5：沉淀模板（1 天）

把 MiniCPM-o 的经验做成 cookiecutter：

- **探针脚本模板**：`v0_forward_signature.sh` / `v0_position_ids.sh` / `v0_mrope.sh` / `v0_chat_template.sh` ...
  - 适配决策是被**测量**出来的，不是被**猜**出来的
  - 先写探针把 6~10 个关键事实测出来，再写适配器
- **标注式脚本移植法**：从最近的模型移植，每处改动标 `[ModelName]`
- **适配器骨架**：`adapter.py` + `rollout.py` + `patches.py` + `dataset.py`

**收益**：下一个模型的适配周期直接减半。

---

## 步骤 6：CI 漂移检测（半天）

L2 的致命伤是静默失效，所以需要专门的定时任务：

```yaml
# .github/workflows/patch-drift.yml
# 每晚 CI:
#   1. pip install -U transformers vllm
#   2. 逐个调用所有 apply_*() 并断言返回 True
#   3. 断言每个被 patch 的目标仍存在且签名匹配
#   4. 跑 tests/ 的适配器边界 CPU 测试
#   → 任一失败，报警
```

**收益**：把"补丁悄悄失效"变成"补丁失效时有人知道"。

---

## 执行顺序说明

| 步骤 | 工作量 | 收益 | 依赖 |
|------|--------|------|------|
| 1 补台账 | 半天 | 消除最大合并盲区 | 无 |
| 2 仓库卫生 | 半天 | 让后续策略可执行 | 无 |
| 3 入口点 | 1 天 | 冲突面归零 | 2 |
| 4 拆仓 | 1~2 天 | 上游跟随成本→零 | 3 |
| 5 沉淀模板 | 1 天 | 适配周期减半 | 4 |
| 6 CI 漂移 | 半天 | L2 有监控 | 4 |

1、2 是止血——在垃圾文件和隐性补丁存在的情况下做 3、4 会把问题一起搬过去。3 是结构性收益最高的单点改动。4 可以延后——做完 3 后即使不拆仓，冲突面也已经很小了。
