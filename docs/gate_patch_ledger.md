# L3 Gate Patch 台账

> 硬性上限：≤ 5 条。超了说明扩展点不够，该给上游提 issue 要扩展点。

---

## 当前台账

| ID | 文件:行 | 现象 | gate 变量 | 上游 issue/PR | 状态 |
|----|---------|------|-----------|---------------|------|
| GP-001 | `verl/trainer/main_ppo.py:140` | omni trainer 未注册（手加了 `import verl_omni.trainer.omni`） | 无（无条件） | 未提 | ⚠ 最紧急，未纳管 |
| GP-002 | `verl_omni/workers/rollout/utils.py:263` | MoE weight_loader 丢失（`_attach_moe_weight_loaders`） | `VERL_OMNI_MOE_LOADER_FIX` | 应提 | 待办 |
| GP-003 | `verl_omni/workers/rollout/vllm_omni_async_server.py` | `additional_config` 提升 | 无 | 应提 | 待办 |

---

## 准入规则

1. **判据**：确实改了 verl-omni 源码，且 5 个扩展点够不着
2. **必须有 gate 变量**（默认 off），除非像 GP-001 那样无法条件化
3. **必须有上游 PR 计划**（销账计划）
4. **关闭时行为与上游逐字相同**——这是合并时可验证的前提

---

## 合并验证流程

```bash
# gate 关掉，跑上游测试 → 断言没破坏上游
VERL_OMNI_GATES=off pytest tests/

# gate 打开，跑你的测试 → 断言行为还在
VERL_OMNI_GATES=on pytest tests/

# ext 仓测试 → 断言扩展点契约没变
cd ../verl-omni-ext && pytest tests/
```

gate 的真正价值：把"合并对不对"的模糊判断变成**两个可执行断言**。

---

## 上游跟随工作流

```bash
cd verl-omni-vendor
git fetch upstream
git checkout upstream/main && git pull      # 镜像分支保持干净
git checkout vendor/main
git rebase upstream/main                     # 只处理 ≤5 个 gate patch 的冲突

git config rerere.enabled true               # 重复冲突只解一次
```

---

## 已消除的项

| 原问题 | 消除方式 |
|--------|---------|
| `pipelines/__init__.py` 注释掉 8 个上游 pipeline | 改成 try/except 降级 import |
| `models/transformers/__init__.py` 的 42 行 import 拼接 | 入口点自动发现（`entry_points`） |
