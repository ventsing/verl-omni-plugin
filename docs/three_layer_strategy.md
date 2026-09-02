# 三层分治策略

> 三者不构成"选一个"的关系。它们的适用对象不同：插件用于你能扩展的地方，
> monkey patch 用于你不拥有的代码，gate patch 用于你能改但上游没给扩展点的地方。

---

## 一、定义

### L1 插件（Plugin）

上游预留扩展点（注册表 / 入口点 / 钩子），你的代码作为独立单元填进去，**不修改上游任何一行源码**。上游"知道"扩展的存在——扩展点是它自己设计的。

verl-omni 的 5 个扩展点都是 L1 的载体。

### L2 Monkey Patch

运行时替换第三方对象的属性/方法。上游不知道你的存在。不改磁盘上的源码，改的是进程内存里的对象。

适用对象：transformers / vllm / checkpoint remote code / 模型实例——你没有 PR 权的地方。

### L3 Gate Patch

直接修改上游源码，但把新行为包在一个**默认关闭**的开关后面：

```python
# verl-omni 上游文件里
if os.environ.get("VERL_OMNI_ASCEND_NZ_FIX", "0") == "1":
    <你的新逻辑>
else:
    <原逻辑，逐字未动>
```

关键性质：关闭时行为与上游逐字相同 → 合并冲突范围小 → 冲突解决的正确性可执行验证。

---

## 二、六维对比

| 维度 | L1 插件 | L2 Monkey Patch | L3 Gate Patch |
|------|---------|-----------------|---------------|
| 上游合并冲突 | ✅ 几乎为零 | ✅ 零 | ⚠️ 有，但被 gate 限制 |
| 上游改了内部实现 | ✅ 契约通常仍成立 | ❌ **静默失效** | ⚠️ 冲突或测试失败 |
| 团队并行开发 | ✅ 一模型一目录 | ⚠️ 全局生效，顺序敏感 | ❌ 多人改同一文件 |
| 可发现性/可调试 | ✅ 栈里有你的类名 | ❌ 最差——grep 找不到 | ✅ 源码里就是 |
| 能力上限 | ❌ 受限于上游扩展点 | ✅ 无限 | ✅ 无限 |
| 回馈上游/销账 | ✅ 天然 | ❌ 永久技术债 | ✅ 去掉 gate 就是一个 PR |

---

## 三、三个反直觉判断

### ① 不是三选一

按对象所有权分层，三者共存。问题从来不是"选哪个"，而是"每类占比压到该有的水平 + 给例外建台账"。

### ② monkey patch 最大的风险不是"脏"，是"静默失效"

上游把被 patch 的函数改名/换签名/删掉 → 补丁还是打上去了 → 不报错 → 但修的已经不是原来那个 bug 了。RL 训练的失效形态是"数字慢慢不对"，不是崩溃，发现成本极高。

**强制 4 条规范（写进 `_patchkit.py`）：**

| 规范 | 作用 |
|------|------|
| (a) 打补丁前断言目标存在且签名符合预期 | 上游改名/删函数时立即响 |
| (b) 记录被 patch 版本的指纹 | 知道在哪个版本区间验证过 |
| (c) 有 CPU 测试验证补丁行为 | 上游修了原 bug 时测试红 |
| (d) 返回 bool，启动脚本断言为 True | 目标不存在时启动 3 秒内退出 |

### ③ gate patch 的价值不在"运行时可切换"，在"合并时可验证"

gate 的真正价值：上游合并冲突时，关掉 gate 跑上游测试（断言没破坏上游行为），再打开跑你的测试（断言行为还在）。把"合并对不对"的模糊判断变成两个可执行断言。

---

## 四、准入规则

### L1 — 默认选择，无需审批

判据：能通过 5 个扩展点表达。新模型适配的 95% 应落在这里。

### L2 — PR 里回答 4 个问题

1. 被打的对象归谁所有？（**必须**是第三方：transformers / vllm / checkpoint remote code。打 verl-omni 自己的东西说明该用 L3 或该给上游提扩展点。）
2. 被替换的原行为是什么，为什么替换是安全的？（论证密度参考 register guard："这个注册本来就是死代码——mapping 按 config class 查，str key 永远命中不了"）
3. 目标不存在或签名变了会怎样？（**必须响**，不能静默）
4. 对应的 CPU 测试在哪？

统一走 `_patchkit.idempotent_patch` 装饰器，强制 (a)~(d)。

### L3 — 台账 + 上游 PR 计划

判据：确实改了 verl-omni 源码，且扩展点够不着。

每条必须登记：`| ID | 文件:行 | 现象 | gate 变量 | 上游 issue/PR | 状态 |`

**硬性上限：≤ 5 条。** 超了说明扩展点不够，该给上游提 issue 要扩展点。

---

## 五、当前补丁归属

| 补丁 | 归入 | 说明 |
|------|------|------|
| `apply_qwen3_5_vision_device_fix` | L2 | 打 transformers/remote code，合规 |
| `apply_minicpmo_auto_register_guard` | L2 | 打 transformers AutoClass，合规 |
| `apply_minicpmo_automodel_fallback` | L2 | 打 transformers，合规 |
| `build_minicpmo_forward_adapter` | L2 | 打模型实例，合规 |
| `build_minicpmo_processor` | **L1** | 其实是槽位①的 configure_processor 正常实现，不算补丁 |
| `_attach_moe_weight_loaders` | **L3 → GP-002** | ⚠️ 直接改了 verl-omni 的 vllm_rollout/utils.py，当前无 gate |
| `vllm_omni_async_server additional_config` | **L3 → GP-003** | 改了 verl-omni 核心文件 |
| `main_ppo.py:140 import` | **L3 → GP-001** | 改的是 verl，且未纳管，最紧急 |
| `pipelines/__init__.py 注释 8 个 pipeline` | **应消除** | 改成 try/except 降级 import |

**关键发现**：有 3 条 L3 改动是隐性的——混在 d78a8b7 那个 205 文件的大 commit 里，没有 gate、没有台账、没有上游计划。下次合并上游时，这三处是最可能出问题且最难定位的地方。
