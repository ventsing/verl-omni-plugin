"""
verl-omni-ext: verl-omni 的 out-of-tree 扩展包

三层分治架构（按对象所有权分层）：
  L1 插件 (≥95%)       — 自己的适配代码，走 verl/verl-omni 预留的扩展点
  L2 monkey patch (~4%) — 第三方库/checkpoint remote code
  L3 gate patch (≤1%)  — verl-omni 自身、扩展点够不着的地方

加载方式：
  在启动脚本里设置环境变量：
    export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext

  verl-omni 会在每个 Ray worker 进程里调用 import_external_libs("verl_omni_ext")，
  执行本文件，触发入口点自动发现。

扩展点覆盖：
  模型适配器   — @OmniModelBase.register           → entry_points("verl_omni.models")
  推理拓扑     — @OmniRolloutPipelineBase.register → entry_points("verl_omni.models")
  训练范式     — @register_trainer                 → entry_points("verl_omni.trainers")
  reward 管理器 — @register (reward_loop)          → entry_points("verl_omni.reward")
  优势估计器   — @register_adv_est                 → import 触发（algos/）
  policy loss  — @register_policy_loss             → import 触发（algos/）
  数据集       — data.custom_cls                   → import 触发（models/）
  自定义 worker — worker_cls 配置                   → import 触发（workers/）
"""

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

# 所有入口点组——加新组只需在这里加一行
_ENTRY_POINT_GROUPS = [
    "verl_omni.models",     # 模型适配器 + 推理拓扑
    "verl_omni.trainers",   # 训练范式（如全双工）
    "verl_omni.reward",     # reward 管理器
]


def _load_all() -> None:
    """入口点自动发现：遍历所有入口点组，逐个 import 触发 @register 装饰器。

    每个 import 用 try/except 降级——继承 MiniCPM-o rollout adapter 的正确模式：
    如果运行环境的依赖没装上，硬失败会把已跑通的其他模块一起带下水。
    降级后只 warning，不影响其他模块。

    这取代了旧方案里手动编辑 verl_omni/pipelines/__init__.py 添加
    from .qwen35_moe import ... 的做法。
    """
    try:
        eps = entry_points()
        if hasattr(eps, "select"):
            # Python 3.10+: 用 select 遍历多个组
            groups = {group: eps.select(group=group) for group in _ENTRY_POINT_GROUPS}
        else:
            groups = {group: eps.get(group, []) for group in _ENTRY_POINT_GROUPS}
    except Exception as e:
        logger.error("Failed to enumerate entry_points: %s", e)
        groups = {}

    for group, group_eps in groups.items():
        for ep in group_eps:
            try:
                ep.load()  # import 触发 @register + 模块级 patch
                logger.info("Loaded %s: %s -> %s", group, ep.name, ep.value)
            except Exception as e:
                # 降级：某个模块加载失败不影响其他模块
                logger.warning(
                    "Plugin %r in group %s unavailable (skipped): %s",
                    ep.name, group, e,
                )


# 模块 import 时自动执行——这样 VERL_USE_EXTERNAL_MODULES=verl_omni_ext
# 触发 importlib.import_module("verl_omni_ext") 时就会自动发现所有插件。
_load_all()

# algos 和 workers 不走 entry_points（它们的注册表不在 verl_omni 命名空间下），
# 直接 import 触发 @register_adv_est / @register_policy_loss 装饰器。
from . import algos   # noqa: F401
from . import workers  # noqa: F401
