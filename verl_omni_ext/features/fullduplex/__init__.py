"""
全双工 Omni RL 训练（两种正交的"全双工"）

A. 训练系统级全双工（feature_fullduplex.md）：
   trainer.py + async_worker.py —— 训练和 rollout 并发执行 + 权重同步

B. 交互范式级全双工（feature_fullduplex_omniflow.md，论文 arXiv 2604.27393）：
   omniflow_dataset.py — 时间索引样本 → g_k=[v_k;a_k;o_k] 序列化（槽位④）
   duplex_rollout.py   — 回放式 episode rollout + GRPO group 轨迹收集
   rewards.py          — 决策/及时性/打断/内容四类 reward（槽位⑥）
   _vllm_omni_bridge.py— vllm-omni /v1/duplex 客户端桥（运行时延迟加载）

   其中 B 的 omniflow_dataset / rewards / duplex_rollout 无 verl/vllm-omni
   依赖，可独立单测；trainer / bridge 依赖运行时环境，延迟降级导入。

配置方式（config.yaml，交互范式级）：
  trainer:
    v1:
      trainer_name: omni_fullduplex
      adv_estimator: grpo
  actor_rollout_ref:
    rollout:
      worker_cls: pkg://verl_omni_ext.features.fullduplex.duplex_rollout
      fullduplex:
        episode_max_chunks: 150
        group_size: 4
        chunk_period_ms: 200
  data:
    custom_cls:
      path: pkg://verl_omni_ext.features.fullduplex.omniflow_dataset
      collate_fn: omniflow_collate_fn

详见 docs/feature_fullduplex.md 与 docs/feature_fullduplex_omniflow.md
"""

# --- 无环境依赖的核心模块（单测可跑）---
from . import omniflow_dataset, rewards  # noqa: F401

try:
    from . import duplex_rollout  # noqa: F401  # 纯逻辑，无重依赖
except ImportError as _e:  # pragma: no cover
    import logging as _logging

    _logging.getLogger(__name__).debug("duplex_rollout unavailable: %s", _e)

# --- 依赖 verl/verl-omni 运行时的模块（延迟降级）---
try:
    from . import trainer, async_worker  # noqa: F401
except ImportError as _e:  # pragma: no cover
    import logging as _logging

    _logging.getLogger(__name__).debug(
        "fullduplex trainer/worker unavailable (no verl env): %s", _e
    )
