"""
自定义 worker 模块

用于需要自定义 worker 行为的场景（如全双工推理 worker）。

verl 的 worker 体系：
  - ActorRolloutRefWorker —— actor + rollout + ref 三合一
  - 分离模式下可拆分为独立 worker

配置方式（config.yaml）：
  actor_rollout_ref:
    rollout:
      worker_cls: verl_omni_ext.workers.async_rollout.AsyncRolloutWorker
"""
from . import async_rollout  # noqa: F401
