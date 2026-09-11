"""
全双工 omni RL trainer（L1 插件，零侵入）

通过 verl 的 @register_trainer 注册表注册——不修改 verl/verl-omni 任何源码。

两种"全双工"（正交，可叠加）：
  A. 训练系统级（本文件原始职责）：训练和推理真正并发
     传统 RL：训练 → 停 → 推理生成 rollout → 停 → 训练（串行）
     全双工：  训练和推理同时进行，推理结果实时反馈到训练
  B. 交互范式级（Omni-Flow，见 docs/feature_fullduplex_omniflow.md）：
     模型边听边说、可打断、可主动——数据/rollout/reward 在
     omniflow_dataset.py / duplex_rollout.py / rewards.py

verl 已有的异步基础：
  - @register_trainer("separate_async") — 训练和 rollout 分离，可部分重叠
  - @register_trainer("colocate_async") — 训练和 rollout 共置，partial rollout
  - FullyAsyncLLMServerClient — 异步推理服务客户端
  - agent_loop_tq — TransferQueue 异步数据流
  - checkpoint_manager.update_weights() — 权重同步
  - @register_adv_est("grpo") — GRPO 优势估计（Omni-Flow RL 用，论文 §5.4）

本 trainer 继承 OmniPPOTrainerSync，改为异步并发执行。
"""

import logging
import os

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("omni_fullduplex")
class OmniPPOTrainerFullDuplex(OmniPPOTrainerSync):
    """全双工 omni PPO trainer

    与 omni_sync 的区别：
    1. 训练和推理并发（不是串行交替）
    2. 使用 FullyAsyncLLMServerClient 进行异步推理
    3. 权重定期同步（parameter_sync_step 控制频率）
    4. 推理结果实时反馈到训练队列

    Omni-Flow 全双工交互训练（论文 arXiv 2604.27393）的推荐配置：

        trainer:
          v1:
            trainer_name: omni_fullduplex
            adv_estimator: grpo            # 论文 §5.4 用 GRPO
        actor_rollout_ref:
          rollout:
            worker_cls: pkg://verl_omni_ext.features.fullduplex.duplex_rollout
            fullduplex:
              episode_max_chunks: 150      # episode 窗口（30s @0.2s chunk）
              group_size: 4                # GRPO group：同 episode N 条采样
              chunk_period_ms: 200         # 论文 ablation 最优 chunk
        data:
          custom_cls:
            path: pkg://verl_omni_ext.features.fullduplex.omniflow_dataset
            collate_fn: omniflow_collate_fn

    加载链路（零侵入）：
      1. main_omni.py 的 uses_v1_trainer() 返回 True（trainer_type=policy_gradient）
      2. run_ppo() 加载 V1 trainer 系统
      3. @register_trainer("omni_fullduplex") 从 config.trainer.v1.trainer_name 匹配
      4. 本 trainer 被实例化

      全程不需要改 main_omni.py 或 verl 的任何源码——
      @register_trainer 是 verl 的公开扩展点（注册表字典）。
    """

    def get_llm_client(self):
        """使用异步推理客户端（不是同步的）"""
        return self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def on_train_begin(self):
        """预热：往推理队列放 warmup batches，让推理先跑起来"""
        if self.config.skip.rollout_tq.enable:
            return

        num_warmup = self.config.trainer.v1.get("fullduplex", {}).get(
            "num_warmup_batches", 2
        )
        for _ in range(num_warmup):
            self._add_batch_to_generate()

        logger.info(
            f"FullDuplex: added {num_warmup} warmup batches to agent loop. "
            f"Training and inference will run concurrently."
        )

    def on_step_end(self):
        """每 parameter_sync_step 步同步一次权重到推理引擎

        Omni-Flow 的权重同步契约（episode 边界对齐，见
        docs/feature_fullduplex_omniflow.md §七）：
          活跃 duplex session 的 KV 是旧权重算的——权重变了 KV 语义错位。
          rollout 引擎在 episode 边界应用新权重：close_session 释放全部
          KV lease → update_weights → 下个 episode 用新权重 open_session。
          weight_version 随轨迹记录，训练侧校验 batch 内版本一致。

          实现上：duplex_rollout 的 worker 在 episode 间检查权重版本戳，
          本方法的 update_weights 是推新版本；两边在 episode 边界汇合。
        """
        sync_step = self.config.trainer.v1.get("fullduplex", {}).get(
            "parameter_sync_step", 4
        )

        if self.global_steps % sync_step == 0:
            with marked_timer("update_weights", self.timing_raw, color="red"):
                self.checkpoint_manager.update_weights(self.global_steps)
                logger.debug(
                    f"FullDuplex: weights synced at step {self.global_steps}"
                )

    def on_train_end(self):
        """训练结束时清理异步推理"""
        logger.info("FullDuplex: training ended, cleaning up async inference")
        # agent_loop_manager 会自动清理


# ============================================================================
# 推理侧的依赖（vllm-omni experimental/fullduplex）
# ============================================================================
#
# 训练侧零侵入了，推理侧由 vllm-omni 主干承担（PR #3907 已合并）：
#
# vllm_omni/experimental/fullduplex/
#   ├── core/        模型无关契约（DuplexAdapter / session / turn runtime）
#   ├── engine/      scheduler 数据面（session KV lease、barge-in epoch）
#   ├── openai/      WS 传输 / Realtime 投影（/v1/duplex、/v1/realtime?duplex=1）
#   ├── minicpmo45/  MiniCPM-o 4.5 帧协议、listen/speak policy、Stage0
#   └── personaplex/ Moshi 级 lockstep 语音到语音
#
# 本仓库的对接（零新增 gate patch）：
#   _vllm_omni_bridge.py 把 request_client 适配成 DuplexSessionClient 协议；
#   新增全双工模型的 seam 是 core.DuplexAdapter（上游已定义），
#   模型定义经 GP-004（VLLM_OMNI_EXTERNAL_MODULES）注册，照旧。
