"""
Omni-Flow 回放式 duplex rollout（L1 插件）

论文（arXiv 2604.27393）的全双工交互在 RL 里的 rollout 形态：
**回放式 episode**——环境 = 录制的时间索引 (env-visual, env-audio) 流，
策略 = 模型每 chunk 的 listen/speak 决策与内容。

    open_session(chunk_period_ms=200)
      └─ for k in 0..K-1:
           push_chunk(env_visual[k], env_audio[k])    # /v1/duplex input append
           collect o_k（控制位 + 内容 + speech delta + barge-in 事件）
      └─ close_session() → trajectory 归档

对接 vllm-omni（PR #3907 已合并主干，零新增 gate patch）：
    - 会话生命周期 / barge-in epoch / playback cursor：vllm-omni
      experimental/fullduplex/{openai,engine} 管理
    - 本 worker 只做"环境回放 + 轨迹收集"的编排
    - WS 客户端复用 experimental/fullduplex/request_client.py 的能力；
      无 vllm-omni 环境时可用 ReplayDuplexClient 接口做离线单测

权重同步契约（episode 边界对齐，见 docs/feature_fullduplex_omniflow.md §七）：
    episode 是有限窗口；close_session 释放全部 KV lease → update_weights →
    下个 episode 用新权重 open_session。weight_version 随轨迹记录，
    训练侧校验 batch 内版本一致。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from verl_omni_ext.features.fullduplex.omniflow_dataset import (
    CONTROL_LISTEN,
    CONTROL_SPEAK,
    ChunkSpec,
)

logger = logging.getLogger(__name__)


# ============================================================================
# duplex session 客户端协议（对接 vllm-omni /v1/duplex）
# ============================================================================

class DuplexSessionClient(Protocol):
    """vllm-omni 全双工会话客户端的最小协议。

    真实实现：包装 vllm_omni.experimental.fullduplex.request_client
    （WS /v1/duplex）。测试实现：ReplayDuplexClient。
    """

    def open_session(self, session_config: dict[str, Any]) -> str:
        """打开会话，返回 session_id。"""
        ...

    def push_chunk(
        self, session_id: str,
        visual: list[Any] | None, audio: Any | None,
    ) -> dict[str, Any]:
        """推一个时间块的环境输入，返回模型本块的输出增量。

        返回结构：
            {
              "control": "<|listen|>" | "<|speak|>",
              "text_tokens": [...],      # speak 时
              "speech_tokens": [...],    # TAIL 归属本块
              "events": [...],           # barge_in / playback_ack / cancel
              "latency_ms": float,
            }
        """
        ...

    def barge_in(self, session_id: str, scope: str = "current") -> int:
        """注入 barge-in（回放环境里用户开始说话的时刻），返回新 epoch。"""
        ...

    def close_session(self, session_id: str) -> None:
        """关闭会话（释放 KV lease——权重同步的前提）。"""
        ...


class ReplayDuplexClient:
    """离线回放客户端（单测/无 GPU 排障用）。

    不起 vllm-omni——直接用标注的 ground truth 序列模拟模型输出。
    用途：验证 episode 编排、轨迹结构、barge-in 注入的时序逻辑，
    以及 reward 计算（rewards.py 单测的配套）。
    """

    def __init__(self, ground_truth_chunks: list[ChunkSpec] | None = None,
                 control: str = CONTROL_SPEAK, barge_at: int | None = None):
        self.gt = ground_truth_chunks or []
        self.control = control
        self.barge_at = barge_at
        self._sessions: dict[str, dict[str, Any]] = {}
        self._epoch = 0

    def open_session(self, session_config: dict[str, Any]) -> str:
        sid = f"replay_{int(time.time() * 1000) % 100000}"
        self._sessions[sid] = {"k": 0, "closed": False}
        return sid

    def push_chunk(self, session_id: str,
                   visual: list[Any] | None, audio: Any | None) -> dict[str, Any]:
        s = self._sessions[session_id]
        k = s["k"]
        s["k"] += 1
        events: list[dict[str, Any]] = []
        if self.barge_at is not None and k == self.barge_at:
            self._epoch += 1
            events.append({"type": "barge_in", "epoch": self._epoch, "t_ms": k * 200.0})
        # 从 ground truth 或固定控制位模拟模型决策
        if self.gt and k < len(self.gt):
            g = self.gt[k]
            control = g.control
            text = list(g.text_tokens)
            speech = list(g.speech_tokens)
        else:
            control = self.control
            text = ["x"] if control == CONTROL_SPEAK else []
            speech = list(text)
        return {
            "control": control,
            "text_tokens": text,
            "speech_tokens": speech,
            "events": events,
            "latency_ms": 50.0,
        }

    def barge_in(self, session_id: str, scope: str = "current") -> int:
        self._epoch += 1
        return self._epoch

    def close_session(self, session_id: str) -> None:
        self._sessions[session_id]["closed"] = True


# ============================================================================
# 轨迹结构
# ============================================================================

@dataclass
class DuplexTrajectory:
    """一条 episode 的完整轨迹（GRPO group 的一条样本）。"""

    episode_id: str
    group_id: str                       # GRPO group：同 episode 的 N 条轨迹共享
    weight_version: str                 # 权重版本戳（episode 边界同步的依据）
    chunk_size_s: float = 0.2
    chunks: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = ""

    def to_reward_input(self) -> list[dict[str, Any]]:
        """转 rewards.py 的 trajectory_chunks 输入格式。"""
        return self.chunks

    def summary(self) -> str:
        n_speak = sum(1 for c in self.chunks if c["control"] == CONTROL_SPEAK)
        n_barge = sum(1 for c in self.chunks
                      for e in c.get("events", []) if e.get("type") == "barge_in")
        return (f"trajectory {self.episode_id}(group={self.group_id}): "
                f"{len(self.chunks)} chunks, speak={n_speak}, barge_in={n_barge}, "
                f"weights={self.weight_version}")


# ============================================================================
# Episode 编排：回放循环
# ============================================================================

def run_replay_episode(
    client: DuplexSessionClient,
    episode_chunks: list[ChunkSpec],
    episode_id: str,
    group_id: str | None = None,
    weight_version: str = "init",
    barge_in_ks: list[int] | None = None,
    session_config: dict[str, Any] | None = None,
) -> DuplexTrajectory:
    """跑一个回放式 episode，产出完整轨迹。

    Args:
        client: duplex 会话客户端（真实 = vllm-omni WS；测试 = ReplayDuplexClient）
        episode_chunks: 环境（录制流）的 ChunkSpec 序列——只消费其中的
            visual_frames / audio_chunks（输入侧）；text/speech 是标注，
            留给 reward 做 ground truth，不推给模型
        episode_id / group_id: GRPO group 标识
        weight_version: 本 episode 用的权重版本戳
        barge_in_ks: 在哪些块注入 barge-in（从标注的用户说话时刻来；
            回放环境里由编排层注入，保证 group 内 N 条轨迹的打断一致）

    时序契约：
        push_chunk(k) → 模型在 chunk k 的上下文里产出 o_k（先感知后输出，
        论文 §3.2），o_k 条件于 g_1..g_k 的全部输入——正是 Omni-Flow 的
        因果结构，也天然匹配 vllm-omni duplex-append 模式。
    """
    cfg = {"model": "omniflow", "chunk_period_ms": int(episode_chunks[0].t_end
             - episode_chunks[0].t_start) if episode_chunks else 200, **(session_config or {})}
    sid = client.open_session(cfg)
    barge_in_ks = set(barge_in_ks or [])
    traj = DuplexTrajectory(
        episode_id=episode_id,
        group_id=group_id or episode_id,
        weight_version=weight_version,
        chunk_size_s=(episode_chunks[1].t_start - episode_chunks[0].t_start)
        if len(episode_chunks) > 1 else 0.2,
        session_id=sid,
    )

    for k, env in enumerate(episode_chunks):
        if k in barge_in_ks:
            client.barge_in(sid)
        out = client.push_chunk(sid, env.visual_frames, env.audio_chunks)
        traj.chunks.append(
            {
                "k": k,
                "t_range": [env.t_start, env.t_end],
                "control": out.get("control", CONTROL_LISTEN),
                "text_tokens": out.get("text_tokens", []),
                "speech_tokens": out.get("speech_tokens", []),
                "latency_ms": out.get("latency_ms", 0.0),
                "events": out.get("events", []),
            }
        )

    client.close_session(sid)
    return traj


def collect_group(
    client: DuplexSessionClient,
    episode_chunks: list[ChunkSpec],
    episode_id: str,
    group_size: int = 4,
    weight_version: str = "init",
    barge_in_ks: list[int] | None = None,
) -> list[DuplexTrajectory]:
    """收集一个 GRPO group：同一录制环境上 N 条采样轨迹。

    GRPO 的 group 语义：同上下文（同环境、同打断时刻）不同采样。
    group_id 统一 = episode_id，advantage 由 verl 的 grpo 估计器组内归一化。
    """
    group_id = f"grp_{episode_id}"
    return [
        run_replay_episode(
            client, episode_chunks,
            episode_id=episode_id, group_id=group_id,
            weight_version=weight_version, barge_in_ks=barge_in_ks,
        )
        for _ in range(group_size)
    ]


# ============================================================================
# verl rollout worker 接入（worker_cls 配置）
# ============================================================================

def build_worker_cls(default_chunk_ms: int = 200):
    """构造可配置的 rollout worker 类（verl worker_cls 动态加载）。

    config（actor_rollout_ref.rollout）：
        worker_cls: pkg://verl_omni_ext.features.fullduplex.duplex_rollout
        fullduplex:
          episode_max_chunks: 150      # episode 窗口（30s @0.2s）
          group_size: 4                # GRPO group
          chunk_period_ms: 200         # 与 omniflow_dataset 的 chunk_size_s 一致

    worker 的完整 Ray actor 化依赖运行时 verl 版本的 worker 基类（见
    async_worker.py 的骨架说明）；本函数返回的类实现 generate_sequences
    协议，编排层（trainer 的 llm_client.submit）按 batch 驱动。
    """

    class DuplexSessionRolloutWorker:
        """全双工 rollout worker：episode 驱动 + 轨迹收集。"""

        def __init__(self, config: Any):
            self.config = config
            fd = getattr(getattr(getattr(config, "actor_rollout_ref", None),
                                 "rollout", None), "fullduplex", None) or {}
            self.episode_max_chunks = fd.get("episode_max_chunks", 150)
            self.group_size = fd.get("group_size", 4)
            self.chunk_period_ms = fd.get("chunk_period_ms", default_chunk_ms)
            self._client: DuplexSessionClient | None = None

        def _ensure_client(self) -> DuplexSessionClient:
            if self._client is None:
                # 生产路径：包装 vllm-omni 的 duplex request client
                # （延迟 import——无 vllm-omni 环境时本模块可导入可单测）
                from verl_omni_ext.features.fullduplex._vllm_omni_bridge import (
                    VllmOmniDuplexClient,
                )
                self._client = VllmOmniDuplexClient()
            return self._client

        def generate_sequences(self, batch: dict[str, Any], **kwargs) -> list[DuplexTrajectory]:
            """batch 的 episodes → GRPO group 轨迹列表。

            batch 来自 omniflow_collate_fn（槽位④）：
                {"episodes": [{episode_id, group_id, chunks, ...}], ...}
            """
            client = kwargs.get("client") or self._ensure_client()
            weight_version = batch.get("weight_version", "init")
            out: list[DuplexTrajectory] = []
            for ep in batch.get("episodes", []):
                chunks = ep["chunks"][: self.episode_max_chunks]
                # 打断时刻：标注里用户开始说话的块（t_start 处有新 audio 段
                # 且上一段已结束）——由数据管道预计算为 barge_in_ks 字段更佳，
                # 这里给默认推导
                barge_ks = ep.get("barge_in_ks") or _infer_barge_in_ks(chunks)
                out.extend(
                    collect_group(
                        client, chunks,
                        episode_id=ep["episode_id"],
                        group_size=ep.get("group_size", self.group_size),
                        weight_version=weight_version,
                        barge_in_ks=barge_ks,
                    )
                )
            return out

    return DuplexSessionRolloutWorker


def _infer_barge_in_ks(chunks: list[ChunkSpec]) -> list[int]:
    """从环境块推导 barge-in 时刻（默认策略，数据管道可覆盖）。

    推导规则：用户音频段从标注 text 的"上一 assistant 输出块之后"开始——
    简化为：ground truth 控制位从 speak → listen 的转换块（assistant 停下、
    用户即将说话）。真实数据管道应直接标注 barge_in_ks。
    """
    ks: list[int] = []
    prev_speak = False
    for c in chunks:
        if prev_speak and c.control == CONTROL_LISTEN:
            ks.append(c.k)
        prev_speak = c.control == CONTROL_SPEAK
    return ks


if __name__ == "__main__":
    # 自检：用 ReplayDuplexClient 跑一个 8 块 episode、group_size=2
    from verl_omni_ext.features.fullduplex.omniflow_dataset import (
        make_episode_row, serialize_episode,
    )

    demo = make_episode_row(
        episode_id="ep_demo",
        visual=[{"t_start": 0.0, "t_end": 0.2, "frames": ["f0"]}],
        audio=[{"t_start": 0.0, "t_end": 1.6, "samples": [0] * 25600}],
        text=[{"t_start": 0.4, "t_end": 0.8, "tokens": ["你", "好", "呀"]}],
    )
    env = serialize_episode(demo)
    client = ReplayDuplexClient(ground_truth_chunks=env, barge_at=4)
    group = collect_group(client, env, "ep_demo", group_size=2,
                          weight_version="step_0", barge_in_ks=[4])
    for t in group:
        print(t.summary())
        for c in t.chunks[:5]:
            print(f"  k={c['k']} control={c['control']} events={c['events']}")