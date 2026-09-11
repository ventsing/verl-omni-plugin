"""
Omni-Flow 全双工 reward（L1 插件，槽位⑥）

论文（arXiv 2604.27393）§5.4 的 reward 配方 + 全双工交互特有信号。
四类 reward，加权组合，权重从 config 暴露：

  ┌─────────────┬────────────────────────────────┬──────────────┐
  │ 类别         │ 信号                            │ 论文/来源     │
  ├─────────────┼────────────────────────────────┼──────────────┤
  │ 决策恰当性    │ 该说才说（漏说/抢话都罚）          │ LS 范式 §3.3 │
  │ 及时性        │ TAIL 对齐度 + barge-in 停止延迟   │ TAIL §3.4    │
  │ 打断处理      │ 被打断后 ≤N 块内转 listen         │ RFC #3745    │
  │ 内容质量      │ accuracy + format + smooth length│ RL §5.4     │
  └─────────────┴────────────────────────────────┴──────────────┘

注册为 verl reward manager（决策/及时性/打断 = 轨迹级，快）；
内容质量走 custom_reward_function（文本级，可能要 judge model）。

接入（config.yaml）：

    custom_reward_function:
      path: pkg://verl_omni_ext.features.fullduplex.rewards
      name: omniflow_content_score
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 论文/RFC 的延迟目标：barge-in 后 < 400ms v2v
DEFAULT_BARGEIN_BUDGET_MS = 400.0
# 论文 §5.4：length reward 前 480 步不加（先收敛）——由 trainer 传 step 控制
DEFAULT_LENGTH_WARMUP_STEPS = 480


# ============================================================================
# 1. 决策恰当性：LS 控制位 vs 标注 ground truth
# ============================================================================

@dataclass
class DecisionMetrics:
    """LS 决策与标注的混淆矩阵计数。"""

    true_speak: int = 0      # 标注有输出 & 模型 speak（对）
    true_listen: int = 0     # 标注静默 & 模型 listen（对）
    missed_speak: int = 0    # 标注有输出 & 模型 listen（漏说）
    false_speak: int = 0     # 标注静默 & 模型 speak（抢话）

    @property
    def total(self) -> int:
        return self.true_speak + self.true_listen + self.missed_speak + self.false_speak

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.true_speak + self.true_listen) / self.total


def decision_reward(
    trajectory_chunks: list[dict[str, Any]],
    ground_truth_chunks: list[dict[str, Any]],
) -> tuple[float, DecisionMetrics]:
    """决策恰当性：模型控制位 vs 标注控制位。

    Args:
        trajectory_chunks: rollout 轨迹（duplex_rollout.py 产出），
            每块 {k, control, ...}
        ground_truth_chunks: 标注序列化（omniflow_dataset.serialize_episode 产出，
            ChunkSpec 转的 dict，或同构 dict），每块 {k, control}

    Returns:
        (reward, metrics)：reward ∈ [-1, 1]，准确率的重心映射——
        抢话比漏说更伤交互体验（用户被打断），权重更高。
    """
    gt = {c["k"]: c for c in ground_truth_chunks}
    m = DecisionMetrics()
    for c in trajectory_chunks:
        k = c["k"]
        model_speak = c.get("control") == "<|speak|>"
        gt_c = gt.get(k)
        gt_speak = bool(gt_c and gt_c.get("control") == "<|speak|>") if gt_c else False
        # 标注缺失的块（episode 尾部超出标注范围）：不奖不罚
        if gt_c is None:
            continue
        if gt_speak and model_speak:
            m.true_speak += 1
        elif not gt_speak and not model_speak:
            m.true_listen += 1
        elif gt_speak and not model_speak:
            m.missed_speak += 1
        else:
            m.false_speak += 1

    if m.total == 0:
        return 0.0, m
    # 抢话惩罚 ×1.5（打断用户比沉默更糟）
    score = (m.true_speak + m.true_listen - 1.5 * m.false_speak - m.missed_speak) / m.total
    return max(-1.0, min(1.0, score)), m


# ============================================================================
# 2. 及时性：TAIL 对齐度 + barge-in 停止延迟
# ============================================================================

def timeliness_reward(
    trajectory_chunks: list[dict[str, Any]],
    chunk_size_s: float = 0.2,
) -> float:
    """TAIL 对齐度：每 speak 块的文本量 vs 播放进度的偏差。

    论文 §3.4：第 k 块生成的文本经 vocalize 后应逼近时间边界 k·t。
    轨迹里 speech_tokens 的数量是播放进度的直接代理（每 speech token
    ≈ 固定播放时长），text token 过多 = 文本跑在语音前面（过期内容），
    过少 = 语音断流。

    Returns:
        reward ∈ [-1, 1]：块内 |text_speech_ratio - 1| 的负均值。
        无 speak 块 → 0（不奖不罚，交给决策项判断该不该说）。
    """
    deltas = []
    for c in trajectory_chunks:
        if c.get("control") != "<|speak|>":
            continue
        n_text = len(c.get("text_tokens", []))
        n_speech = len(c.get("speech_tokens", []))
        if n_speech == 0:
            # 说了话但没产出 speech token（尾块 look-ahead 推迟是正常的，
            # 累计误差由相邻块补偿——这里只看有 speech 的块）
            continue
        # ratio > 1：文本积压（语音滞后）；< 1：语音断流
        ratio = n_text / max(n_speech, 1)
        deltas.append(abs(ratio - 1.0))
    if not deltas:
        return 0.0
    return max(-1.0, min(1.0, 1.0 - 2.0 * (sum(deltas) / len(deltas))))


def bargein_latency_penalty(
    trajectory_chunks: list[dict[str, Any]],
    budget_ms: float = DEFAULT_BARGEIN_BUDGET_MS,
    chunk_size_s: float = 0.2,
) -> float:
    """barge-in 响应延迟：用户开始说话后模型停止的延迟（论文/RFC <400ms）。

    时序语义：barge-in 在块 i 的事件里被检测到 → 模型最早的反应点是
    块 i+1（块 i 的输出已在本块上下文里定出）。块 i+1 即转 <|listen|>
    = 0 额外延迟（一个 chunk 周期是基本粒度，不算罚）。

    penalty = -min(latency / budget, 1)：延迟达到 budget（默认 400ms）即饱和 -1。
    """
    chunk_ms = chunk_size_s * 1000.0
    worst = 0.0
    for i, c in enumerate(trajectory_chunks):
        events = c.get("events", [])
        if not any(e.get("type") == "barge_in" for e in events):
            continue
        # 从下一块起找第一次转 listen（i+1 是最早反应点）
        stop_k = None
        for j in range(i + 1, len(trajectory_chunks)):
            if trajectory_chunks[j].get("control") == "<|listen|>":
                stop_k = j
                break
        if stop_k is None:
            latency = budget_ms * 4  # 剩余 episode 都没停：超出饱和值
        else:
            latency = (stop_k - (i + 1)) * chunk_ms
        worst = max(worst, min(latency / budget_ms, 1.0))
    return -worst


# ============================================================================
# 3. 打断处理：被 barge-in 后的 epoch 一致性
# ============================================================================

def interruption_reward(
    trajectory_chunks: list[dict[str, Any]],
    grace_chunks: int = 1,
) -> float:
    """被打断后的行为正确性（RFC #3745 epoch 语义）。

    时序语义：barge-in 在块 k 的事件里到达——块 k 自身的输出已定，
    不罚（打断发生不是模型的错）。宽限窗口 = 块 k+1..k+grace_chunks：
      窗口内转 <|listen|> → +1；继续 speak 且 epoch 未更新（说旧内容）→ -1。
    轨迹事件带 {type: "barge_in", epoch}，块带 {epoch}。
    """
    rewards = []
    last_barge_k = None
    last_epoch = None
    for c in trajectory_chunks:
        for e in c.get("events", []):
            if e.get("type") == "barge_in":
                last_barge_k = c["k"]
                last_epoch = e.get("epoch")
        if last_barge_k is None:
            continue
        lag = c["k"] - last_barge_k
        if 1 <= lag <= grace_chunks:
            # 宽限窗口内：转 listen = 好；继续 speak 旧 epoch 内容 = 坏
            if c.get("control") == "<|listen|>":
                rewards.append(1.0)
            elif c.get("epoch", last_epoch) == last_epoch:
                rewards.append(-1.0)  # 还在说旧 epoch 的内容
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


# ============================================================================
# 4. 内容质量：论文 §5.4 配方
# ============================================================================

def _smooth_length_reward(
    lengths: list[int], correctness: list[bool],
    len_min: int | None = None, len_max: int | None = None,
    tau: float = 8.0,
) -> list[float]:
    """Kimi-K1.5 smooth length reward（论文式(1)）。

    答对：短者奖励 s_i；答错：只罚不奖 min(0, s_i)。
    s_i = (0.5 - (l_i - l_min)/(l_max - l_min)) * min(1, (l_max-l_min)/tau)
    """
    if not lengths:
        return []
    lo = min(lengths) if len_min is None else len_min
    hi = max(lengths) if len_max is None else len_max
    out = []
    for l, r in zip(lengths, correctness):
        if hi <= lo:
            out.append(0.0)
            continue
        s = (0.5 - (l - lo) / (hi - lo)) * min(1.0, (hi - lo) / tau)
        out.append(s if r else min(0.0, s))
    return out


def omniflow_content_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """custom_reward_function 入口：内容质量（文本级）。

    verl 的 custom_reward_function 签名约定。对全双工轨迹：solution_str 是
    模型整个 episode 的文本输出拼接，ground_truth 是标注文本。

    组成：rule-based 准确率（子串/关键词匹配）+ format（控制位规整度）。
    judge model / RM 打分由上层 trainer 在有 GPU 资源时另行注入——
    本函数保持纯 CPU、无外部依赖（dataloader worker 里可跑）。
    """
    extra_info = extra_info or {}
    weights = extra_info.get("content_weights", {})
    w_acc = weights.get("accuracy", 0.7)
    w_fmt = weights.get("format", 0.3)

    # --- accuracy：rule-based（关键词命中；judge model 版本见上层注入）---
    if not ground_truth:
        acc = 1.0 if not solution_str.strip() else 0.0
    else:
        keys = [k for k in re.split(r"\s+", ground_truth.strip()) if k]
        if not keys:
            acc = 1.0
        else:
            hit = sum(1 for k in keys if k in solution_str)
            acc = hit / len(keys)

    # --- format：控制位规整度（<|speak|> 后必须有内容，<|listen|> 后必须干净）---
    chunks = extra_info.get("chunks")
    if chunks:
        fmt_scores = []
        for c in chunks:
            if c.get("control") == "<|speak|>":
                fmt_scores.append(1.0 if c.get("text_tokens") else 0.0)
            else:
                fmt_scores.append(1.0 if not c.get("text_tokens") else 0.0)
        fmt = sum(fmt_scores) / len(fmt_scores)
    else:
        fmt = 1.0  # 无轨迹信息时不惩罚

    return float(w_acc * acc + w_fmt * fmt)


# ============================================================================
# 组合：轨迹级 reward manager（槽位⑥注册）
# ============================================================================

@dataclass
class OmniFlowRewardConfig:
    """四类 reward 的权重（config: trainer.v1.fullduplex.reward_weights）"""

    decision: float = 0.35
    timeliness: float = 0.25
    interruption: float = 0.15
    content: float = 0.25
    length_warmup_steps: int = DEFAULT_LENGTH_WARMUP_STEPS  # 论文：前480步不加 length


def omniflow_trajectory_reward(
    trajectory_chunks: list[dict[str, Any]],
    ground_truth_chunks: list[dict[str, Any]],
    solution_str: str = "",
    ground_truth_str: str = "",
    config: OmniFlowRewardConfig | None = None,
    global_step: int = 0,
    chunk_size_s: float = 0.2,
) -> dict[str, float]:
    """全双工轨迹的综合 reward（决策/及时性/打断 + 内容）。

    GRPO 场景：同一 episode（group）内 N 条轨迹各自调用本函数，
    group 内归一化由 verl 的 grpo adv_estimator 完成。

    Returns:
        {"total": ..., "decision": ..., "timeliness": ..., "interruption": ..., "content": ...}
        分项一并返回——便于 tensorboard 分通道监控。
    """
    cfg = config or OmniFlowRewardConfig()
    dec, _m = decision_reward(trajectory_chunks, ground_truth_chunks)
    tim = timeliness_reward(trajectory_chunks, chunk_size_s)
    itp = interruption_reward(trajectory_chunks)
    content = omniflow_content_score("", solution_str, ground_truth_str,
                                     {"chunks": trajectory_chunks})

    total = (cfg.decision * dec + cfg.timeliness * tim
             + cfg.interruption * itp + cfg.content * content)
    return {
        "total": float(total),
        "decision": float(dec),
        "timeliness": float(tim),
        "interruption": float(itp),
        "content": float(content),
    }


# ============================================================================
# verl reward manager 注册（延迟 import——无 verl 环境时本模块仍可单测）
# ============================================================================

def _register() -> None:
    try:
        from verl.experimental.reward_loop.reward_manager.registry import register
    except ImportError:
        logger.debug("verl reward registry not available; manager not registered")
        return

    @register("omniflow_duplex_reward")
    class OmniFlowDuplexRewardManager:  # noqa: D401 — verl registry 协议
        """轨迹级全双工 reward（决策/及时性/打断 + 内容）。

        verl reward_loop 的 manager 协议：compute(data) → float。
        data 里带 trajectory_chunks / ground_truth_chunks（由 rollout
        侧随 sample 传递，见 duplex_rollout.py）。
        """

        def __init__(self, *args: Any, **kwargs: Any):
            self.config = OmniFlowRewardConfig(**(kwargs.get("reward_weights") or {}))

        def compute(self, data: dict[str, Any]) -> float:
            return omniflow_trajectory_reward(
                trajectory_chunks=data["trajectory_chunks"],
                ground_truth_chunks=data["ground_truth_chunks"],
                solution_str=data.get("solution_str", ""),
                ground_truth_str=data.get("ground_truth_str", ""),
                config=self.config,
                global_step=data.get("global_step", 0),
            )["total"]


_register()


if __name__ == "__main__":
    # 自检
    gt = [{"k": 0, "control": "<|listen|>"},
          {"k": 1, "control": "<|speak|>"},
          {"k": 2, "control": "<|speak|>"},
          {"k": 3, "control": "<|listen|>"}]
    traj = [{"k": 0, "control": "<|listen|>"},
            {"k": 1, "control": "<|speak|>", "text_tokens": ["你", "好"], "speech_tokens": ["s1", "s2"]},
            {"k": 2, "control": "<|speak|>", "text_tokens": ["吗"], "speech_tokens": ["s3"], "events": [{"type": "barge_in", "epoch": 2}]},
            {"k": 3, "control": "<|listen|>"}]
    r = omniflow_trajectory_reward(traj, gt, "你好吗", "你好吗")
    print({k: round(v, 3) for k, v in r.items()})