"""
Omni-Flow 时间对齐数据集（L1 插件，槽位④ data.custom_cls）

把论文（arXiv 2604.27393）§3 的 Omni-Flow 序列化落成 verl-omni 的数据管道：

    g_k = [v_k ; a_k ; o_k]     第 k 个时间块的 token 组
    序列 = g_1 g_2 g_3 ...      标准 causal LM next-token prediction

设计要点（对应论文结论）：
  - chunk size 默认 0.2s（论文 §3.3 ablation 最优：1.0s 太钝、0.1s 不稳）
  - LS 控制形式（论文 §3.3：LS > LT）：o_k 以 <|listen|>/<|speak|> 控制位开头，
    "是否说"与"说什么"解耦
  - TAIL 有界 look-ahead（论文 §3.4）：块内最后几个文本 token 的 speech token
    推迟到下一块，给发音留局部上下文

与 vllm-omni 的一致性约束：
    帧协议常量（SAMPLES_PER_AUDIO_TOKEN 等）必须与 vllm-omni 的
    experimental/fullduplex/minicpmo45/policy.py 完全一致，否则训练/推理
    token 数错位，KV 里出现 pad embedding 会破坏 listen/speak 行为。
    因此运行时优先 import vllm-omni 的常量，本地值仅作 fallback。

接入方式（config.yaml）：

    data:
      train_files: /path/to/omniflow_episodes.parquet
      custom_cls:
        path: pkg://verl_omni_ext.features.fullduplex.omniflow_dataset
        collate_fn: omniflow_collate_fn
"""

from __future__ import annotations

import base64
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# 帧协议常量：优先 import vllm-omni 的权威定义（防漂移）
# ============================================================================

try:  # vllm-omni 主干（PR #3907 已合并）
    from vllm_omni.experimental.fullduplex.minicpmo45.policy import (
        MiniCPMO45DuplexPolicy as _VllmOmniPolicy,
    )

    SAMPLE_RATE_HZ: int = _VllmOmniPolicy.SAMPLE_RATE_HZ
    SAMPLES_PER_AUDIO_TOKEN: int = _VllmOmniPolicy.SAMPLES_PER_AUDIO_TOKEN
    VISION_EMBEDS_PER_FRAME: int = _VllmOmniPolicy.VISION_EMBEDS_PER_FRAME
except ImportError:  # 本地 fallback（数值与 policy.py 对齐，改一处必须同步另一处）
    SAMPLE_RATE_HZ = 16000
    SAMPLES_PER_AUDIO_TOKEN = 1600  # 100ms 一个 audio embedding
    VISION_EMBEDS_PER_FRAME = 64

# 论文 §3.3：0.2s chunk 最优
DEFAULT_CHUNK_SIZE_S = 0.2
# 论文 §3.4 TAIL：有界 look-ahead 的默认 token 数
DEFAULT_TAIL_LOOKAHEAD = 3

# LS 控制位（论文 §3.3 胜者形式；token 串由 tokenizer 解析为 id）
CONTROL_LISTEN = "<|listen|>"
CONTROL_SPEAK = "<|speak|>"

# vllm-omni policy.SPECIAL_TOKEN_FIELDS 的子集（序列化只用到这些）
UNIT_TOKEN = "<unit>"
UNIT_END_TOKEN = "</unit>"
IMAGE_TOKEN = "<image>"
IMAGE_END_TOKEN = "</image>"


@dataclass
class ChunkSpec:
    """一个时间块的序列化产物（g_k = [v_k; a_k; o_k] 的逻辑视图）。

    训练时由 thinker_adapter 的 configure_model/configure_processor 把
    逻辑视图转成模型真实的 token id 序列——本类不持有 tokenizer。
    """

    k: int                                   # 块序号（0-based）
    t_start: float                           # 块起始时间（秒）
    t_end: float                             # 块结束时间（秒）

    # v_k：视觉输入（帧列表，帧对象由 processor 消费）
    visual_frames: list[Any] = field(default_factory=list)

    # a_k：音频输入（样本数组的引用/base64，由 processor 消费）
    audio_chunks: list[Any] = field(default_factory=list)

    # o_k：输出监督
    control: str = CONTROL_LISTEN            # <|listen|> / <|speak|>（LS 控制位）
    text_tokens: list[str] = field(default_factory=list)   # 本块文本 token
    speech_tokens: list[Any] = field(default_factory=list)  # TAIL 归属本块的 speech token

    # 事件（rollout 期填充；SFT 数据为空）
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def should_speak(self) -> bool:
        return self.control == CONTROL_SPEAK


# ============================================================================
# 时间归块：论文 §3.4 TAIL 监督构造规则的实现
# ============================================================================

def _chunk_index(t: float, chunk_size: float) -> int:
    """时间 t → 块序号。token 按 start time 归块（论文原文）。

    epsilon 抵消浮点边界误差：t 恰在块边界（如 0.6/0.2）时
    浮点除法可能得 2.9999…，floor 会归错块。
    """
    return int(math.floor(t / chunk_size + 1e-9))


def serialize_episode(
    episode: dict[str, Any],
    chunk_size_s: float = DEFAULT_CHUNK_SIZE_S,
    tail_lookahead: int = DEFAULT_TAIL_LOOKAHEAD,
) -> list[ChunkSpec]:
    """把一条时间索引的 episode 样本序列化成 ChunkSpec 序列。

    Args:
        episode: parquet 行，结构见 docs/feature_fullduplex_omniflow.md §五
            {
              "streams": {
                "visual": [{t_start, t_end, frames}],
                "audio":  [{t_start, t_end, samples}],
                "text":   [{t_start, t_end, tokens}],
                "speech": [{t_start, t_end, tokens}],   # speech token 的时间戳
              }
            }
        chunk_size_s: 时间块长度（论文最优 0.2）
        tail_lookahead: TAIL 有界 look-ahead 的文本 token 数

    序列化规则：
        v_k/a_k：t_start 落在块内的输入 → 归块 k
        o_k 控制位：块内有文本输出 → <|speak|>，否则 <|listen|>
        text：按 start time 归块
        speech：TAIL 规则——若所属文本 token 是其块内最后 tail_lookahead 个之一，
                speech token 推迟到下一块（有界 look-ahead）
    """
    streams = episode.get("streams", {}) or {}
    visual = streams.get("visual", []) or []
    audio = streams.get("audio", []) or []
    text = streams.get("text", []) or []
    speech = streams.get("speech", []) or []

    # 1. 数块数：覆盖所有流的最后结束时间。
    #    语义：时长 T 的 episode → ceil(T/chunk) 块（流结束于边界 1.0s
    #    说明最后一块是 [0.8,1.0)，不需要第 6 块——t_end 是闭区间右端点，
    #    不产生新块；token 的 t_start 才会（floor 归块 + 越界丢弃）。
    t_max = 0.0
    for entries in (visual, audio, text, speech):
        for e in entries:
            t_max = max(t_max, float(e.get("t_end", 0.0)), float(e.get("t_start", 0.0)))
    num_chunks = max(1, int(math.ceil(t_max / chunk_size_s - 1e-9)))

    chunks = [
        ChunkSpec(
            k=k,
            t_start=k * chunk_size_s,
            t_end=(k + 1) * chunk_size_s,
        )
        for k in range(num_chunks)
    ]

    # 2. v_k / a_k：输入按 start time 归块
    for e in visual:
        k = _chunk_index(float(e.get("t_start", 0.0)), chunk_size_s)
        if 0 <= k < num_chunks:
            chunks[k].visual_frames.extend(e.get("frames", []))
    for e in audio:
        k = _chunk_index(float(e.get("t_start", 0.0)), chunk_size_s)
        if 0 <= k < num_chunks:
            chunks[k].audio_chunks.append(e.get("samples", e))

    # 3. text token 按 start time 归块（论文：tokens whose start times fall into chunk k）
    for e in text:
        tokens = e.get("tokens", [])
        # 逐 token 有时间戳：[{token, t_start}]；或整段一个时间戳
        per_token = e.get("token_times")
        if per_token:
            for tok, t in zip(tokens, per_token):
                k = _chunk_index(float(t), chunk_size_s)
                if 0 <= k < num_chunks:
                    chunks[k].text_tokens.append(tok)
        else:
            k = _chunk_index(float(e.get("t_start", 0.0)), chunk_size_s)
            if 0 <= k < num_chunks:
                chunks[k].text_tokens.extend(tokens)

    # 4. speech token：TAIL 有界 look-ahead
    #    文本 token i 的 speech token 归属 = text 所在块 + (1 if i 在块内最后
    #    tail_lookahead 个之内 else 0)
    for e in speech:
        tok = e.get("tokens", [])
        t_text_start = float(e.get("t_start", 0.0))  # speech 对应文本的起始时间
        k_text = _chunk_index(t_text_start, chunk_size_s)
        if not (0 <= k_text < num_chunks):
            continue
        block_tokens = chunks[k_text].text_tokens
        # 该 speech 段对应的文本是否落在块尾 lookahead 窗口：用段起始时间在块内的
        # 相对位置近似（段起止与块边界的关系）
        rel = (t_text_start - chunks[k_text].t_start) / max(chunks[k_text].t_end - chunks[k_text].t_start, 1e-9)
        k_speech = k_text + (1 if rel >= 1.0 - tail_lookahead / max(len(block_tokens), 1) and k_text + 1 < num_chunks else 0)
        chunks[k_speech].speech_tokens.extend(tok)

    # 5. LS 控制位：有文本输出的块 → <|speak|>（语音 token 不触发控制位，
    #    因为 speech 是 text 的播放形态——论文的 out-stream 以文本为主干）
    for c in chunks:
        c.control = CONTROL_SPEAK if c.text_tokens else CONTROL_LISTEN

    return chunks


# ============================================================================
# loss mask：v_k/a_k 不训，o_k 训练位 = 控制位 + 内容位
# ============================================================================

def build_loss_mask(chunks: list[ChunkSpec]) -> list[bool]:
    """输出侧每个逻辑位的训练掩码。

    对应序列化后的展开（控制位, text…, speech…）：
        输入侧（v_k/a_k 展开的 token）不在本函数输出中——它们由
        thinker_adapter 在构造模型输入时统一 mask。
    """
    mask: list[bool] = []
    for c in chunks:
        mask.append(True)                    # LS 控制位：训练（决策是策略核心）
        mask.extend([True] * len(c.text_tokens))    # 文本内容：训练
        mask.extend([True] * len(c.speech_tokens))  # speech token：训练（SFT 期）
        # v_k/a_k 输入 token 不出现于此
    return mask


# ============================================================================
# 槽位④ 接入：verl-omni data.custom_cls 的 collate_fn
# ============================================================================

def omniflow_collate_fn(batch: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
    """verl-omni RLHDataset 的 custom collate（pkg:// 加载）。

    输入：parquet 行列表（每行一个 episode，结构见 serialize_episode）
    输出：batch dict，含序列化后的 chunk 序列（逻辑视图）+ GRPO group 信息。
    模型侧的 token 化（<|listen|> → id 等）由 thinker_adapter 完成——
    这里保持 tokenizer 无关，dataset 在 dataloader worker 里跑。
    """
    chunk_size = kwargs.get("chunk_size_s", DEFAULT_CHUNK_SIZE_S)
    lookahead = kwargs.get("tail_lookahead", DEFAULT_TAIL_LOOKAHEAD)

    out_episodes = []
    for row in batch:
        chunks = serialize_episode(row, chunk_size_s=chunk_size, tail_lookahead=lookahead)
        out_episodes.append(
            {
                "episode_id": row.get("episode_id", ""),
                "group_id": row.get("group_id", row.get("episode_id", "")),  # GRPO group 默认同 episode
                "chunks": chunks,
                "loss_mask": build_loss_mask(chunks),
                "meta": row.get("meta", {}),
            }
        )

    return {
        "episodes": out_episodes,
        "batch_size": len(out_episodes),
        "chunk_size_s": chunk_size,
        # 提示 rollout/训练侧：这是 Omni-Flow 全双工样本，不是普通 prompt→response
        "omniflow": True,
    }


# ============================================================================
# 工具：episode parquet 行的构造器（数据预处理用）
# ============================================================================

def make_episode_row(
    episode_id: str,
    visual: list[dict] | None = None,
    audio: list[dict] | None = None,
    text: list[dict] | None = None,
    speech: list[dict] | None = None,
    meta: dict | None = None,
) -> dict[str, Any]:
    """构造 serialize_episode 可消费的 episode 行（数据 pipeline 用）。

    audio samples 支持 bytes（自动 base64）或直接数组引用。
    """
    def _norm_audio(a: list[dict]) -> list[dict]:
        out = []
        for e in a:
            samples = e.get("samples")
            if isinstance(samples, (bytes, bytearray)):
                e = {**e, "samples_b64": base64.b64encode(bytes(samples)).decode()}
                e.pop("samples", None)
            out.append(e)
        return out

    return {
        "episode_id": episode_id,
        "streams": {
            "visual": visual or [],
            "audio": _norm_audio(audio or []),
            "text": text or [],
            "speech": speech or [],
        },
        "meta": meta or {},
    }


def episode_summary(chunks: list[ChunkSpec]) -> str:
    """人类可读的序列化结果摘要（排障/探针用）。"""
    n_speak = sum(1 for c in chunks if c.should_speak)
    n_text = sum(len(c.text_tokens) for c in chunks)
    n_speech = sum(len(c.speech_tokens) for c in chunks)
    n_vis = sum(len(c.visual_frames) for c in chunks)
    n_aud = sum(len(c.audio_chunks) for c in chunks)
    return (
        f"Omni-Flow episode: {len(chunks)} chunks "
        f"(speak={n_speak}, listen={len(chunks)-n_speak}) | "
        f"text={n_text} speech={n_speech} visual={n_vis} audio={n_aud}"
    )


if __name__ == "__main__":
    # 自检：一条 1.0s 的样例 episode，0.2s chunk → 5 块
    demo = make_episode_row(
        episode_id="demo_ep",
        visual=[{"t_start": 0.0, "t_end": 0.2, "frames": ["f0"]}],
        audio=[{"t_start": 0.0, "t_end": 1.0, "samples": [0] * 16000}],
        text=[{"t_start": 0.4, "t_end": 0.6, "tokens": ["你", "好"]}],
        speech=[{"t_start": 0.4, "t_end": 0.8, "tokens": ["st1", "st2"]}],
    )
    chunks = serialize_episode(demo)
    print(episode_summary(chunks))
    for c in chunks:
        print(
            f"  chunk {c.k} [{c.t_start:.1f},{c.t_end:.1f}): "
            f"control={c.control} text={c.text_tokens} speech={c.speech_tokens} "
            f"vis={len(c.visual_frames)} aud={len(c.audio_chunks)}"
        )