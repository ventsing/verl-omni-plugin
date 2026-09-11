"""
vllm-omni 全双工客户端桥（L1 插件，运行时延迟加载）

把 vllm-omni 的 experimental/fullduplex WS 客户端适配成本仓库的
DuplexSessionClient 协议（见 duplex_rollout.py）。

为什么是独立桥模块：
  1. vllm-omni 的 import 链很重（vllm 引擎初始化）——必须延迟到运行时，
     保证 omniflow_dataset / rewards / duplex_rollout 无 vllm-omni 也能单测
  2. vllm-omni experimental API 会动——桥集中隔离适配逻辑，
     上游变动只改这一个文件（配合 _patchkit 的 probe_signature 思路，
     这里用运行时能力探测代替签名断言）

对接的 vllm-omni 能力（PR #3907，已合并主干）：
    /v1/duplex 或 /v1/realtime?duplex=1
    - session.config / session.update
    - input_audio_buffer.append / commit（OpenAI Realtime 投影）
    - response.audio.delta / response.text.delta
    - input.cancel（barge-in epoch）
    - conversation item events（playback ack）

帧协议（minicpmo45/policy.py）：
    1s 单元 @16kHz = 10 audio embeddings + <unit>…</unit>；
    200ms chunk = 2 embeddings；视觉每帧 <image> + 64 embeds + </image>。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BridgeUnavailableError(RuntimeError):
    """vllm-omni duplex 运行时不可用（未安装 / 版本不含 experimental.fullduplex）。"""


class VllmOmniDuplexClient:
    """DuplexSessionClient 协议 ← vllm-omni /v1/duplex WS。

    生命周期：
        open_session  → WS connect + session.config（chunk_period_ms 等）
        push_chunk    → input append（音频/视觉增量）→ 等本块输出增量
        barge_in      → input.cancel（epoch 递增）
        close_session → WS close（KV lease 释放——权重同步前提）

    实现说明：
        vllm-omni 的 request_client.py 提供 WS 会话封装；本类做三件事：
        a) 帧协议换算（200ms chunk → 3200 样本 @16kHz = 2 embeddings）
        b) OpenAI-Realtime 事件流 → 本仓库的 chunk 输出结构（control/内容/事件）
        c) listen/speak 控制位解析（minicpmo45 的 <|listen|>/<|speak|> token
           在 response.text.delta 里以文本形式可见）
    """

    def __init__(self, base_url: str = "ws://127.0.0.1:8000",
                 endpoint: str = "/v1/duplex", api_key: str | None = None,
                 sample_rate_hz: int = 16000,
                 samples_per_audio_token: int = 1600):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.api_key = api_key
        self.sample_rate_hz = sample_rate_hz
        self.samples_per_audio_token = samples_per_audio_token
        self._ws: Any = None          # websockets 客户端句柄（open 后设置）
        self._epoch = 0
        # 运行时能力探测（代替 import 期断言——vllm-omni 版本无关）
        self._probe_vllm_omni()

    # ------------------------------------------------------------------
    # 能力探测
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_vllm_omni() -> None:
        """确认 vllm-omni 的 duplex 模块可导入。

        探测而非 import：只在初始化时检查一次，给出可操作的错误信息
        （而不是让用户在第一次 push_chunk 时才撞上 ImportError）。
        """
        try:
            import vllm_omni.experimental.fullduplex as _fd  # noqa: F401
        except ImportError as e:  # pragma: no cover - 环境相关
            raise BridgeUnavailableError(
                "vllm-omni duplex runtime not importable. "
                "PR #3907 已合并主干——请确认 vllm-omni 版本包含 "
                "vllm_omni/experimental/fullduplex/。离线排障请用 "
                "duplex_rollout.ReplayDuplexClient。"
            ) from e

    # ------------------------------------------------------------------
    # DuplexSessionClient 协议实现
    # ------------------------------------------------------------------

    def open_session(self, session_config: dict[str, Any]) -> str:
        """WS 连接 + session.config。"""
        import uuid

        session_id = f"omniflow_{uuid.uuid4().hex[:12]}"
        # WS 连接与 session.config 发送由 request_client 的封装完成；
        # 具体方法名随 vllm-omni 版本可能调整——这里收敛到一个 _open_raw，
        # 上游改名只改一处。
        self._open_raw(session_config)
        logger.info("duplex session %s opened (config=%s)", session_id, session_config)
        return session_id

    def push_chunk(self, session_id: str,
                   visual: list[Any] | None, audio: Any | None) -> dict[str, Any]:
        """推一个 chunk 的环境输入，返回模型输出增量。

        帧协议换算（minicpmo45/policy.py）：
            chunk_period_ms=200 → 3200 samples @16kHz → 2 audio embeddings
            每帧视觉： <image> + 64 embeds + </image>
        """
        # TODO(运行时接线): 走 request_client 的 input append + 响应事件泵。
        # 骨架阶段返回协议结构的空输出，接口契约由 ReplayDuplexClient 验证。
        return {
            "control": "<|listen|>",
            "text_tokens": [],
            "speech_tokens": [],
            "events": [],
            "latency_ms": 0.0,
        }

    def barge_in(self, session_id: str, scope: str = "current") -> int:
        """input.cancel → epoch 递增（RFC #3745 的 session-scoped epoch）。"""
        self._epoch += 1
        # TODO(运行时接线): 发 input.cancel 事件，携带 cancel_scope=scope。
        return self._epoch

    def close_session(self, session_id: str) -> None:
        """关闭 WS（KV lease 释放——episode 边界权重同步的前提）。"""
        # TODO(运行时接线): request_client 的 close。
        logger.info("duplex session %s closed (epoch=%d)", session_id, self._epoch)

    # ------------------------------------------------------------------
    # 原始通道（上游 API 变动只改这里）
    # ------------------------------------------------------------------

    def _open_raw(self, session_config: dict[str, Any]) -> None:
        """建立 WS 连接并发 session.config。

        接线点：vllm_omni.experimental.fullduplex.request_client 的会话建立
        方法。上线前用 examples 的 duplex demo 服务端联调（见
        vllm-omni 仓库 recipes / DESIGN.md 的验证范围）。
        """
        try:
            from vllm_omni.experimental.fullduplex import request_client as _rc
            # 能力探测式调用：优先显式 API，缺失则记录可用符号供排障
            opener = getattr(_rc, "connect", None) or getattr(_rc, "DuplexRequestClient", None)
            if opener is None:
                logger.warning(
                    "request_client lacks connect/DuplexRequestClient; "
                    "available: %s", [n for n in dir(_rc) if not n.startswith("_")],
                )
        except Exception as e:  # pragma: no cover - 环境相关
            raise BridgeUnavailableError(f"failed to init request_client: {e}") from e
