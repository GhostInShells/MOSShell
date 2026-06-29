"""
G1 PlayStream 音频播放器 — 基于 BaseAudioStreamPlayer，用 G1 内置喇叭发声。

PCM 格式: 16kHz mono s16le (与 G1 PlayStream 契约一致)。
流式语义: 同 stream_id 分块推送 → G1 无缝拼接。新 stream_id → 抢占旧流。
打断语义: PlayStop 即时中断，clear() 中调用。

前置: 使用前必须 bootstrap. G1StreamPlayer 不再隐式 bootstrap.
"""

from __future__ import annotations

import time
import queue

import numpy as np
from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.core.speech.base_player import BaseAudioStreamPlayer

__all__ = ["G1StreamPlayer"]


class G1StreamPlayer(BaseAudioStreamPlayer):
    """G1 PlayStream 音频播放器。

    构造函数不做 DDS 连接 — 延迟到 _audio_stream_start (worker 线程)。
    需要先调用 ghoshell_moss_contrib.unitree.g1.bootstrap(nic) 初始化 DDS。
    """

    def __init__(
            self,
            *,
            sample_rate: int = 16000,
            channels: int = 1,
            logger: LoggerItf | None = None,
            safety_delay: float = 0.15,
    ):
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            logger=logger,
            safety_delay=safety_delay,
        )
        self._app_name = "moss_tts"
        self._stream_id = ""
        self._buf = b""
        self._stream_count = 0
        self._data_queue = queue.Queue()
        self._audio = None  # 由 _audio_stream_start() 赋值

    def _next_stream_id(self) -> str:
        self._stream_count += 1
        return f"moss_{int(time.time() * 1000)}_{self._stream_count}"

    # -- 抽象方法实现 ----------------------------------------------------------

    def _audio_stream_start(self):
        """worker 线程: 初始化 DDS + AudioClient + 生成新 stream_id。"""
        from ghoshell_moss_contrib.unitree.g1._bootstrap import get_audio_client
        self._audio = get_audio_client()  # raise if not bootstrapped
        self._stream_id = self._next_stream_id()

    def _audio_stream_write(self, data: np.ndarray):
        """worker 线程: 推送 PCM chunk 到 PlayStream。"""
        if self._audio is None:
            return
        pcm = data.tobytes()
        code, _ = self._audio.PlayStream(self._app_name, self._stream_id, pcm)
        if code != 0:
            self.logger.warning(
                "%s PlayStream failed: code=%d stream_id=%s len=%d",
                self._log_prefix, code, self._stream_id, len(pcm),
            )

    def _audio_stream_stop(self):
        """worker 线程: 停止当前 stream，不创建新 stream_id。"""
        if self._audio is not None:
            self._audio.PlayStop(self._app_name)

    # -- 生命周期覆写 ----------------------------------------------------------

    async def clear(self) -> None:
        """清空播放队列 + PlayStop 即时打断 + 准备新 stream。"""
        if self._audio is not None:
            self._audio.PlayStop(self._app_name)
        self._stream_id = self._next_stream_id()
        self._data_queue = queue.Queue()
        self._buf = b""
        self._estimated_end_time = time.time()
        self._play_done_event.set()
        self.logger.info("%s cleared, new stream_id=%s", self._log_prefix, self._stream_id)
