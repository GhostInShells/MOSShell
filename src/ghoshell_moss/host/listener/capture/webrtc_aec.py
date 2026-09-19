"""pywebrtc-audio 实现的回声消除表面 (AcousticEchoCanceller).

far 环形缓冲 + WebRTC AEC3 的 delay estimator 做无感对齐: 上层只按各自节奏喂帧,
不手动对齐. ``push_far`` (player.on_play 回调线程) 与 ``process`` (采集消费线程)
可能不同线程, far 环形缓冲用锁保护.
"""
import collections
import threading

import numpy as np

from ghoshell_moss.contracts.audio import AcousticEchoCanceller

__all__ = ["PyWebrtcEchoCanceller"]

#: AEC3 帧长 (10ms).
_FRAME_MS = 10


class PyWebrtcEchoCanceller(AcousticEchoCanceller):
    """WebRTC AEC3 实现.

    far 缓冲一个固定时长的滚动窗口, ``process`` 逐 10ms 帧切 near, 取最近 far 调
    底层 ``EchoCanceller.process``. 残余延迟由 AEC3 的 delay estimator 自行估计
    (``stream_delay_ms`` 只作 hint).
    """

    def __init__(
            self,
            *,
            sample_rate: int = 16000,
            num_channels: int = 1,
            stream_delay_ms: int = 0,
            far_capacity_s: float = 2.0,
    ):
        from pywebrtc_audio import EchoCanceller

        self.sample_rate = sample_rate
        self.stream_delay_ms = stream_delay_ms

        self._aec = EchoCanceller(
            sample_rate=sample_rate,
            num_channels=num_channels,
            stream_delay_ms=stream_delay_ms,
        )
        self._frame = int(sample_rate * _FRAME_MS / 1000)
        self._far_capacity = int(sample_rate * far_capacity_s)
        self._far_ring: collections.deque[np.ndarray] = collections.deque()
        self._far_total = 0
        self._lock = threading.Lock()
        self._pending = np.zeros(0, dtype=np.float32)

    def push_far(self, frame: np.ndarray) -> None:
        arr = np.asarray(frame, dtype=np.float32).ravel()
        if arr.size == 0:
            return
        with self._lock:
            self._far_ring.append(arr)
            self._far_total += arr.size
            while self._far_ring and self._far_total > self._far_capacity:
                self._far_total -= self._far_ring[0].size
                self._far_ring.popleft()

    def process(self, near: np.ndarray) -> np.ndarray:
        arr = np.asarray(near, dtype=np.float32).ravel()
        if arr.size == 0:
            return np.zeros(0, dtype=np.float32)
        self._pending = np.concatenate([self._pending, arr])
        blocks: list[np.ndarray] = []
        while self._pending.size >= self._frame:
            blk = self._pending[: self._frame]
            self._pending = self._pending[self._frame:]
            far = self._far_window(self._frame)
            blocks.append(np.asarray(self._aec.process(blk, far), dtype=np.float32))
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocks)

    def _far_window(self, frame: int) -> np.ndarray:
        """取最近写入的 far 样本 (单帧长), 不足补零."""
        with self._lock:
            if not self._far_ring:
                return np.zeros(frame, dtype=np.float32)
            parts: list[np.ndarray] = []
            need = frame
            for arr in reversed(self._far_ring):
                take = arr[-need:] if arr.size > need else arr
                parts.append(take)
                need -= take.size
                if need <= 0:
                    break
            if need > 0:
                parts.append(np.zeros(need, dtype=np.float32))
        parts.reverse()
        return np.concatenate(parts)
