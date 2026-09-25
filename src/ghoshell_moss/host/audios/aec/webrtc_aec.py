"""pywebrtc-audio 实现的回声消除表面 (AcousticEchoCanceller).

far 环形缓冲 + WebRTC AEC3 的 delay estimator 做无感对齐: 上层只按各自节奏喂帧,
不手动对齐. ``push_far`` (player.on_play 回调线程) 与 ``process`` (采集消费线程)
可能不同线程, far 环形缓冲用锁保护.

对齐的本质是**消费节奏**, 不是"取最新": far 入队后按 near 的节奏逐 10ms 消费,
两条时间轴就以同一速率前进, AEC3 的 delay estimator 只需吸收一个常量偏移. 若每次
取"最新的 10ms", 一个 50ms 采集帧被切成 5 个 10ms 块时会拿到同一段参考 —— render
轴碎帧化, 滤波器无从收敛 (live 实测抑制只剩 ~3dB, 见
tests/ghoshell_moss/host/listener/test_webrtc_aec.py 的 live 节拍用例).
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

    far 缓冲一个固定时长的滚动窗口, ``process`` 逐 10ms 帧切 near, 按序取走 far 调
    底层 ``EchoCanceller.process``. 残余延迟由 AEC3 的 delay estimator 自行估计
    (``stream_delay_ms`` 只作 hint).

    far 是**消费式**的: 每个 10ms near 块取走 10ms far (行首), 不足补零. 于是
    render 与 capture 以同一速率前进, 时间轴不失真 —— 这是 delay estimator 能锁住
    常量偏移的前提.
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
        """入队一帧真实播放的参考 (player.on_play 的写入帧). 只追加, 不覆盖历史.

        播放侧按片段突发写入, 消费侧按 near 的节奏逐 10ms 取走 —— 队列占用因此只
        在一个片段量级上下浮动. ``far_capacity_s`` 是防御上限: 真的溢出时从最老的
        端丢弃还未被消费的参考, 那已经是异常工况 (播放侧远超实时喂帧).
        """
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
            far = self._take_far(self._frame)
            blocks.append(np.asarray(self._aec.process(blk, far), dtype=np.float32))
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocks)

    def _take_far(self, frame: int) -> np.ndarray:
        """按序取走 frame 个 far 样本 (消费式), 不足补零.

        只推进读指针, 不回退也不重复 —— near 的每一块对应 far 的下一段, 两条时间轴
        因此严格同速. 环空 (还没开播 / 播完) 时补零: 零参考即"当下没有回声", 正是
        AEC3 该看到的东西.
        """
        parts: list[np.ndarray] = []
        need = frame
        with self._lock:
            while need > 0 and self._far_ring:
                head = self._far_ring[0]
                if head.size <= need:
                    parts.append(head)
                    need -= head.size
                    self._far_total -= head.size
                    self._far_ring.popleft()
                else:
                    parts.append(head[:need])
                    self._far_ring[0] = head[need:]
                    self._far_total -= need
                    need = 0
        if need > 0:
            parts.append(np.zeros(need, dtype=np.float32))
        if len(parts) == 1:
            return np.asarray(parts[0], dtype=np.float32)
        return np.concatenate(parts)
