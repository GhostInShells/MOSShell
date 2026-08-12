import queue
from typing import Optional

from ghoshell_moss.depends import depend_host

depend_host()
import miniaudio
import numpy as np
from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.core.speech.base_player import BaseAudioStreamPlayer

__all__ = ["MiniAudioStreamPlayer"]


class MiniAudioStreamPlayer(BaseAudioStreamPlayer):
    """
    基于 miniaudio 的异步音频播放器实现。
    miniaudio 零系统依赖，跨平台一致，纯 wheel 安装即用。

    miniaudio 1.x 使用 generator 模式：PlaybackDevice.start() 接受一个
    callback generator，内部线程通过 gen.send(frame_count) 请求音频帧。

    Player 不依赖 topic 或 transport——它只通过 on_play / on_play_done /
    observe 回调向外提供数据。装线层 (cell/node) 负责订阅回调、构造 topic、
    通过 TopicService 发布。
    """

    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        channels: int = 1,
        logger: LoggerItf | None = None,
        safety_delay: float = 0.1,
    ):
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            logger=logger,
            safety_delay=safety_delay,
        )
        self._playback: Optional[miniaudio.PlaybackDevice] = None

    def _make_generator(self):
        """创建 audio generator，每次 yield 精确 frame_count 的字节。"""
        bytes_per_frame = self.channels * 2  # SIGNED16 = 2 bytes/sample

        def _audio_generator():
            frames_needed = yield b""  # prime
            while not self._stop_event.is_set():
                bytes_needed = (frames_needed or 0) * bytes_per_frame

                while len(self._buf) < bytes_needed:
                    try:
                        self._buf += self._data_queue.get_nowait()
                    except queue.Empty:
                        break

                if len(self._buf) >= bytes_needed and bytes_needed > 0:
                    frames_needed = yield self._buf[:bytes_needed]
                    self._buf = self._buf[bytes_needed:]
                else:
                    missing = max(bytes_needed - len(self._buf), 0)
                    result = self._buf + b"\x00" * missing
                    self._buf = b""
                    frames_needed = yield result

        return _audio_generator()

    def _start_playback(self):
        """启动 miniaudio 播放设备。"""
        gen = self._make_generator()
        next(gen)  # prime
        self._playback = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=self.channels,
            sample_rate=self.sample_rate,
        )
        self._playback.start(gen)

    def _audio_stream_start(self):
        self._data_queue: queue.Queue[bytes] = queue.Queue()
        self._buf = b""
        self._start_playback()

    async def clear(self) -> None:
        """清空播放队列并立即停止音频输出。"""
        # 停止 miniaudio 设备 → 立即中断所有音频输出
        if self._playback is not None:
            self._playback.stop()
            self._playback = None
        # 清空内部缓冲区
        self._data_queue = queue.Queue()
        self._buf = b""
        # 重启设备，准备接收新音频
        self._start_playback()
        # 父类清空 _audio_queue 并重置时间估算
        await super().clear()

    def _audio_stream_write(self, data: np.ndarray):
        self._data_queue.put(data.tobytes())

    def _audio_stream_stop(self):
        if self._playback is not None:
            try:
                self._playback.stop()
            except Exception:
                pass  # miniaudio 内部线程可能已因 generator 返回而自行停止
            self._playback = None
