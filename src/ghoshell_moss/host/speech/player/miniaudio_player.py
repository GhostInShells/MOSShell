import queue
import threading
from typing import Optional

import miniaudio
import numpy as np
from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.core.speech.base_player import BaseAudioStreamPlayer
from ghoshell_moss.host.speech.capture.audio_transport import AudioTransport
from ghoshell_moss.topics.audio import AudioRuntimeTopic

__all__ = ["MiniAudioStreamPlayer"]


class MiniAudioStreamPlayer(BaseAudioStreamPlayer):
    """
    基于 miniaudio 的异步音频播放器实现。
    miniaudio 零系统依赖，跨平台一致，纯 wheel 安装即用。

    miniaudio 1.x 使用 generator 模式：PlaybackDevice.start() 接受一个
    callback generator，内部线程通过 gen.send(frame_count) 请求音频帧。

    If *transport* is provided, the player publishes AudioRuntimeTopic
    (device_name="speaker") on play state changes — symmetric to how
    MiniAudioCaptureSource reports capture state.  The listener app
    consumes this to suppress ASR while the ghost is speaking.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        channels: int = 1,
        logger: LoggerItf | None = None,
        safety_delay: float = 0.1,
        transport: AudioTransport | None = None,
    ):
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            logger=logger,
            safety_delay=safety_delay,
        )
        self._playback: Optional[miniaudio.PlaybackDevice] = None
        if transport is not None:
            self.logger.info("%s wiring speaker AudioRuntimeTopic via transport", self._log_prefix)
            self._wire_speaker_topic(transport)
        else:
            self.logger.warning("%s no transport provided, TTS gate disabled", self._log_prefix)

    def _wire_speaker_topic(self, transport: AudioTransport) -> None:
        """Publish AudioRuntimeTopic(device_name="speaker") on play/done.

        Uses a debounce timer for running=False because on_play_done fires
        per-chunk when _audio_queue empties, but miniaudio buffers are still
        playing.  Without debouncing, running=True→False flickers within 1ms
        and the listener gate never sees a stable True.
        """
        _state = {"was_playing": False, "done_timer": None}

        def _publish_done() -> None:
            if _state["was_playing"]:
                _state["was_playing"] = False
                self.logger.info("%s TTS gate: publishing speaker running=False", self._log_prefix)
                transport.pub_topic(AudioRuntimeTopic(
                    running=False,
                    device_name="speaker",
                    device_explain="TTS speech output",
                ))

        def on_play(_data: np.ndarray) -> None:
            if _state["done_timer"] is not None:
                _state["done_timer"].cancel()
                _state["done_timer"] = None
            if not _state["was_playing"]:
                _state["was_playing"] = True
                self.logger.info("%s TTS gate: publishing speaker running=True", self._log_prefix)
                transport.pub_topic(AudioRuntimeTopic(
                    running=True,
                    device_name="speaker",
                    device_explain="TTS speech output",
                ))

        def on_play_done() -> None:
            if _state["done_timer"] is not None:
                _state["done_timer"].cancel()
            delay = self._time_to_wait() + 0.1
            _state["done_timer"] = threading.Timer(delay, _publish_done)
            _state["done_timer"].daemon = True
            _state["done_timer"].start()

        self.on_play(on_play)
        self.on_play_done(on_play_done)

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
            self._playback.stop()
            self._playback = None
