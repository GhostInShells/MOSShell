import asyncio
import queue
import time
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

    设备排空是 miniaudio 自己的事: 基类 ``wait_play_done`` 只等 worker 队列空, 而
    CoreAudio 还有一层输出缓冲, 所以本实现覆写它, 等播放头真正走完最后一帧真实音频.
    """

    # 设备播放头停止推进多久后放弃排空等待 (设备停摆/异常的兜底).
    _DRAIN_STALL_TIMEOUT = 0.5

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        logger: LoggerItf | None = None,
        safety_delay: float = 0.1,
        device_pattern: str = "",
    ):
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            logger=logger,
            safety_delay=safety_delay,
        )
        self._device_pattern = device_pattern
        self._playback: Optional[miniaudio.PlaybackDevice] = None
        # 设备播放头追踪 (见 _make_generator / wait_play_done).
        self._frames_delivered = 0
        self._playhead_frames = 0
        self._playhead_time = 0.0
        self._last_real_frame_pos = 0
        self._reset_drain_state()

    def _reset_drain_state(self) -> None:
        """设备重启/清空后重置播放头追踪.

        _frames_delivered: 已交付给设备的帧数 (含补的静音).
        _playhead_frames/_playhead_time: 最近一次设备回调开始时刻设备已播完的帧数.
        _last_real_frame_pos: 交付流里最后一帧真实音频的结束位置.
        """
        self._frames_delivered = 0
        self._playhead_frames = 0
        self._playhead_time = time.monotonic()
        self._last_real_frame_pos = 0

    def _find_device(self):
        """按 ``device_pattern`` (名字子串) 匹配输出设备; 空则用 miniaudio 默认."""
        pattern = self._device_pattern.strip().lower()
        if not pattern:
            return None
        try:
            for d in miniaudio.Devices().get_playbacks():
                if pattern in d['name'].lower():
                    return d['id']
        except Exception as e:
            self.logger.warning("Playback device enumeration failed: %s, using default", e)
        return None

    def _make_generator(self):
        """创建 audio generator，每次 yield 精确 frame_count 的字节。"""
        bytes_per_frame = self.channels * 2

        def _audio_generator():
            frames_needed = yield b""  # prime
            while not self._stop_event.is_set():
                bytes_needed = (frames_needed or 0) * bytes_per_frame

                # 回调被触发的这一刻, 上一块刚播完 — 设备播放头 = 已交付帧数.
                # 这是设备自己给出的进度, 比 worker 的模拟 sleep 更可信.
                self._playhead_frames = self._frames_delivered
                self._playhead_time = time.monotonic()

                if bytes_needed <= 0:
                    # 设备没要帧 (异常): 不丢数据、不推进播放头.
                    frames_needed = yield b""
                    continue

                while len(self._buf) < bytes_needed:
                    try:
                        self._buf += self._data_queue.get_nowait()
                    except queue.Empty:
                        break

                if len(self._buf) >= bytes_needed:
                    real_bytes = bytes_needed
                    result = self._buf[:bytes_needed]
                    self._buf = self._buf[bytes_needed:]
                else:
                    # _buf 里只有真实音频 (补的静音不进 _buf), 不足部分补零.
                    real_bytes = len(self._buf)
                    result = self._buf + b"\x00" * (bytes_needed - real_bytes)
                    self._buf = b""

                out_frames = len(result) // bytes_per_frame
                real_frames = real_bytes // bytes_per_frame
                start_pos = self._frames_delivered
                self._frames_delivered = start_pos + out_frames
                if real_frames > 0:
                    self._last_real_frame_pos = start_pos + real_frames
                frames_needed = yield result

        return _audio_generator()

    def _device_drained(self) -> bool:
        """设备是否已把交付给它的真实音频全部播完.

        ``_data_queue``/``_buf`` 里可能还有真实音频没交给设备; 都交付完之后, 只要播放头
        推过 ``_last_real_frame_pos`` 就算排空.
        """
        if self._playback is None or self.sample_rate <= 0:
            return True
        if not self._data_queue.empty() or self._buf:
            return False
        return self._playhead_frames >= self._last_real_frame_pos

    async def _drain_device_output(self, deadline: Optional[float]) -> bool:
        """等播放头推过最后一帧真实音频; 设备停摆超过 ``_DRAIN_STALL_TIMEOUT`` 就放弃.

        判据只用设备回调给出的播放头 (不乐观外推提前返回); 但播放头只在回调时更新,
        所以用回调时刻外推下一次该复核的时间点, 避免空转轮询.
        """
        progress = self._frames_delivered + self._last_real_frame_pos
        progress_at = time.monotonic()
        while not self._stop_event.is_set():
            if self._device_drained():
                return True
            now = time.monotonic()
            current = self._frames_delivered + self._last_real_frame_pos
            if current != progress:
                progress, progress_at = current, now
            elif now - progress_at > self._DRAIN_STALL_TIMEOUT:
                self.logger.warning(
                    "%s device playhead stalled at %d/%d frames, stop draining",
                    self._log_prefix, self._playhead_frames, self._last_real_frame_pos,
                )
                return True
            if deadline is not None and now >= deadline:
                return False
            if self._data_queue.empty() and not self._buf:
                # 真实音频都交付了: 睡到播放头名义上走到目标帧附近再复核.
                eta = self._playhead_time + (
                    self._last_real_frame_pos - self._playhead_frames
                ) / self.sample_rate
                wait = min(max(eta - now, 0.005), 0.05)
            else:
                # 还有真实音频没交给设备, 等下一次回调来取.
                wait = 0.005
            await asyncio.sleep(wait)
        return True

    async def wait_play_done(self, timeout: Optional[float] = None) -> bool:
        """等音频真正放完 — 基类只等 worker 队列空, 这里补上设备输出缓冲.

        基类在 worker 输入队列一空就置完成事件, 但 ``_audio_stream_write`` 只是把数据
        交给设备, 设备回调要一个周期后才真的播出. 所以在契约方法里补设备排空: 返回即
        代表最后一帧真实音频已经播出, 调用方随后的 clear()/stop() 不会切掉尾音.
        """
        started = time.monotonic()
        if not await super().wait_play_done(timeout):
            return False
        deadline = started + timeout if timeout is not None and timeout > 0 else None
        return await self._drain_device_output(deadline)

    def is_playing(self) -> bool:
        """设备缓冲里还有声音就算在播 — 基类只看内部事件会提前报停."""
        return super().is_playing() or not self._device_drained()

    def _start_playback(self):
        """启动 miniaudio 播放设备。"""
        gen = self._make_generator()
        next(gen)  # prime
        self._playback = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=self.channels,
            sample_rate=self.sample_rate,
            device_id=self._find_device(),
        )
        self._playback.start(gen)

    def _audio_stream_start(self):
        self._data_queue: queue.Queue[bytes] = queue.Queue()
        self._buf = b""
        self._reset_drain_state()
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
        self._reset_drain_state()
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
