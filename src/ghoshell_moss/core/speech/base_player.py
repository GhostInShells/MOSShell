import asyncio
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional

import numpy as np
from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.contracts.speech import AudioFormat, PlaybackSample, StreamAudioPlayer
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_common.helpers import Timeleft

__all__ = ["BaseAudioStreamPlayer"]


# author: deepseek v3.1


class BaseAudioStreamPlayer(StreamAudioPlayer, ABC):
    """
    基础的 AudioStream
    使用单独的线程处理音频输出，通过 asyncio 队列进行通信
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        logger: LoggerItf | None = None,
        safety_delay: float = 0.2,
    ):
        """
        使用单独的线程处理阻塞的音频输出操作。
        """
        self.logger = logger or logging.getLogger("moss")
        self._log_prefix = "[StreamAudioPlayer][%s] " % self.__class__.__name__
        self.audio_type = AudioFormat.PCM_S16LE
        # self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self._safety_delay = safety_delay
        self._play_done_event = ThreadSafeEvent()
        self._play_done_event.set()
        self._committed = True
        self._estimated_end_time = 0.0
        self._closed = False

        # 使用线程安全的队列进行线程间通信. 每项是 (resampled_pcm, stream_id, fragment_id).
        self._audio_queue: queue.Queue[tuple[np.ndarray, str, str] | None] = queue.Queue()
        self._thread = None
        self._stop_event = threading.Event()
        self._on_play_callbacks = []
        self._on_play_done_callbacks = []

        # 实际播放可感知观察者 — 全局注册, 非永久挂载.
        # stream 生命周期由治理层管理; 无观察者时不计算 PlaybackSample.
        self._playback_observers: list[Callable[[PlaybackSample], None]] = []

    def on_play(self, callback: Callable[[np.ndarray], None]) -> None:
        self._on_play_callbacks.append(callback)

    def on_play_done(self, callback: Callable[[], None]) -> None:
        self._on_play_done_callbacks.append(callback)

    def observe(
            self,
            callback: Callable[[PlaybackSample], None],
    ) -> Callable[[], None]:
        """注册一个实际播放可感知观察者 (全局, 非 stream 作用域).

        stream 生命周期由治理层管理; 无观察者时不计算 PlaybackSample.
        """
        if callback not in self._playback_observers:
            self._playback_observers.append(callback)

        def _unsubscribe() -> None:
            if callback in self._playback_observers:
                self._playback_observers.remove(callback)

        return _unsubscribe

    async def start(self) -> None:
        """启动音频播放器"""
        if self._thread and self._thread.is_alive():
            return

        # 启动音频工作线程
        # todo: 改成 asyncio.to_thread task
        self._thread = threading.Thread(target=self._audio_worker, daemon=True)
        self._thread.start()
        self.logger.info("%s player is started", self._log_prefix)

    async def close(self) -> None:
        """关闭音频播放器"""
        self._closed = True
        self._stop_event.set()

        # 等待工作线程结束
        if self._thread and self._thread.is_alive():
            # 放入停止信号
            self._audio_queue.put_nowait(None)
            self._thread.join(timeout=2.0)

        self.logger.info("%s player is closed", self._log_prefix)

    async def clear(self) -> None:
        """清空播放队列并重置"""
        # 清空队列
        old_queue = self._audio_queue
        self._audio_queue = queue.Queue()
        while not old_queue.empty():
            try:
                _ = old_queue.get_nowait()
            except queue.Empty:
                break
        old_queue.put_nowait(None)
        # 重置时间估计
        self._estimated_end_time = time.time()
        self._play_done_event.set()
        self.logger.info(
            "%s player is cleared, estimated_end_time is %.2f",
            self._log_prefix,
            self._estimated_end_time,
        )

    @classmethod
    def resample(
        cls,
        audio_data: np.ndarray,
        *,
        origin_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        """使用线性插值进行采样率转换。需要更好的重采样算法时覆写此方法。"""
        if origin_rate == target_rate:
            return audio_data
        if not isinstance(audio_data, np.ndarray):
            raise TypeError("audio_data must be numpy ndarray")
        if origin_rate <= 0 or target_rate <= 0:
            raise ValueError("sample rate must greater than 0")

        target_len = int(len(audio_data) * target_rate / origin_rate)
        x_orig = np.arange(len(audio_data))
        x_target = np.linspace(0, len(audio_data) - 1, target_len)
        return np.interp(x_target, x_orig, audio_data).astype(np.int16)

    def add(
        self,
        chunk: np.ndarray,
        *,
        audio_type: AudioFormat,
        rate: int,
        channels: int = 1,
        stream_id: str = "",
        fragment_id: str = "",
    ) -> float:
        """添加音频片段到播放队列, 返回一个期望的终结时间."""
        if self._closed:
            self.logger.warning("%s player receive audio but is closed", self._log_prefix)
            return time.time()

        # 格式转换
        if audio_type == AudioFormat.PCM_F32LE:
            # float32 [-1, 1] -> int16
            audio_data = (chunk * 32767).astype(np.int16)
        else:
            # 假设已经是 int16
            audio_data = chunk.astype(np.int16)

        # 格式校验
        if rate <= 0:
            raise ValueError("rate must be greater than 0")

        # 计算持续时间
        duration = len(audio_data) / rate
        resampled_audio_data = self.resample(audio_data, origin_rate=rate, target_rate=self.sample_rate)

        # 添加到线程安全队列
        self._audio_queue.put_nowait((resampled_audio_data, stream_id, fragment_id))
        if self._play_done_event.is_set():
            self.logger.debug("%s player start to playing audio", self._log_prefix)
            self._play_done_event.clear()
        if duration > 0.0:
            # 更新预计结束时间
            current_time = time.time()
            if current_time > self._estimated_end_time:
                self._estimated_end_time = current_time + duration
            else:
                self._estimated_end_time += duration
        return self._estimated_end_time

    def _wait_consumed(self, audio_data: np.ndarray) -> None:
        """等待音频数据被设备消费.

        默认用分片 sleep 模拟播放时钟 — 每 10ms 检查 _stop_event,
        close() 后最多 10ms 即可响应. worker 是独立 daemon 线程, 不阻塞
        event loop. 子类可覆写以使用设备原生回调.
        """
        duration = len(audio_data) / self.sample_rate if self.sample_rate else 0.0
        if duration <= 0:
            return
        tick = 0.01  # 短阻塞 — close() 响应延迟上限
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline and not self._stop_event.is_set():
            time.sleep(min(tick, max(0, deadline - time.monotonic())))

    def _time_to_wait(self) -> float:
        time_to_wait = (self._estimated_end_time + self._safety_delay) - time.time()
        if time_to_wait > 0.0:
            return time_to_wait
        return 0.0

    async def wait_play_done(self, timeout: Optional[float] = None) -> bool:
        """等待所有音频播放完成"""
        timeleft = None
        if timeout is not None and timeout > 0.0:
            timeleft = Timeleft(timeout)
        time_to_wait = self._time_to_wait()
        self.logger.info("%s start to wait %.2fs for playing", self._log_prefix, time_to_wait)
        while time_to_wait > 0.0:
            # 循环检查预计等待的最后播放时间.
            if timeleft:
                try:
                    await asyncio.wait_for(asyncio.sleep(time_to_wait), timeout=timeleft.left())
                except asyncio.TimeoutError:
                    self.logger.info("%s wait for playing done timeout", self._log_prefix)
                    return False
            else:
                await asyncio.sleep(time_to_wait)
            time_to_wait = self._time_to_wait()
        # 同时等待播放结束.
        await self._play_done_event.wait()
        self.logger.info("%s wait for play done successful", self._log_prefix)
        return True

    def is_playing(self) -> bool:
        """检查是否还有音频在播放"""
        return time.time() < self._estimated_end_time or not self._play_done_event.is_set()

    def is_closed(self) -> bool:
        """检查播放器是否已关闭"""
        return self._closed

    @abstractmethod
    def _audio_stream_start(self):
        pass

    @abstractmethod
    def _audio_stream_stop(self):
        pass

    @abstractmethod
    def _audio_stream_write(self, data: np.ndarray):
        pass

    def _audio_worker(self):
        """音频工作线程：处理阻塞的音频输出操作"""
        try:
            self._audio_stream_start()
            self.logger.info("%s audio stream start", self._log_prefix)

            while not self._stop_event.is_set():
                audio_queue = self._audio_queue
                if audio_queue.empty() and not self._play_done_event.is_set():
                    self._play_done_event.set()
                    for callback in self._on_play_done_callbacks:
                        callback()
                    continue
                try:
                    # 从队列获取音频数据（阻塞调用，但有超时）
                    item = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    # 队列为空，继续循环
                    continue

                if item is None:
                    # 收到停止信号
                    # 通过下一个循环判断应该怎么处理.
                    continue
                audio_data, stream_id, fragment_id = item
                self._play_done_event.clear()
                # 写入音频数据（非阻塞 — 仅放入设备缓冲区）
                self._audio_stream_write(audio_data)
                # on_play 在 write 时刻触发 — 用于 TTS gate 等需要尽早知道
                # "开始播放" 的消费者
                for callback in self._on_play_callbacks:
                    callback(audio_data)
                # 等待音频被设备消费 — 分片 sleep, 每 tick 检查 _stop_event
                self._wait_consumed(audio_data)
                # 消费时刻回调 — 但若在等待期间被 stop/close, 不发
                if not self._stop_event.is_set():
                    self._dispatch_playback_sample(audio_data, stream_id, fragment_id)

        except Exception as e:
            self.logger.exception("%s audio stream fatal error %s", self._log_prefix, e)
        finally:
            # 清理资源 — miniaudio 等实现可能在 stop() 时因内部线程已退出而抛异常.
            try:
                self._audio_stream_stop()
            except Exception:
                self.logger.exception("%s error during stream stop", self._log_prefix)
            self.logger.info("%s audio stream stopped", self._log_prefix)

    def _dispatch_playback_sample(self, audio_data: np.ndarray, stream_id: str, fragment_id: str) -> None:
        """在音频真正写入设备的时刻, 计算并分发 PlaybackSample.

        携带原始 PCM bytes + 响度摘要 (rms_db / peak), 不预加工频谱.
        无观察者直接跳过 — 避免不必要的 bytes 拷贝与计算.
        """
        if not self._playback_observers:
            return
        duration = len(audio_data) / self.sample_rate if self.sample_rate else 0.0
        sample = self._compute_playback_sample(
            audio_data, stream_id=stream_id, fragment_id=fragment_id, duration=duration
        )
        for callback in list(self._playback_observers):
            callback(sample)

    def _compute_playback_sample(
        self,
        audio_data: np.ndarray,
        *,
        stream_id: str,
        fragment_id: str,
        duration: float,
    ) -> PlaybackSample:
        """从 resampled int16 PCM 构造 PlaybackSample: raw bytes + 响度摘要."""
        f32 = audio_data.astype(np.float64) / 32768.0
        if len(f32) == 0:
            return PlaybackSample(
                stream_id=stream_id, fragment_id=fragment_id, duration=duration, sample_rate=self.sample_rate,
            )
        rms = float(np.sqrt(np.mean(f32**2)))
        rms_db = 20.0 * np.log10(max(rms, 1e-10))
        peak = float(np.max(np.abs(f32)))

        return PlaybackSample(
            pcm=audio_data.tobytes(),
            stream_id=stream_id,
            fragment_id=fragment_id,
            timestamp=time.time(),
            duration=duration,
            sample_rate=self.sample_rate,
            rms_db=round(rms_db, 1),
            peak=round(peak, 3),
        )

    @staticmethod
    def _compute_spectrum_bins(audio_data: np.ndarray, n_bins: int = 16) -> list[float]:
        """Compute N equal-width spectrum bins from int16 PCM, returning dB values."""
        if len(audio_data) == 0:
            return [-96.0] * n_bins
        f32 = audio_data.astype(np.float64) / 32768.0
        fft = np.abs(np.fft.rfft(f32))
        n_fft = len(fft)
        if n_fft < n_bins * 2:
            return [float(20.0 * np.log10(max(fft.mean(), 1e-10)))] * n_bins
        bins = []
        for i in range(n_bins):
            lo = int(i * n_fft / n_bins)
            hi = int((i + 1) * n_fft / n_bins)
            db = 20.0 * np.log10(max(float(fft[lo:hi].mean()), 1e-10))
            bins.append(round(db, 1))
        return bins
