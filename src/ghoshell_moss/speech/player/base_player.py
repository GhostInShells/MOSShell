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
from scipy import signal

from ghoshell_moss.core.concepts.speech import AudioFormat, StreamAudioPlayer
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
        基于 PyAudio 的异步音频播放器实现
        使用单独的线程处理阻塞的音频输出操作
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

        # 使用线程安全的队列进行线程间通信
        self._audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread = None
        self._stop_event = threading.Event()
        self._on_play_callbacks = []
        self._on_play_done_callbacks = []

    def on_play(self, callback: Callable[[np.ndarray], None]) -> None:
        self._on_play_callbacks.append(callback)

    def on_play_done(self, callback: Callable[[], None]) -> None:
        self._on_play_done_callbacks.append(callback)

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
            self._audio_queue.put(None)
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

    @staticmethod
    def resample(
        audio_data: np.ndarray,
        *,
        origin_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        """使用 scipy.signal.resample 进行采样率转换

        Args:
            audio_data: 原始音频数据
            origin_rate: 原始采样率
            target_rate: 目标采样率

        Returns:
            np.ndarray: 重采样后的音频数据
        """
        if origin_rate == target_rate:
            return audio_data

        if not isinstance(audio_data, np.ndarray):
            raise TypeError("audio_data must be numpy ndarray")

        if origin_rate <= 0 or target_rate <= 0:
            raise ValueError("sample rate must greater than 0")

        number_of_samples = int(len(audio_data) * float(target_rate) / origin_rate)
        resampled_audio_data: np.ndarray = signal.resample(audio_data, number_of_samples)
        return resampled_audio_data.astype(np.int16)

    def add(
        self,
        chunk: np.ndarray,
        *,
        audio_type: AudioFormat,
        rate: int,
        channels: int = 1,
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

        # 计算持续时间
        duration = len(audio_data) / rate
        resampled_audio_data = self.resample(audio_data, origin_rate=rate, target_rate=self.sample_rate)

        # 添加到线程安全队列
        self._audio_queue.put_nowait(resampled_audio_data)
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
        return time.time() < self._estimated_end_time and not self._play_done_event.is_set()

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
                    audio_data = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    # 队列为空，继续循环
                    continue

                if audio_data is None:
                    # 收到停止信号
                    # 通过下一个循环判断应该怎么处理.
                    continue
                self._play_done_event.clear()
                # 写入音频数据（期望是阻塞调用）
                self._audio_stream_write(audio_data)
                for callback in self._on_play_callbacks:
                    callback(audio_data)

        except Exception as e:
            self.logger.exception("%s audio stream fatal error %s", self._log_prefix, e)
        finally:
            # 清理资源
            self._audio_stream_stop()
            self.logger.info("%s audio stream stopped", self._log_prefix)
