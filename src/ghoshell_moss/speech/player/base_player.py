import asyncio
import time
import numpy as np
from abc import ABC, abstractmethod
from ghoshell_moss.concepts.speech import StreamAudioPlayer, AudioFormat
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_common.contracts import LoggerItf
import scipy.signal as signal
import queue
import threading
import logging

from typing import Optional

__all__ = ['BaseAudioStreamPlayer']


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
            safety_delay: float = 0.1,
    ):
        """
        基于 PyAudio 的异步音频播放器实现
        使用单独的线程处理阻塞的音频输出操作
        """
        self.logger = logger or logging.getLogger("PyAudioPlayer")
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
        self._audio_queue = queue.Queue()
        self._thread = None
        self._stop_event = threading.Event()

    async def start(self) -> None:
        """启动音频播放器"""
        if self._thread and self._thread.is_alive():
            return

        # 启动音频工作线程
        self._thread = threading.Thread(target=self._audio_worker, daemon=True)
        self._thread.start()
        self.logger.info("PyAudio 播放器已启动")

    async def close(self) -> None:
        """关闭音频播放器"""
        self._closed = True
        self._stop_event.set()

        # 等待工作线程结束
        if self._thread and self._thread.is_alive():
            # 放入停止信号
            self._audio_queue.put(None)
            self._thread.join(timeout=2.0)

        self.logger.info("PyAudio 播放器已关闭")

    async def clear(self) -> None:
        """清空播放队列并重置"""
        # 清空队列
        old_queue = self._audio_queue
        self._audio_queue = queue.Queue()
        while not old_queue.empty():
            try:
                old_queue.get_nowait()
            except queue.Empty:
                break

        # 重置时间估计
        self._estimated_end_time = time.time()
        self.logger.info("播放队列已清空")

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
            raise TypeError("audio_data必须是numpy数组")

        if origin_rate <= 0 or target_rate <= 0:
            raise ValueError("采样率必须大于0")

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
        """添加音频片段到播放队列"""
        if self._closed:
            return time.time()

        # 格式转换
        if audio_type == AudioFormat.PCM_F32LE:
            # float32 [-1, 1] -> int16
            audio_data = (chunk * 32767).astype(np.int16)
        else:
            # 假设已经是 int16
            audio_data = chunk.astype(np.int16)

        # 计算持续时间
        duration = len(audio_data) / (2 * rate)  # 2 bytes/sample
        resampled_audio_data = self.resample(audio_data, origin_rate=rate, target_rate=self.sample_rate)

        # 添加到线程安全队列
        data = resampled_audio_data.tobytes()
        self._audio_queue.put_nowait(data)
        self._play_done_event.clear()

        # 更新预计结束时间
        current_time = time.time()
        if current_time > self._estimated_end_time:
            self._estimated_end_time = current_time + duration
        else:
            self._estimated_end_time += duration
        return self._estimated_end_time

    async def wait_play_done(self, timeout: Optional[float] = None) -> bool:
        """等待所有音频播放完成"""
        time_to_wait = (self._estimated_end_time + self._safety_delay) - time.time()
        if time_to_wait > 0.0:
            self.logger.info(f"等待 {time_to_wait:.2f}s 让音频播放完成")
            if timeout is not None and timeout > 0.0:
                try:
                    await asyncio.wait_for(asyncio.sleep(time_to_wait), timeout)
                except asyncio.TimeoutError:
                    self.logger.warning("等待音频播放超时")
                    return False
            else:
                await asyncio.sleep(time_to_wait)

        # 同时等待播放结束.
        await self._play_done_event.wait()
        self.logger.info("音频播放完成")
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
    def _audio_stream_write(self, data: bytes):
        pass

    def _audio_worker(self):
        """音频工作线程：处理阻塞的音频输出操作"""
        try:
            self._audio_stream_start()
            self.logger.info("PyAudio 输出流已创建")

            while not self._stop_event.is_set():
                try:
                    audio_queue = self._audio_queue
                    if audio_queue.empty() and not self._play_done_event.is_set():
                        self._play_done_event.set()
                    # 从队列获取音频数据（阻塞调用，但有超时）
                    audio_data = audio_queue.get(timeout=0.2)

                    if audio_data is None:
                        # 收到停止信号
                        # 通过下一个循环判断应该怎么处理.
                        continue

                    if self._play_done_event.is_set():
                        self._play_done_event.clear()

                    # 写入音频数据（阻塞调用）
                    self._audio_stream_write(audio_data)

                except queue.Empty:
                    # 队列为空，继续循环
                    continue

        except Exception as e:
            self.logger.error(f"音频工作线程错误: {e}")
        finally:
            # 清理资源
            self._audio_stream_stop()
            self.logger.info("音频工作线程已退出")
