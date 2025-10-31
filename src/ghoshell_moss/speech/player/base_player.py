import asyncio
import time
import numpy as np
from abc import ABC, abstractmethod
from ghoshell_moss.concepts.speech import StreamAudioPlayer, AudioType
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_common.contracts import LoggerItf
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
        self.audio_type = AudioType.PCM_S16LE
        # self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self._safety_delay = safety_delay
        self._play_done_event = ThreadSafeEvent()
        self._play_done_event.set()
        self._committed = True
        self._estimated_end_time = 0.0
        self._closed = False
        self._lock = asyncio.Lock()

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
        async with self._lock:
            # 清空队列
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

            # 重置时间估计
            self._estimated_end_time = time.time()
            self.logger.info("播放队列已清空")

    async def add(
            self,
            chunk: np.ndarray,
            *,
            audio_type: AudioType,
            rate: int,
            channel: int = 1,
    ) -> float:
        """添加音频片段到播放队列"""
        if self._closed:
            return time.time()

        # 格式转换
        if audio_type == AudioType.PCM_F32LE:
            # float32 [-1, 1] -> int16
            audio_data = (chunk * 32767).astype(np.int16)
        else:
            # 假设已经是 int16
            audio_data = chunk

        # 计算持续时间
        duration = len(audio_data) / (2 * rate)  # 2 bytes/sample

        async with self._lock:
            # 添加到线程安全队列
            try:
                self._audio_queue.put_nowait(audio_data.tobytes())
                self._play_done_event.clear()
            except queue.Full:
                self.logger.warning("音频队列已满，丢弃音频数据")
                return time.time()

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
                    if self._audio_queue.empty() and not self._play_done_event.is_set():
                        self._play_done_event.set()
                    # 从队列获取音频数据（阻塞调用，但有超时）
                    audio_data = self._audio_queue.get(timeout=0.2)

                    if audio_data is None:
                        # 收到停止信号
                        break

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
