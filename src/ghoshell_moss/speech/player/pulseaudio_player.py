import asyncio
import time
import numpy as np
from ghoshell_moss.depends import check_pulseaudio
from ghoshell_moss.concepts.speech import StreamAudioPlayer, AudioType
from ghoshell_common.contracts import LoggerItf
import queue
import threading
import logging

if check_pulseaudio():
    import pulsectl


class PulseAudioStreamPlayer(StreamAudioPlayer):
    """
    基于 PulseAudio 的异步音频播放器实现
    使用单独的线程处理音频输出，通过队列进行通信
    """

    def __init__(
            self,
            *,
            sink_name: str | None = None,
            sample_rate: int = 16000,
            channels: int = 1,
            logger: LoggerItf | None = None,
            safety_delay: float = 0.1,
    ):
        """
        基于 PulseAudio 的异步音频播放器实现
        使用单独的线程处理阻塞的音频输出操作
        """
        self.safety_delay = safety_delay
        self.logger = logger or logging.getLogger("PulseAudioPlayer")
        self.audio_type = AudioType.PCM_S16LE
        self.sink_name = sink_name  # PulseAudio 接收器名称
        self.sample_rate = sample_rate
        self.channels = channels
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
        self.logger.info("PulseAudio 播放器已启动")

    async def close(self) -> None:
        """关闭音频播放器"""
        self._closed = True
        self._stop_event.set()

        # 等待工作线程结束
        if self._thread and self._thread.is_alive():
            # 放入停止信号
            self._audio_queue.put(None)
            self._thread.join(timeout=2.0)

        self.logger.info("PulseAudio 播放器已关闭")

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
            audio_type: AudioType,
            channel: int,
            rate: int,
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
        duration += self.safety_delay

        async with self._lock:
            # 添加到线程安全队列
            try:
                self._audio_queue.put_nowait(audio_data.tobytes())
            except queue.Full:
                self.logger.warning("音频队列已满，丢弃音频数据")
                return time.time()

            # 更新预计结束时间
            current_time = time.time()
            if current_time > self._estimated_end_time:
                self._estimated_end_time = current_time + duration
            else:
                self._estimated_end_time += duration

            self.logger.debug(f"已添加音频片段，预计 {duration:.2f}s 后播放完成")
            return self._estimated_end_time

    async def wait_played(self, timeout: float | None = None) -> None:
        """等待所有音频播放完成"""
        async with self._lock:
            time_to_wait = self._estimated_end_time - time.time()
            if time_to_wait > 0:
                self.logger.info(f"等待 {time_to_wait:.2f}s 让音频播放完成")
                try:
                    await asyncio.wait_for(asyncio.sleep(time_to_wait), timeout)
                except asyncio.TimeoutError:
                    self.logger.warning("等待音频播放超时")
                self.logger.info("音频播放完成")

    def is_playing(self) -> bool:
        """检查是否还有音频在播放"""
        return time.time() < self._estimated_end_time

    def is_closed(self) -> bool:
        """检查播放器是否已关闭"""
        return self._closed

    def _audio_worker(self):
        """音频工作线程：处理阻塞的音频输出操作"""
        pulse = None
        stream = None

        try:
            # 创建 PulseAudio 连接
            pulse = pulsectl.Pulse('moshell-audio-player')

            # 获取接收器（如果没有指定，使用默认接收器）
            if self.sink_name is None:
                self.sink_name = pulse.server_info().default_sink_name
                self.logger.info(f"使用默认音频设备: {self.sink_name}")

            # 创建音频流
            stream_props = {
                'media.name': 'MOSShell Audio Stream',
                'application.name': 'MOSShell',
            }

            stream = pulse.stream_connect_playback(
                device=self.sink_name,
                stream_name='moshell-audio-stream',
                format='s16le',  # PulseAudio 使用字符串格式
                rate=self.sample_rate,
                channels=self.channels,
                properties=stream_props
            )

            self.logger.info("PulseAudio 输出流已创建")

            while not self._stop_event.is_set():
                try:
                    # 从队列获取音频数据（阻塞调用，但有超时）
                    audio_data = self._audio_queue.get(timeout=0.1)

                    if audio_data is None:
                        # 收到停止信号
                        break

                    # 写入音频数据（阻塞调用）
                    pulse.stream_write(stream, audio_data)

                except queue.Empty:
                    # 队列为空，继续循环
                    continue
                except Exception as e:
                    self.logger.error(f"音频输出错误: {e}")
                    break

        except Exception as e:
            self.logger.error(f"音频工作线程错误: {e}")
        finally:
            # 清理资源
            if stream:
                try:
                    pulse.stream_kill(stream)
                except:
                    pass
            if pulse:
                try:
                    pulse.close()
                except:
                    pass
            self.logger.info("音频工作线程已退出")


async def test_pulse_audio_player():
    """测试 PulseAudio 播放器"""
    player = PulseAudioStreamPlayer(
        sink_name=None,  # 使用默认设备
        sample_rate=16000,
        channels=1,
        safety_delay=0.1
    )

    try:
        async with player:
            # 生成测试音频：1秒的440Hz正弦波
            sample_rate = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            sine_wave = 0.3 * np.sin(2 * np.pi * 440 * t)

            # 添加音频片段
            end_time = await player.add(sine_wave, AudioType.PCM_F32LE, 0, sample_rate)
            print(f"预计完成时间: {end_time}")

            # 等待播放完成
            await player.wait_played()

            # 检查状态
            print(f"是否仍在播放: {player.is_playing()}")
            print(f"是否已关闭: {player.is_closed()}")

    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_pulse_audio_player())
