from typing import Optional

import numpy as np
from ghoshell_common.contracts import LoggerItf
from reachy_mini import ReachyMini

try:
    import pyaudio
except ImportError as e:
    raise ImportError(f"failed to import audio dependencies, please try to install ghoshell-shell[audio]: {e}")

from ghoshell_moss.speech.player.base_player import BaseAudioStreamPlayer



class ReachyMiniStreamPlayer(BaseAudioStreamPlayer):

    def __init__(
        self,
        mini: ReachyMini,
        *,
        logger: LoggerItf | None = None,
        safety_delay: float = 0.1,
    ):
        """
        基于 PyAudio 的异步音频播放器实现
        使用单独的线程处理阻塞的音频输出操作
        """
        self.mini = mini
        super().__init__(
            sample_rate=self.mini.media.get_output_audio_samplerate(),
            channels=self.mini.media.get_output_channels(),
            logger=logger,
            safety_delay=safety_delay,
        )

    def _audio_stream_start(self):
        self.mini.media.start_playing()

    def _audio_stream_stop(self):
        self.mini.media.stop_playing()

    def _audio_stream_write(self, data: np.ndarray):
        # 1. 校验输入数据的合法性（避免格式错误）
        if data.dtype != np.int16:
            raise ValueError(f"输入数据类型错误，需要int16，实际是{data.dtype}")
        if len(data.shape) != 1:
            raise ValueError(f"输入数据维度错误，需要一维数组，实际是{data.shape}")

        # 2. 核心转换：int16 → float32（归一化到音频标准范围[-1.0, 1.0]）
        audio_f32 = data.astype(np.float32) / 32768.0

        # 3. 单声道转双通道（复制左声道数据到右声道，硬件要求双通道）
        audio_f32_stereo = np.column_stack((audio_f32, audio_f32))

        self.mini.media.push_audio_sample(audio_f32_stereo)
