"""miniaudio 实现的一对音频 stream + AEC 装线.

播放与采集由同一个 factory 产出 —— 设备几何在一处声明, AEC 因此在**构造时**就
知道两条时间轴的关系, 而不是事后由 runtime 用 bridge 去猜.

runtime 不再需要 AEC 装线: Speech 取的 ``StreamAudioPlayer`` 就是本 factory 的
player 实例 (provider 单例), far 参考直接从这里挂上去.
"""
from typing import Callable

import numpy as np

from ghoshell_moss.depends import depend_host

depend_host()

from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.contracts.audio import resample
from ghoshell_moss.contracts.workspace import Workspace

from .configs import MiniAudioFactoryConfig
from .miniaudio_capture import MiniAudioCaptureSource
from .miniaudio_player import MiniAudioStreamPlayer

__all__ = ["MiniAudioFactory"]


class MiniAudioFactory:
    """同一进程内的一对 miniaudio stream.

    两个器官各自按需取用; AEC 在**两条 stream 都到手**的那一刻装线 —— 与取用
    顺序无关. 只取一条时不装 AEC: 没有参考就没有回声可消.
    """

    def __init__(
            self,
            *,
            config: MiniAudioFactoryConfig,
            workspace: Workspace,
            logger: LoggerItf,
    ):
        self._config = config
        self._workspace = workspace
        self._logger = logger
        self._player: MiniAudioStreamPlayer | None = None
        self._capture: MiniAudioCaptureSource | None = None
        self._aec = None
        self._dispose_far: Callable[[], None] | None = None

    def player(self) -> MiniAudioStreamPlayer:
        if self._player is None:
            conf = self._config.player
            self._player = MiniAudioStreamPlayer(
                sample_rate=conf.samplerate,
                channels=1,
                logger=self._logger,
                safety_delay=conf.safety_delay,
                device_pattern=conf.device_pattern,
            )
            self._install_aec()
        return self._player

    def capture(self) -> MiniAudioCaptureSource:
        if self._capture is None:
            self._capture = MiniAudioCaptureSource(
                config=self._config.capture,
                workspace=self._workspace,
                logger=self._logger,
            )
            self._install_aec()
        return self._capture

    def shutdown(self) -> None:
        if self._dispose_far is not None:
            self._dispose_far()
            self._dispose_far = None
        if self._capture is not None:
            self._capture.set_aec(None)
        self._aec = None

    def _install_aec(self) -> None:
        """两条 stream 齐了才装 —— 谁先来都在这里收敛."""
        aec_conf = self._config.aec
        if self._aec is not None or not aec_conf.enabled:
            return
        if self._player is None or self._capture is None:
            return

        from ghoshell_moss.host.audios.aec.webrtc_aec import PyWebrtcEchoCanceller

        # near 免重采样: AEC 率取 capture 原生率, far (player) 需要时再转.
        aec = PyWebrtcEchoCanceller(
            sample_rate=self._capture.sample_rate,
            stream_delay_ms=aec_conf.stream_delay_ms,
            far_capacity_s=aec_conf.far_capacity_s,
        )
        self._capture.set_aec(aec)
        self._aec = aec

        play_rate = self._player.sample_rate
        aec_rate = aec.sample_rate

        # far 挂在 on_emit (设备消费时钟) 上, 而不是 on_play (入队时钟) —— 前者相对
        # 出声只有设备缓冲这一个常量偏移, 后者是可变的片段突发 lead, 见根因 2.
        def _on_emit(frame: np.ndarray) -> None:
            arr = np.asarray(frame).ravel()
            if play_rate != aec_rate:
                arr = resample(arr.astype(np.int16), origin_rate=play_rate, target_rate=aec_rate)
            aec.push_far(arr.astype(np.float32) / 32768.0)

        self._dispose_far = self._player.on_emit(_on_emit)
