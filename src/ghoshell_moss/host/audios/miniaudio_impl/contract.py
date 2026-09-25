"""miniaudio 实现的一对音频 stream + AEC 装线.

播放与采集由同一个 factory 产出 —— 设备几何在一处声明, AEC 因此在**构造时**就
知道两条时间轴的关系, 而不是事后由 runtime 用 bridge 去猜.

runtime 不再需要 AEC 装线: Speech 取的 ``StreamAudioPlayer`` 就是本 factory 的
player 实例 (provider 单例), far 参考直接从这里挂上去.
"""
from abc import ABC, abstractmethod

from ghoshell_moss.contracts.audio import AudioCaptureSource
from ghoshell_moss.contracts.speech import StreamAudioPlayer


class AbstractMiniAudioFactory(ABC):

    @abstractmethod
    def capture(self) -> AudioCaptureSource:
        ...

    @abstractmethod
    def player(self) -> StreamAudioPlayer:
        ...
