from typing import Iterable, Type

from ghoshell_moss.contracts.speech import StreamAudioPlayer
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.configs import ConfigType, ConfigStore
from ghoshell_container import IoCContainer, Provider
from ghoshell_moss.depends import depend_pyaudio

depend_pyaudio()
from ghoshell_moss.core.speech.player.pyaudio_player import PyAudioStreamPlayer
from pydantic import Field

__all__ = ['PyAudioPlayerProvider', 'PyAudioPlayerConfig']


class PyAudioPlayerConfig(ConfigType):
    device_index: int = Field(
        default=0,
        description="Index of device to use in pyaudio stream",
    )
    samplerate: int = Field(
        default=44100,
        description="Sample rate of pyaudio player stream",
    )
    safety_delay: float = Field(
        default=0.1,
        description="Delay for time calculation after pyaudio player play a stream",
    )

    @classmethod
    def conf_name(cls) -> str:
        return 'pyaudio_player'


class PyAudioPlayerProvider(Provider[StreamAudioPlayer]):

    def singleton(self) -> bool:
        return False

    def aliases(self) -> Iterable[Type]:
        yield PyAudioStreamPlayer

    def factory(self, con: IoCContainer) -> StreamAudioPlayer:
        store = con.force_fetch(ConfigStore)
        conf = store.get_or_create(PyAudioPlayerConfig())
        logger = con.force_fetch(LoggerItf)
        return PyAudioStreamPlayer(
            device_index=conf.device_index,
            sample_rate=conf.samplerate,
            channels=1,
            logger=logger,
            safety_delay=conf.safety_delay,
        )
