from ghoshell_moss.contracts.speech import StreamAudioPlayer
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.configs import ConfigType, ConfigStore
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.host.speech.capture.matrix_audio_transport import MatrixAudioTransport
from ghoshell_moss.host.speech.player.miniaudio_player import MiniAudioStreamPlayer
from ghoshell_container import IoCContainer, Provider
from pydantic import Field

__all__ = ["AudioPlayerProvider", "AudioPlayerConfig"]


class AudioPlayerConfig(ConfigType):
    samplerate: int = Field(
        default=44100,
        description="Sample rate of audio player stream",
    )
    safety_delay: float = Field(
        default=0.1,
        description="Delay for time calculation after player finishes a stream",
    )

    @classmethod
    def conf_name(cls) -> str:
        return "audio_player"


class AudioPlayerProvider(Provider[StreamAudioPlayer]):

    def singleton(self) -> bool:
        return False

    def factory(self, con: IoCContainer) -> StreamAudioPlayer:
        store = con.force_fetch(ConfigStore)
        conf = store.get_or_create(AudioPlayerConfig())
        logger = con.force_fetch(LoggerItf)
        matrix = con.force_fetch(Matrix)
        transport = MatrixAudioTransport(matrix=matrix)
        return MiniAudioStreamPlayer(
            sample_rate=conf.samplerate,
            channels=1,
            logger=logger,
            safety_delay=conf.safety_delay,
            transport=transport,
        )
