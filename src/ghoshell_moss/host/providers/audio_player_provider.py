from ghoshell_moss.contracts.speech import StreamAudioPlayer
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.configs import ConfigType, ConfigStore
from ghoshell_container import IoCContainer, Provider
from pydantic import Field

__all__ = ["AudioPlayerProvider", "AudioPlayerConfig"]


class AudioPlayerConfig(ConfigType):
    DefaultEnvValues = {"MOSS_AUDIO_PLAYER_DEVICE": ""}

    samplerate: int = Field(
        default=16000,
        description="Sample rate of audio player stream",
    )
    safety_delay: float = Field(
        default=0.1,
        description="Delay for time calculation after player finishes a stream",
    )
    #: 输出设备名子串匹配; 空 = 交给 miniaudio 默认发现. 经 $MOSS_AUDIO_PLAYER_DEVICE
    #: 环境变量配置, 未设置时回退 DefaultEnvValues (空).
    device_pattern: str = Field(
        default="$MOSS_AUDIO_PLAYER_DEVICE",
        description="Output device name substring; empty = miniaudio default discovery",
    )

    @classmethod
    def conf_name(cls) -> str:
        return "audio_player"


class AudioPlayerProvider(Provider[StreamAudioPlayer]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> StreamAudioPlayer:
        from ghoshell_moss.host.speech.player.miniaudio_player import MiniAudioStreamPlayer

        store = con.force_fetch(ConfigStore)
        conf = store.get_or_create(AudioPlayerConfig())
        logger = con.force_fetch(LoggerItf)
        return MiniAudioStreamPlayer(
            sample_rate=conf.samplerate,
            channels=1,
            logger=logger,
            safety_delay=conf.safety_delay,
            device_pattern=conf.device_pattern,
        )
