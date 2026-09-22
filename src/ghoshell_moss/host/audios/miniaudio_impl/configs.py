from ghoshell_moss.contracts.configs import ConfigType
from ghoshell_moss.core.blueprint.environment_options import (
    ENV_AUDIO_CAPTURE_DEVICE_KEY,
    ENV_AUDIO_PLAYER_DEVICE_KEY,
)
from pydantic import BaseModel, Field

__all__ = [
    "CaptureConfig",
    "MiniAudioFactoryConfig",
    "PlayerConfig",
    "AECConfig",
]


class CaptureConfig(BaseModel):
    """Format consensus — consumers read this to know stream parameters."""

    sample_rate: int = 16000
    channels: int = 1
    format: str = "pcm_s16le"
    frame_duration_ms: int = 50
    #: 输入设备名子串匹配; 空 = 交给 miniaudio 默认发现. 经 $MOSS_AUDIO_CAPTURE_DEVICE
    #: 环境变量配置, 未设置时回退 DefaultEnvValues (空).
    device_pattern: str = f"${ENV_AUDIO_CAPTURE_DEVICE_KEY}"


class PlayerConfig(BaseModel):
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
        default=f"${ENV_AUDIO_PLAYER_DEVICE_KEY}",
        description="Output device name substring; empty = miniaudio default discovery",
    )


class AECConfig(BaseModel):
    """回声消除开关与参数 (AEC 实现自己的配置, 不属于 capture/player)."""

    enabled: bool = True
    #: 延迟 hint (0 = 估计器自寻). 两条 stream 同源后只剩设备缓冲这一个常量,
    #: 已知时填真值可省掉估计器的收敛期.
    stream_delay_ms: int = 0
    #: far 环形缓冲的防御上限 (秒). 溢出时丢最老的未消费参考, 属异常工况.
    far_capacity_s: float = 2.0


class MiniAudioFactoryConfig(ConfigType):
    DefaultEnvValues = {
        ENV_AUDIO_PLAYER_DEVICE_KEY: "",
        ENV_AUDIO_CAPTURE_DEVICE_KEY: "",
    }

    capture: CaptureConfig = Field(
        default_factory=CaptureConfig,
    )
    player: PlayerConfig = Field(
        default_factory=PlayerConfig,
    )
    aec: AECConfig = Field(
        default_factory=AECConfig,
    )

    @classmethod
    def conf_name(cls) -> str:
        return "miniaudio_factory"
