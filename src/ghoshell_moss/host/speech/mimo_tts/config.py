import os
from pydantic import BaseModel, Field
from ghoshell_moss.contracts.speech import (
    TTSInfo,
    AudioFormat,
)

__all__ = [
    'MiMoSpeakerConf',
    'MiMoTTSConf',
    'MIMO_VOICE_MAP',
]

MIMO_VOICE_MAP: dict[str, dict[str, str]] = {
    "冰糖": {"voice": "冰糖", "language": "中文", "gender": "女"},
    "茉莉": {"voice": "茉莉", "language": "中文", "gender": "女"},
    "苏打": {"voice": "苏打", "language": "中文", "gender": "男"},
    "白桦": {"voice": "白桦", "language": "中文", "gender": "男"},
    "Mia": {"voice": "Mia", "language": "English", "gender": "Female"},
    "Chloe": {"voice": "Chloe", "language": "English", "gender": "Female"},
    "Milo": {"voice": "Milo", "language": "English", "gender": "Male"},
    "Dean": {"voice": "Dean", "language": "English", "gender": "Male"},
    "MimoDefault": {"voice": "MimoDefault", "language": "Auto", "gender": "Auto"},
    "DefaultZh": {"voice": "DefaultZh", "language": "中文", "gender": "—"},
    "DefaultEn": {"voice": "DefaultEn", "language": "English", "gender": "—"},
}


class MiMoSpeakerConf(BaseModel):
    """MiMo 音色配置"""

    voice: str = Field(description="MiMo API voice ID")
    description: str = Field(default="", description="音色描述")


class MiMoTTSConf(BaseModel):
    """MiMo HTTP TTS 配置"""

    api_key: str = Field(default="$MIMO_API_KEY", description="MiMo API key")
    base_url: str = Field(
        default="https://api.xiaomimimo.com/v1",
        description="MiMo API base URL",
    )
    model: str = Field(default="mimo-v2.5-tts", description="TTS 模型")
    sample_rate: int = Field(default=24000, description="音频采样率")
    stream: bool = Field(default=True, description="使用 SSE 流式传输")
    request_timeout: float = Field(default=60.0, description="HTTP 请求超时秒数")

    speakers: dict[str, MiMoSpeakerConf] = Field(
        default_factory=lambda: {
            name: MiMoSpeakerConf(
                voice=info["voice"],
                description=f"language: {info['language']}, gender: {info['gender']}",
            )
            for name, info in MIMO_VOICE_MAP.items()
        },
        description="可用音色",
    )
    default_speaker: str = Field(default="冰糖", description="默认音色展示名")

    @classmethod
    def unwrap_env(cls, value: str, default: str = "") -> str:
        if value.startswith("$"):
            return os.environ.get(value[1:], default)
        return value or default

    def default_speaker_conf(self) -> MiMoSpeakerConf:
        return self.speakers.get(
            self.default_speaker,
            MiMoSpeakerConf(voice=self.default_speaker),
        )

    def to_tts_info(self, current_tone: str = "") -> TTSInfo:
        return TTSInfo(
            sample_rate=self.sample_rate,
            channels=1,
            audio_format=AudioFormat.PCM_S16LE.value,
            tones={key: value.description for key, value in self.speakers.items()},
            current_tone=current_tone or self.default_speaker,
        )
