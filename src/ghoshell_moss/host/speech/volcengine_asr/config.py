import os

from pydantic import BaseModel, Field
from typing_extensions import Self


class VolcengineASRConfig(BaseModel):
    """火山引擎大模型 ASR 配置。

    环境变量:
        VOLCENGINE_BM_ASR_APPID  — appid
        VOLCENGINE_BM_ASR_TOKEN  — access token
        VOLCENGINE_BM_ASR_API_KEY — new-console API key, optional
        VOLCENGINE_BM_ASR_URL — websocket url override
        VOLCENGINE_BM_ASR_RESOURCE_ID — resource id override
        VOLCENGINE_BM_ASR_MODEL_NAME — model name override
    """

    appid: str = Field("$VOLCENGINE_BM_ASR_APPID", description="火山引擎 asr 的 appid")
    token: str = Field("$VOLCENGINE_BM_ASR_TOKEN", description="火山引擎的 asr app token")
    api_key: str = Field("$VOLCENGINE_BM_ASR_API_KEY", description="新版控制台 API Key")
    # 默认 URL 保持历史兼容。新版 async 端点可通过 VOLCENGINE_BM_ASR_URL
    # 覆盖，例如 wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async。
    url: str = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
    sample_rate: int = Field(16000, description="默认的采样率")
    bits: int = Field(16)
    channel: int = Field(1)
    model_name: str = Field("bigmodel", description="火山引擎模型类型")
    end_window_size: int = Field(
        500,
        description="静音时长超过该值，直接判停输出 definite。单位 ms。",
    )
    enable_punc: bool = Field(True, description="启用标点")
    enable_ddc: bool = Field(
        True,
        description="语义顺滑，删除停顿词、语气词、语义重复词等。",
    )
    force_to_speech_time: int = Field(
        1000,
        description="音频时长超过该值后才尝试判停。单位 ms，需配合 end_window_size。",
    )
    audio_packet_ms: int = Field(
        200,
        description="发送到火山 ASR 的音频包时长。官方建议 100-200ms；可用环境变量覆盖。",
    )
    resource_id: str = Field("volc.bigasr.sauc.duration")

    def resolve_env(self) -> Self:
        if self.appid.startswith("$"):
            self.appid = os.environ.get(self.appid[1:], self.appid)
        if self.token.startswith("$"):
            self.token = os.environ.get(self.token[1:], self.token)
        if self.api_key.startswith("$"):
            self.api_key = os.environ.get(self.api_key[1:], "")
        self.url = os.environ.get("VOLCENGINE_BM_ASR_URL", self.url)
        self.resource_id = os.environ.get("VOLCENGINE_BM_ASR_RESOURCE_ID", self.resource_id)
        self.model_name = os.environ.get("VOLCENGINE_BM_ASR_MODEL_NAME", self.model_name)
        packet_ms = os.environ.get("VOLCENGINE_BM_ASR_AUDIO_PACKET_MS")
        if packet_ms:
            try:
                self.audio_packet_ms = max(20, int(packet_ms))
            except ValueError:
                pass
        return self
