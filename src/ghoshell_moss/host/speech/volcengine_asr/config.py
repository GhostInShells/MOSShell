import os

from pydantic import BaseModel, Field
from typing_extensions import Self


class VolcengineASRConfig(BaseModel):
    """火山引擎大模型 ASR 配置。

    环境变量:
        VOLCENGINE_BM_ASR_APPID  — appid
        VOLCENGINE_BM_ASR_TOKEN  — access token
    """

    appid: str = Field("$VOLCENGINE_BM_ASR_APPID", description="火山引擎 asr 的 appid")
    token: str = Field("$VOLCENGINE_BM_ASR_TOKEN", description="火山引擎的 asr app token")
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
    resource_id: str = Field("volc.bigasr.sauc.duration")

    def resolve_env(self) -> Self:
        if self.appid.startswith("$"):
            self.appid = os.environ.get(self.appid[1:], self.appid)
        if self.token.startswith("$"):
            self.token = os.environ.get(self.token[1:], self.token)
        return self
