import os

from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = ["VolcengineASRConfig", "VolcengineASRParams"]


class VolcengineASRParams(BaseModel):
    """火山 ASR 模型可见/可 set 的行为参数 — json schema 反射给模型看.

    与固定配置 (url/凭据/音频格式/模型身份) 分离. configure() 校验后更新,
    作用于下一次 recognize() (每次连接 init 下发, 会话级调参).
    """

    end_window_size: int = Field(
        500,
        description="VAD 静音时长超过该值直接判停输出 definite。单位 ms。",
    )
    force_to_speech_time: int = Field(
        1000,
        description="VAD 等待语音开头的最大时长, 超过则结束本次识别。单位 ms。",
    )
    enable_punc: bool = Field(True, description="启用标点")
    enable_ddc: bool = Field(
        True,
        description="语义顺滑，删除停顿词、语气词、语义重复词等。",
    )


class VolcengineASRConfig(BaseModel):
    """火山引擎大模型 ASR 配置。

    固定部分: url / 凭据 / 音频格式 / 模型身份 (每实例固定, 工厂选择模型).
    可变部分: params — 模型可见/可 set 的行为旋钮.

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
    model_name: str = Field("bigmodel", description="火山引擎模型类型 — 每实例固定")
    resource_id: str = Field("volc.bigasr.sauc.duration")

    params: VolcengineASRParams = Field(
        default_factory=VolcengineASRParams,
        description="模型可见/可 set 的行为参数",
    )

    def resolve_env(self) -> Self:
        if self.appid.startswith("$"):
            self.appid = os.environ.get(self.appid[1:], self.appid)
        if self.token.startswith("$"):
            self.token = os.environ.get(self.token[1:], self.token)
        return self
