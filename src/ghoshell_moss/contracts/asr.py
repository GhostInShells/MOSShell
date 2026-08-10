"""
ASR contracts — audio speech recognition abstractions.

ASR is the ear: raw audio → text stream. Kept separate from speech (the mouth)
to avoid cross-infection in the contract layer.

模型自解释面 (镜像 TTSInfo): get_info() 暴露音频输入契约 + 当前模型身份 +
可调行为参数的 json schema 与当前值; configure() 设置行为参数, 作用于下一次
recognize(). 模型身份是每实例固定的 — 模型选择由工厂/provider 负责, 不是
让一个有状态实现自己切换模型 (不同模型可能协议冲突).
"""
from abc import ABC, abstractmethod
from typing import AsyncIterable, NamedTuple

import numpy as np
from pydantic import BaseModel, Field

__all__ = [
    "ASR",
    "ASRInfo",
    "ASRResult",
]


class ASRResult(NamedTuple):
    """ASR recognition result fragment.

    If *error* is non-empty, the recognition encountered a server-side error
    and the text may be empty.  Consumers should surface the error visibly.
    """

    text: str
    is_final: bool = False
    error: str = ""


class ASRInfo(BaseModel):
    """ASR 运行时自解释信息 — 镜像 TTSInfo.

    模型先 get_info() 读: 音频输入契约 (sample_rate/bits/channel) + 当前模型身份 +
    可调行为参数的 json schema 与当前值. 再 configure() 调行为旋钮, 作用于下一次
    recognize(). 各实现暴露自己的 params BaseModel, 契约只背 schema 与当前值两个 dict.
    """

    sample_rate: int = Field(default=16000, description="识别期望的音频采样率")
    bits: int = Field(default=16, description="位深")
    channel: int = Field(default=1, description="通道数")

    model: str = Field(default="", description="当前模型身份 — 固定 per instance, 由工厂/provider 选择")

    params_schema: dict = Field(default_factory=dict, description="可调行为参数的 json schema (各实现暴露自己的 BaseModel)")
    params: dict = Field(default_factory=dict, description="当前行为参数值")


class ASR(ABC):
    """Audio perception organ — ear. Symmetric to TTS (mouth).

    输入: 1-D int16 PCM 音频流, 采样率/位深/通道以 get_info() 返回的 ASRInfo 为准.
    输出: 文本流. text 是累计转写 (非增量), 中间结果 is_final=False, 尾包 is_final=True.
    """

    @abstractmethod
    def get_info(self) -> ASRInfo:
        """返回运行时自解释信息 — 音频契约 + 模型身份 + 可调参数的 schema 与当前值."""

    @abstractmethod
    def configure(self, params: dict) -> None:
        """设置行为参数, 作用于下一次 recognize(). 校验与取值空间由各实现的 params BaseModel 定义."""

    @abstractmethod
    async def recognize(
        self,
        audio_chunks: AsyncIterable[np.ndarray],
    ) -> AsyncIterable[ASRResult]:
        """Streaming recognition. Yields intermediate results; last one is is_final=True."""

    async def recognize_once(self, audio_chunks: AsyncIterable[np.ndarray]) -> str:
        """Recognize a complete audio segment, return final text. Default implementation."""
        async for result in self.recognize(audio_chunks):
            if result.is_final:
                return result.text
        return ""

    @abstractmethod
    async def close(self) -> None:
        """Release ASR resources."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
