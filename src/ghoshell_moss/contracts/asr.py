"""
ASR contracts — audio speech recognition abstractions.

ASR is the ear: raw audio → text stream. Kept separate from speech (the mouth)
to avoid cross-infection in the contract layer.

自解释面 (镜像 TTSInfo): get_info() 暴露音频输入契约 (sample_rate/bits/channel)
+ 可调行为参数的 json schema 与当前值; configure() 设置行为参数, 作用于下一次
recognize(). 模型身份由工厂/provider 在创建时固定, 不属于运行时自解释面.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterable, Callable

import numpy as np
from pydantic import BaseModel, Field

__all__ = [
    "ASR",
    "ASRInfo",
    "RecognitionStream",
    "RecognitionPhase",
    "RecognitionResult",
    "RecognitionSegment",
]


class RecognitionPhase(str, Enum):
    """识别结果的三层相位 — 对齐火山引擎的响应语义.

    - PARTIAL: 中间结果 (utterance definite=false), 边说话边出字.
    - CLAUSE:   稳定分句 (utterance definite=true).
    - TAIL:     尾包 (帧级 is_last_package=true), commit / 音频断 后.

    注意 ``definite`` 只标记"这一句稳定了", 不是流结束 — 流结束由帧级
    ``is_last_package`` 承载 (对应 TAIL). 混淆两者会丢音频.
    """

    PARTIAL = "partial"
    CLAUSE = "clause"
    TAIL = "tail"


@dataclass
class RecognitionResult:
    """识别流在某个 phase 产出的一个完整结果 (text 轴).

    平铺字段, 不做嵌套 — 分句结构就是 text + start_ms + end_ms 三个字段.
    ``segment_id`` 关联同一 segment 的 RecognitionSegment (即其 ``id``).
    """

    stream_id: str
    segment_id: str  # segment id
    phase: RecognitionPhase
    text: str
    start_ms: int = 0
    end_ms: int = 0
    error: str = ""


@dataclass
class RecognitionSegment:
    """一个 segment (一次 turn) 的音频留档 (audio 轴).

    每个 tail 切一次 — ``text`` 是整段累积文本, ``audio`` 是整段累积音频.
    ``start_ms``/``end_ms`` 是段的流相对时间戳; ``offset_ms`` 是 ``audio`` 起点的
    流相对时间, 供 ``precise_cut`` 精确切片.

    与 RecognitionResult 通过 ``segment_id`` (即 ``id``) + ``stream_id`` 关联, 走
    独立回调 (on_segment), 不混进 text 轴.
    """

    id: str  # segment id
    stream_id: str # stream id
    text: str = ""
    start_ms: int = 0
    end_ms: int = 0
    sample_rate: int = 16000
    bits: int = 16
    channel: int = 1
    audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    offset_ms: int = 0

    def precise_cut(self) -> np.ndarray:
        """按 start_ms/end_ms 把粗略切 audio 精确切片."""
        sr = self.sample_rate or 1
        start = int((self.start_ms - self.offset_ms) * sr / 1000)
        end = int((self.end_ms - self.offset_ms) * sr / 1000)
        start = max(0, min(start, len(self.audio)))
        end = max(start, min(end, len(self.audio)))
        return self.audio[start:end]


class ASRInfo(BaseModel):
    """ASR 运行时自解释信息 — 镜像 TTSInfo.

    模型先 get_info() 读: 音频输入契约 (sample_rate/bits/channel) + 可调行为
    参数的 json schema 与当前值. 再 configure() 调行为旋钮, 作用于下一次
    recognize(). 各实现暴露自己的 params BaseModel, 契约只背 schema 与当前值两个 dict.
    """

    sample_rate: int = Field(default=16000, description="识别期望的音频采样率")
    bits: int = Field(default=16, description="位深")
    channel: int = Field(default=1, description="通道数")

    params_schema: dict = Field(default_factory=dict,
                                description="可调行为参数的 json schema (各实现暴露自己的 BaseModel)")
    params: dict = Field(default_factory=dict, description="当前行为参数值")


class RecognitionStream(ABC):
    """连续识别 loop — 一条连续音频流 → 一串 RecognitionResult.

    1 stream = ( n segment = ( m result) )

    迭代到音频输入断时: 自动 commit → 执行到最后一帧拿 final → 吐 phase=TAIL →
    loop 结束. 一个周期可以包含多个 CLAUSE 分句, 中间夹着 PARTIAL.

    只允许进入一次 (__aiter__ 不可重入). 若音频提供方输入中断, recognition
    loop 也自然终止 — ``is_input_done()`` 返回是否已停止.
    """

    @property
    @abstractmethod
    def stream_id(self) -> str:
        ...

    @abstractmethod
    def on_segment(self, callback: Callable[[RecognitionSegment], None]) -> None:
        """注册回调, 每个分句切分时投递 audio 轴结果 (RecognitionSegment). """

    @abstractmethod
    def commit(self) -> None:
        """提交当前状态, 尽快获得一个 TAIL (尾包) 切分. """

    @abstractmethod
    def is_input_done(self) -> bool:
        """音频输入的循环是否已停止. """

    @abstractmethod
    def __aiter__(self) -> "RecognitionStream":
        """开启音频发送, 到音频流结束为止. 只允许进入一次."""

    @abstractmethod
    async def __anext__(self) -> RecognitionResult:
        ...


class ASR(ABC):
    """Audio perception organ — ear. Symmetric to TTS (mouth).

    输入: 1-D int16 PCM 音频流 (调用方理解 ASRInfo 后传入重采样对齐的数据).
    输出: 连续识别 loop (Recognition) — partial/clause/tail 三种相位的结果流.
    """

    @abstractmethod
    def get_info(self) -> ASRInfo:
        """返回运行时自解释信息 — 音频契约 + 可调参数的 schema 与当前值."""

    @abstractmethod
    def configure(self, params: dict) -> None:
        """设置行为参数, 作用于下一次 recognize(). 校验与取值空间由各实现的 params BaseModel 定义."""

    @abstractmethod
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """运行时异常观测接口 — 连接断链等长程故障经此上报, 与逐结果的 error 字段正交."""

    @abstractmethod
    def recognize(
            self,
            audio_chunks: AsyncIterable[np.ndarray],
            *,
            stream_id: str | None = None,
    ) -> RecognitionStream:
        """开启一个连续识别 loop, 消费音频流, 返回 Recognition. """

    async def recognize_once(self, audio_chunks: AsyncIterable[np.ndarray]) -> str:
        """Recognize a complete audio stream, return the accumulated text. Default implementation."""
        texts: list[str] = []
        async for result in self.recognize(audio_chunks):
            if result.phase == RecognitionPhase.TAIL:
                texts.append(result.text)
                break
            if result.phase == RecognitionPhase.CLAUSE:
                texts.append(result.text)
        return "".join(texts)

    @abstractmethod
    async def close(self) -> None:
        """Release ASR resources."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
