"""Warrant 存储装线的协议 topic 模型.

两个 topic (v8): 一个写请求, 一个真值. 信任方向相反 — 写请求是提案
(谁都能发, 只有 host 落盘才成真), 真值是事实 (只有 host 发).
混在一个 topic 会混淆"我想存"与"已存", 故必须两个.

topic name 走 TopicName pattern, 不能用 QA 的 `_warrant` (那是 QA namespace).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ghoshell_moss.core.concepts.topic import TopicModel, TopicName

WARRANT_WRITE_TOPIC = TopicName("warrant/write")
WARRANT_TRUTH_TOPIC = TopicName("warrant/truth")

__all__ = [
    "WarrantWriteRequest",
    "WarrantTruth",
    "WARRANT_WRITE_TOPIC",
    "WARRANT_TRUTH_TOPIC",
]


class WarrantWriteRequest(TopicModel):
    """非 host -> host: 请求持久化一份授权状态. 提案, 非事实."""

    key: str = Field(description="语言无关唯一键, 对应 permission.key()")
    seq: int = Field(description="本 cell 提议的每 key 单调序号")
    data: dict[str, Any] = Field(description="StateT 序列化本体")

    @classmethod
    def topic_type(cls) -> str:
        return "warrant/write_request"

    @classmethod
    def default_topic_name(cls) -> TopicName:
        return WARRANT_WRITE_TOPIC


class WarrantTruth(TopicModel):
    """host -> all: 某 key 的权威真值 (落盘确认). 事实, 只有 host 发."""

    key: str = Field(description="语言无关唯一键, 对应 permission.key()")
    seq: int = Field(description="该 key 当前权威序号")
    data: dict[str, Any] = Field(description="StateT 序列化本体")

    @classmethod
    def topic_type(cls) -> str:
        return "warrant/truth"

    @classmethod
    def default_topic_name(cls) -> TopicName:
        return WARRANT_TRUTH_TOPIC
