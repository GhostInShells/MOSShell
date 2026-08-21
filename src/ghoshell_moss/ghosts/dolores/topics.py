"""Dolores ghost 的 topic 模型 — dsh session event 等运行时事件载体.

与 core.concepts.topic 的 TopicModel 对齐, 经 TopicService 在运行时发布/消费.
"""

from pydantic import Field

from ghoshell_moss.core.concepts.topic import TopicModel
from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent

__all__ = ["DshSessionEventTopic"]


class DshSessionEventTopic(TopicModel):
    """一个 dsh session 事件的载体 — 不重建模事件载荷, 只挂原始信封 + 路由身份.

    信封 (SessionEvent) 已是强类型 (meta.type/seq/time + data), 需要具体事件时
    消费方按需用 ``SessionEventModel.from_session_event(topic.event)`` 还原
    (如 ``AssistantMessageEvent.from_session_event``). session_id 必须显式携带 —
    SessionEvent 信封本身不含归属 session, 该身份只在传输帧 (MuxFrame) 上.
    """

    session_id: str = Field(default="", description="事件归属的 dsh session id.")
    event: SessionEvent = Field(description="dsh session 事件信封, 原样透传不拆解.")

    @classmethod
    def topic_type(cls) -> str:
        return "dsh/session-event"

    @classmethod
    def default_topic_name(cls) -> str:
        return "dsh/session-event"
