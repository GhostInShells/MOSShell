from ghoshell_moss.core.concepts.topic import TopicModel, TopicName
from ghoshell_moss.message import Message
from .session import ConversationItem
from pydantic import BaseModel, Field

__all__ = ['OutputTopic', 'InputTopic', 'LogRecordTopic']


class OutputTopic(TopicModel):
    """
    对外输出的消息体.
    """
    item: ConversationItem = Field(
        description="一个消息单元, 可以用于 moss 的渲染."
    )

    @classmethod
    def topic_type(cls) -> str:
        return 'moss/Output'

    @classmethod
    def default_topic_name(cls) -> TopicName:
        return 'moss/output'


class InputTopic(TopicModel):
    """
    系统输入的消息体.
    """

    priority: int = Field(
        default=0,
        description="消息体的优先级",
    )
    incomplete_inputs: list[Message] = Field(
        default_factory=list,
        description="未完成的消息体",
    )
    inputs: list[Message] = Field(
        default_factory=list,
        description="输入的消息体",
    )

    @classmethod
    def topic_type(cls) -> str:
        return 'moss/Input'

    @classmethod
    def default_topic_name(cls) -> TopicName:
        return 'moss/input'


class LogRecordTopic(TopicModel):
    """
    系统的状态描述
    """

    level: str = Field(
        default="INFO",
        description="消息的级别"
    )

    record: str = Field(
        description="消息的内容"
    )

    @classmethod
    def topic_type(cls) -> str:
        return 'moss/LogRecord'

    @classmethod
    def default_topic_name(cls) -> TopicName:
        return 'moss/log'
