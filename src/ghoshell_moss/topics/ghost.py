"""
ghost 运行状态相关的 topic 广播. | 系统级 | alpha
"""
from typing import Literal
from ghoshell_moss.core import TopicName
from ghoshell_moss.core.concepts.topic import TopicModel
from ghoshell_moss.core.blueprint.mindflow import ThinkingEffort
from pydantic import BaseModel, Field


class GhostStateTopic(TopicModel):
    """
    用来记录 Ghost 的输出过程的状态变更.
    """

    ghost: str = Field(
        default='',
        description='ghost name',
    )
    state: Literal['articulating', 'waiting', 'idle'] = Field(
        description='thinking state, 注意不要做 100% 流式 topic 收益不大, 可以分段做. '
    )

    @classmethod
    def topic_type(cls) -> str:
        return 'ghost/State'

    @classmethod
    def default_topic_name(cls) -> TopicName:
        return 'ghost/state'
