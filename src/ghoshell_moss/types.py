from pydantic import BaseModel, Field
from ghoshell_moss.message import Message

"""
创建一个独立的类型存放位置.
核心目的是缩短一些特殊类型的引用路径. 
"""

__all__ = ['Observe']


class Observe(BaseModel):
    """
    Command 的特殊返回值, 当 Command 返回这一结构时, 会立刻中断 Shell Interpreter 的返回值.
    """
    messages: list[Message] = Field(
        default_factory=list,
        description="ghoshell_moss.core.concepts.command:CommandTask 的特殊返回值类型."
    )
