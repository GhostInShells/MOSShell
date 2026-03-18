from abc import ABC, abstractmethod
from typing_extensions import Self
from ghoshell_moss.core.concepts.channel import MutableChannel
from ghoshell_moss.core.concepts.shell import MOSSShell
from ghoshell_container import IoCContainer
from anthropic.types import Message
from pydantic_ai import ModelMessage


class MOSS(ABC):
    """
    MOSShell 的高级抽象封装, 目的是:
    1. 屏蔽底层 shell / interpreter 的具体实现.
    2. 在 Shell 的上层, 针对全双工思考范式, 提供有状态服务. 支持模型的 interactive reasoning.
    3. 支持以工具的形式接入现有的 Agent 生态, 比如用 mcp 的形式接入.
    4. 支持 pydantic ai 实现的双工 Agent. 将流式控制范式推进到流式 思考-观察-行动 范式.

    坚持 Facade 思路, 不暴露任何对用户没有用的 API. 降低用户的心智复杂度.
    让用户自己读源码了解底层的实现与封装.
    """

    @abstractmethod
    async def ctml_run(self, ctml: str):
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """
        用 async 的方式启动.
        """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文, 回收资源.
        """
        pass
