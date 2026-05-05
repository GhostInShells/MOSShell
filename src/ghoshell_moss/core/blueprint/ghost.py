from ghoshell_container import IoCContainer, Contracts
from typing_extensions import Self
from abc import ABC, abstractmethod
from ghoshell_moss.core.blueprint.mindflow import Logos, Mindflow, Nucleus, NucleusMeta, Articulator
from ghoshell_moss.core.blueprint.conversation import ConversationStore, Conversation
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message import Message

__all__ = ['Ghost', 'GhostMeta']


class GhostMeta(ABC):
    """
    MOSS 架构中对 AI 的高阶封装抽象.
    底层可以是简单的模型调用, 或者复杂的 Agent 框架.
    只需要对齐几个基础的 API, 就可以被 MOSS 架构启动运行.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Ghost 的名称, 用于被其它场景读取.
        """
        pass

    @abstractmethod
    def nuclei_metas(self) -> list[NucleusMeta]:
        pass

    @classmethod
    def version(cls) -> str:
        """
        返回 Ghost 版本号.
        """
        return ''

    @classmethod
    def prototype(cls) -> str:
        """
        返回 Ghost 型号.
        """
        prototype_name = cls.__name__
        if prototype_name.endswith('Meta'):
            prototype_name = prototype_name[:-4]
        return prototype_name

    @property
    def identifier(self) -> str:
        """约定的 RESTFul 风格 locator. """
        if version := self.version():
            return f"prototypes/{self.prototype()}-{version}/ghosts/{self.name()}"
        else:
            return f"prototypes/{self.prototype()}/ghosts/{self.name()}"

    @abstractmethod
    def description(self) -> str:
        """
        Ghost 的描述.
        """
        pass

    def contracts(self) -> Contracts:
        """
        定义 Ghost 的各种依赖.
        方便在架构运行时理解它是否可以集成.
        通常用于启动时检查.
        """
        return Contracts([

        ])

    @abstractmethod
    def factory(self, container: IoCContainer) -> "Ghost":
        """
        通过环境提供的 IoC 容器, 完成 Ghost 运行时的初始化.
        它许多能力需要通过 Runtime 提供 (实际上依赖了 Moss 运行时环境提供的 session/conversation store 等各种依赖.
        """
        pass


class Ghost(ABC):
    """
    Ghost 的运行时.
    它基于环境提供的依赖启动, 启动后要提供
    能够被 moss 架构所使用的关键 API.

    系统启动的时候, Ghost 和 GhostMeta 都应该设置到全局 IoC 容器里.
    """

    @property
    @abstractmethod
    def meta(self) -> GhostMeta:
        """
        仍然持有自身的 Meta 信息.
        """
        pass

    @abstractmethod
    def system_prompt(self) -> str:
        """
        描述 Ghost 的指令.

        可以理解为其它 Agent 项目里的 SystemPrompt, Instruction, Soul.md 等等.
        这里倾向于通过三种信息构成:
        - existence: ghost 的存在主义描述.
        - purpose: ghost 的目标. 基于 existence 派生.
        - alignment: ghost 的行为, 风格等对齐状态.
        """
        pass

    def memories(self) -> list[Message]:
        """
        Ghost 的动态记忆.
        """
        return []

    def channel(self) -> Channel | None:
        """
        Ghost 反身性控制的 Channel
        如果提供出来, 会以 'ghost' 为 channel 名注册到 Shell 中.
        从而能够让这个 ghost 去控制它. Ghost 的启动时间在 Shell 之前.
        """
        return None

    def nuclei(self) -> list[Nucleus]:
        """
        返回 ghost 的认知模块
        可以基于不同的 signal 产生 impulse, 驱动 Mindflow 运转, 生成 Attention,
        最后通过 articulator 调用 ghost 的 articulate 函数.

        这些 Nuclei 会注册到系统的 mindflow 中.
        """
        return []

    @abstractmethod
    def conversation(self) -> Conversation:
        """
        当前进行中的会话.
        """
        pass

    @abstractmethod
    def convos(self) -> ConversationStore:
        """
        当前 Ghost 实例下存储的会话历史.
        """
        pass

    def mindflow(self) -> Mindflow | None:
        """
        Ghost 定义自身的 Mindflow. 如果返回 None 的话, 会使用 MOSS 架构提供的默认 mindflow 实现.
        Mindflow 不要自己去启动, 交给 MOSS 架构启动.
        """
        return None

    @abstractmethod
    def articulate(self, articulator: Articulator) -> Logos:
        """
        articulate the logos from context
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """
        定义自身的生命周期.
        可能不需要, 也可以通过这个生命周期做一些特殊的管理.
        """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """结束自身生命周期."""
        pass
