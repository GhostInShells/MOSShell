from typing import AsyncIterable
from ghoshell_container import IoCContainer, Contracts, Provider
from typing_extensions import Self
from abc import ABC, abstractmethod
from ghoshell_moss.core.blueprint.mindflow import Mindflow, NucleusFactory, Articulator
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message import Message

__all__ = ['Ghost', 'GhostMeta']


class GhostMeta(ABC):
    """
    Ghost 的 Bootstrapper — 文件即配置，自解释可注册单元.

    放在约定目录下，通过 manifests 机制被系统自动发现。
    携带元信息（name, nuclei, contracts）让系统在实例化前就能理解其协议。
    通过 factory(container) 产出 Ghost 运行时实例。
    """

    @abstractmethod
    def name(self) -> str:
        """
        Ghost 的名称, 用于被其它场景读取.
        """
        pass

    @abstractmethod
    def nuclei_manifests(self) -> list[NucleusFactory]:
        """
        返回可以自解释, 但依赖运行时的 Nucleus Meta
        """
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

    def providers(self) -> list[Provider]:
        """
        ghost 可以提供自身所需的依赖, 在启动前完成注册.
        如果环境中有注册 container.bound(contract) is True, 则优先用 moss 环境里的注册.
        """
        return []

    @abstractmethod
    def factory(self, container: IoCContainer) -> "Ghost":
        """
        通过环境提供的 IoC 容器, 完成 Ghost 运行时的初始化.
        它许多能力需要通过 Runtime 提供 (实际上依赖了 Moss 运行时环境提供的 session/conversation store 等各种依赖.
        """
        pass


class Ghost(ABC):
    """
    Ghost 的运行时实例，由 GhostMeta.factory(container) 产出.

    它是一个自解释的 Adapter — 大部分方法是 hook（有默认返回值），
    不是 abstractmethod。hook 存在本身 = code as prompt：
    让模型知道"这个能力在框架中可被扩展"。

    必须实现的 abstractmethod 只有少数几个：
    articulate(), system_prompt(), __aenter__, __aexit__.
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

    def mindflow(self) -> Mindflow | None:
        """
        Ghost 定义自身的 Mindflow. 如果返回 None 的话, 会使用 MOSS 架构提供的默认 mindflow 实现.
        Mindflow 不要自己去启动, 交给 MOSS 架构启动.
        """
        return None

    @abstractmethod
    def articulate(self, articulator: Articulator) -> AsyncIterable[str]:
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


# ── 三层抽象 ──────────────────────────────────────────────
#
#   GhostPrototype   = type[GhostMeta]    # class，一族 ghost 的"型号"
#   GhostBootstrapper = GhostMeta(...)    # instance，文件即配置，自解释可注册单元
#   GhostRuntime      = Ghost             # instance，由 bootstrapper.factory(container) 产出
#
# 一个文件 = 一个 GhostMeta 实例 = 一个 Ghost 注册。
# 系统先发现 Bootstrapper（理解元信息/契约），运行时通过 factory() 生成 Runtime。

