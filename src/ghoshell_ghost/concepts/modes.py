from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any, Type, Callable
from typing_extensions import Self
from ghoshell_container import IoCContainer
from pydantic import BaseModel, Field
from .session import Session

"""
抽象复杂度屏蔽声明: 

GhostMode 抽象是必要的: 
1. 对于定义一个拥有复杂生命周期行为逻辑的 AI 实体, 它需要多个 Mode 来管理不同状态下的资源. 
2. 对于开发者和用户而言, GhostMode 可以强制进入某个状态, 比如 "安全模式", "调试模式". 进入不同的模式, 类似电脑的重启. 
3. 这一层是开发者绝对控制如何实现一个 AI 的保障.

但对于简单项目而言, GhostMode 可能只有一个, 配套的整套抽象对开发者而言就会过于复杂. 

解决办法是, 下层抽象 (GhostMode) 不被上层抽象 (Ghost) 依赖, 上层抽象可以完全屏蔽掉下层. 
从上层抽象开始开发, 深度足够时, 才考虑引入下层抽象解决真实的需求.  
"""


class GhostModeConfig(BaseModel):
    """
    GhostMode 的元信息.
    """
    name: str = Field(
        description="状态的名称, 必须是唯一的. "
    )
    description: str = Field(
        description="状态的描述. 让 AI 理解什么时候切换. "
    )
    routes: list[str] = Field(
        default_factory=list,
        description="这个状态可以通向的其它状态 name. 会暴露给 AI 让它了解何时可以切换状态"
    )
    driver: str = Field(
        description="目标驱动类的 ID. "
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description=""
    )


class GhostModeMeta(BaseModel, ABC):
    """
    一个 GhostMode 的具体可配置项.
    基础范式是:
    >>> def make_mode(driver: GhostModeDriver, config: GhostModeConfig) -> GhostMode:
    >>>     return driver.create(config)
    """

    driver_name = Field(
        default="",
        description="必须存在的 driver name 配置项. 默认"
    )

    @classmethod
    @abstractmethod
    def default_driver_name(cls) -> str:
        """
        每一种 GhostModeConfig 都应该对应一个指定的 Driver.
        但有可能有 Driver 版本升级之类的问题, 两个以上的 Driver 共用一个 GhostModeConfig 类型.
        """
        pass

    def get_driver_name(self) -> str:
        """
        获取当前的 DriverName.
        """
        return self.driver_name or self.default_driver_name()

    def to_config(
            self,
            *,
            name: str,
            description: str = "",
            routes: list[str] | None = None,
    ) -> GhostModeConfig:
        """
        转换为一个 Meta 数据.
        """
        driver_name = self.get_driver_name()
        return GhostModeConfig(
            name=name,
            description=description,
            driver_name=driver_name,
            routes=routes or [],
            data=self.model_dump(exclude_none=True),
        )


GHOST_STATE_META = TypeVar('GHOST_STATE_META', bound=GhostModeMeta)

StopFunc = Callable[[], None]


class GhostMode(Generic[GHOST_STATE_META], ABC):
    """
    # 介绍

    Ghost 的仿生生命周期状态机.
    一个 Ghost 在长时间运行时, 拥有多个基础的生命周期状态.
    每个状态拥有的资源, 能力是不同的. 通过状态流转来实现不同的生物行为.
    同时, 并不是每一个 Mode 都需要拥有智力.

    举个例子, 一个机器人有 静坐/行动 两种状态, 静坐时它的脚是不能动的.
    又比如一个机器狗, 它在纯粹的小狗模式下, 可以让它无法说话.
    最后, 开发者可以强制让 AI 进入某个 LifeMode, 进行针对性的管控.

    # 控制

    GhostMode 对于开发者而言, 切换是透明的. 可以通过界面来操作.
    而 AI 也可以通过允许的能力, 在指定的状态中切换 (通常也是接受人类的命令).
    对于 用户 & 开发者  而言,  Ghost 进入了特定的状态后, 就可以暴露不同的操作 & 交互方式去管理它.

    # 状态切换

    GhostMode 无论通过 AI 自身 / 用户 / 开发者  进行切换, 对于整个 Ghost 而言都需要经过切换过程.
    切换过程要完成的包括:
    1. last mode close:
        - 资源关闭 - 资源交接
        - 对话历史管理
    2. new mode start:
        - 资源交接 - 资源启动
        - 初始化运行.

    GhostMode 并不能完全控制整个 Ghost 的生命周期, 开发者/用户 对生命周期的控制是最高优而且非阻塞的.
    当开发者要强制切换 GhostMode 时, 应该要做到立刻生效.

    # 启动与关闭

    GhostMode 涉及资源管理, 所以它不应该是常驻的实例, 否则在不同 Mode 切换时, 资源的生命周期管理会冲突.
    现在假设 GhostMode 切换的时候, 它自己管理的资源都要经过关闭和重启.

    # AI 自主切换

    AI 自主切换状态的行为, 首先受到 Routes 的约束. 它

    # 可控制

    GhostMode 运行时应该要对外暴露 API, 可扩展的 API 让它可以被控制和调试.
    这些 API 并不是为 AI 交互准备的, 是为界面控制和操作准备的.

    # 上下文继承.

    # 异常机制:
    - 如果一个 GhostMode 运行时发生了不可修复的 FatalError, 则应该由外部切换回合理的状态.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """
        Mode 实际上被实例化出来的.
        所以每次实例化, 需要生成一个唯一的 ID.
        我们称为 mode_id.
        在一个 Ghost 完整的生命周期中, 各种维度是洋葱式的嵌套关系, 举例:
        GhostId [ SessionId [ ModeId [ MindId [ TurnId [...] ] ] ] ]

        各种数据生产的状态还原, 都要通过 Id 来对齐.
        """
        pass

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        GhostMode 有自己独立的资源管理体系
        所以需要有一个自己持有的 IoC Container 来屏蔽各种能力的复杂抽象依赖关系.
        """
        pass

    @property
    def description(self) -> str:
        """
        返回当前的描述.
        """
        return self.config.description

    @property
    def name(self) -> str:
        """
        返回当前的名称.
        """
        return self.config.name

    @property
    @abstractmethod
    def meta(self) -> GHOST_STATE_META:
        """
        返回从 Config 中解析出来的 Meta 数据结构.
        """
        pass

    @property
    @abstractmethod
    def config(self) -> GhostModeConfig:
        """
        config 配置项. 运行时不会变更.
        """
        pass

    @abstractmethod
    def __enter__(self) -> Self:
        """
        GhostMode 应该是在同步生命周期中支持 asyncio 的阻塞.
        所以 enter / exit 不应该支持异步.
        对于 Ghost 而言, 运行时的 GhostMode 必须是唯一的.
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        GhostMode 的资源清理逻辑.
        """
        pass

    @abstractmethod
    async def run(self, session: Session) -> None:
        """
        由 GhostMode 完全接管一个 Session.
        Ghost 让 GhostMode 托管了 Session 层面的功能, 但更上层的功能交给 GhostRuntime 管理控制.

        >>> async def run_mode(mode: GhostMode, session: Session) -> None:
        >>>     import asyncio
        >>>     with mode:
        >>>          # 这个 task 可以被控制逻辑按需中断.
        >>>          task = asyncio.create_task(mode.run(session))
        >>>          await task

        :return: 返回一个调用者可以安全阻塞的 Future 对象, 和 cancel 函数.
        """
        pass


class GhostModeDriver(Generic[GHOST_STATE_META], ABC):
    """
    GhostMode 的驱动, 用来实例化具体的 GhostMode.
    拆分 Driver 与 Mode 的核心目标有3个:

    0. 核心开发者定义的通用 GhostMode, 可以被快速地配置出来.
    1. 将 GhostMode 变成可配置的, 从而可以在 UI 界面上完成一个 GhostMode 的定义. 实际上定义的是 GhostModeConfig.
    2. 对于未来的 Meta-Agent 而言, 可以通过定义一个 Pydantic BaseModel 的方式, 定义一个新的 GhostMode. 是一种自迭代范式.
    """

    @abstractmethod
    def name(self) -> str:
        """
        返回 Driver 的名称.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        返回 Driver 本身的描述.
        """
        pass

    @abstractmethod
    def meta_type(self) -> Type[GHOST_STATE_META]:
        """
        返回 Driver 配置项的 Model, 可以用来获取它配置项的 JSON Schema. 从而可以被 AI 阅读和定义.
        代码本身就是对 AI 的 prompt.
        """
        pass

    @abstractmethod
    def create(self, config: GhostModeConfig) -> GhostMode[GHOST_STATE_META]:
        """
        在上下文中创建一个 GhostMode 的实例.
        """
        pass


class GhostModesManager(ABC):
    """
    用来管理, 构建, 保存所有的 GhostMode.
    由于每个具体的 GhostMode 都是在上下文中动态实例化的, 所以能够持续持有的是 Driver.

    每个 GhostMode 本身就有资源的依赖, 比如 speech 等. 这些资源不一定是它自己独立创建的,
    可能是通过 workspace 里的配置项定义的全局单例.
    所以 GhostDriver 初始化时, 就可以完成对全局资源的检查.
    """

    @abstractmethod
    def drivers(self) -> dict[str, GhostModeDriver]:
        """
        返回所有注册的 drivers.
        """
        pass
