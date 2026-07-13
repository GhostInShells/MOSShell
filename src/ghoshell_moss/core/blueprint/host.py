"""
MOSS Host 层抽象 — 基于环境发现构建的高阶运行时门面.

本模块定义 Host 层的三个核心抽象:

- ``MossHost``: 基于环境约定发现项目能力, 创建运行时的入口.
  不需要环境发现时, 应直接使用 ``ghoshell_moss.core.ctml.new_ctml_shell``
  实例化 ``MOSShell``, 不必经过本层.
- ``MossRuntime``: MossHost 发现产物的统一运行时门面. 屏蔽 shell / interpreter /
  matrix 等底层抽象, 对外暴露 moss_exec / moss_observe / moss_interrupt 这一组
  高阶指令接口, instruction / dynamic / static 三类 messages 的访问, 以及
  shell / matrix / session / project / env / container / logger 直通.
- ``GhostRuntime``: 在 MossRuntime 之上叠加 Ghost 的生命周期编排.
  组合优于伪装 — 不实现 MossRuntime ABC, 通过 ``.moss`` 暴露完整 moss 能力.

以及两个辅助类型:

- ``MossSystemPrompter``: 约定的 instruction 层次命名访问器, 供 system prompt 组装.
- ``LoopHealth`` / ``LoopStatus``: GhostRuntime 三循环健康度的可观测快照.
"""

from typing import Callable, Literal
from typing_extensions import Self, TypedDict
from abc import ABC, abstractmethod
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.mindflow import Mindflow
from ghoshell_moss.core.blueprint.project import Project, HostMode
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.message import Message
from ghoshell_moss.contracts import SystemPrompter
from ghoshell_container import IoCContainer
import logging

__all__ = [
    'MossRuntime', 'MossHost',
    'MossSystemPrompter', 'GhostRuntime', 'LoopHealth', 'LoopStatus',
]


# --- MossSystemPrompter --- #

class MossSystemPrompter(SystemPrompter, ABC):
    """MOSS 约定的 instruction 层次 — 命名访问器.

    四个标准层通过 children() 暴露, 命名方法是对 children key 的便捷包装.
    不排斥开发者通过 with_prompter 添加任意其他 key.
    """

    # 约定的 prompt slots.
    CTML_SLOT = 'ctml'
    PROJECT_SLOT = 'project'
    MODE_SLOT = 'mode'
    MOSS_STATIC_SLOT = 'static'

    def ctml_instruction(self) -> str:
        """当前系统所使用的默认 ctml 提示词. 是 moss 运行基础."""
        return self.child_instruction(self.CTML_SLOT)

    def project_instruction(self) -> str:
        """项目级提示词, 定义在 workspace 的 MOSS.md, 所有模式共享."""
        return self.child_instruction(self.PROJECT_SLOT)

    def mode_instruction(self) -> str:
        """模式级别的提示词. 定义在 workspace 的不同模式中 (MODE.md), 每个模式独有."""
        return self.child_instruction(self.MODE_SLOT)

    def moss_static_instruction(self) -> str:
        """moss 运行时的静态提示词. 来自 shell 构建后的 moss static."""
        return self.child_instruction(self.MOSS_STATIC_SLOT)

    def default_instruction(self) -> str:
        """建议使用的默认提示词组合方式. 供参考."""
        # code as prompt — 提示如何使用.
        return self.linear([
            self.CTML_SLOT,
            self.PROJECT_SLOT,
            self.MODE_SLOT,
            self.MOSS_STATIC_SLOT,
        ])


# --- MossRuntime --- #

class MossRuntime(ABC):
    """MOSS 运行时门面 — 由 MossHost 基于环境发现构建后产出.

    MossRuntime 是模型 / 调用方与 MOSS 交互的统一面, 屏蔽 shell / interpreter /
    matrix 等底层抽象, 对外只暴露三组接口:

    1. 指令面: ``moss_exec`` / ``moss_observe`` / ``moss_interrupt`` — 向运行时输入
       CTML 并观察执行结果.
    2. 信息面: ``moss_instruction`` / ``moss_dynamic_messages`` / ``moss_static_messages``
       / ``moss_refresh_metas`` — 拿到组装 system prompt 所需的全部素材.
    3. 直通面: ``shell`` / ``matrix`` / ``session`` / ``project`` / ``env`` /
       ``container`` / ``logger`` — 让调用方按需穿透到下层做精细操作. 这些直通
       属性都写在 ABC 上, 是有意的 code as prompt — 让读者一眼看清 runtime 由
       什么组成.

    生命周期由 ``__aenter__`` / ``__aexit__`` 守护, 并提供 ``wait_close*`` /
    ``wait_closed*`` / ``close`` 一组方法供异步与同步两种阻塞场景使用.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """返回整个环境自定义的名字."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """返回整个环境默认的自解释描述.

        覆盖逻辑: 创建时传参定义 > Host 环境定义描述.
        """
        ...

    @property
    @abstractmethod
    def mode(self) -> HostMode:
        """当前 runtime 所处的模式."""
        ...

    @abstractmethod
    def moss_instruction(self, with_static: bool = True) -> str:
        """返回所有的 instruction 信息, 可以加入到 agent 的 instruction.

        :param with_static: 是否包含 moss static messages.
        """
        ...

    @abstractmethod
    async def moss_dynamic_messages(self, refresh: bool = True, max_wait: float = 2.0) -> list[Message]:
        """返回 moss 运行时的动态信息.

        仅包含组件的 interface, context messages 等等.
        """
        ...

    @abstractmethod
    async def moss_refresh_metas(self) -> None:
        """刷新 channel metas 缓存, 让 static / dynamic 消息反映最新状态."""
        ...

    @abstractmethod
    def moss_static_messages(self) -> str:
        """返回 moss 运行时的静态信息."""
        ...

    @abstractmethod
    async def moss_exec(
            self,
            logos: str,
            call_soon: bool = True,
            wait_done: bool = True,
    ) -> list[Message]:
        """向 MOSS 的运行时添加新的指令. 通常是 CTML.

        :param logos: 基于 ctml 语法提供的 command 字符串.
        :param call_soon: 为 True 时立刻中断任何运行中的命令, 否则只追加新指令.
        :param wait_done: 为 True 时阻塞到运行结束后, 拿到观察的结果.
        """
        ...

    @abstractmethod
    async def moss_observe(
            self,
            timeout: float | None = None,
            with_dynamic: bool = True,
    ) -> list[Message]:
        """观察等待到 moss 运行状态变更.

        通常包含:

        1. 新的高优消息输入.
        2. 当前有命令在执行, 并且已经执行完或发生了异常.
        3. 等待超时, 仍然返回最新的观察结果.

        :param timeout: 指定一个等待时间, 否则会持续等待到有任何事件为止.
        :param with_dynamic: 观察的结果里是否包含最新的 moss dynamic 信息.
        """
        ...

    @abstractmethod
    async def moss_interrupt(self) -> list[Message]:
        """立刻中断所有运行中的命令, 并且返回中断的情况."""
        ...

    @property
    def system_prompter(self) -> MossSystemPrompter:
        """获取运行时提供的各种提示词声明, 可用于组装."""
        return self.matrix.container.force_fetch(MossSystemPrompter)

    @property
    @abstractmethod
    def shell(self) -> MOSShell[PrimeChannel]:
        """全双工运行时.

        可以在它没启动时做一些操作.
        运行时可以直接通过它的 API 去控制 clear / pause 等操作.
        """
        ...

    @property
    def container(self) -> IoCContainer:
        """直通 matrix.container — IoC 容器的来源."""
        return self.matrix.container

    @property
    def session(self) -> Session:
        """直通 matrix.session — 当前会话."""
        return self.matrix.session

    @property
    def project(self) -> Project:
        """直通 matrix.project — 当前项目."""
        return self.matrix.project

    @property
    @abstractmethod
    def matrix(self) -> Matrix:
        """环境通讯的总线."""
        ...

    @property
    @abstractmethod
    def env(self) -> Environment:
        """环境发现的 api."""
        ...

    @property
    def logger(self) -> logging.Logger:
        """运行时 logger. 未启动时回退到 env.logger, 启动后用 matrix.logger."""
        if not self.is_running():
            return self.env.logger
        return self.matrix.logger

    @abstractmethod
    def is_running(self) -> bool:
        """runtime 是否处于已启动且未关闭的状态."""
        ...

    @abstractmethod
    def close(self) -> None:
        """发送关闭触发信号 (closing)."""
        ...

    @abstractmethod
    def wait_close_sync(self, timeout: float | None = None) -> bool:
        """阻塞等待关闭触发信号 (closing)."""
        ...

    @abstractmethod
    async def wait_close(self) -> None:
        """异步阻塞等待关闭触发信号 (closing)."""
        ...

    @abstractmethod
    def wait_closed_sync(self, timeout: float | None = None) -> bool:
        """阻塞等待关闭完成 (closed)."""
        ...

    @abstractmethod
    async def wait_closed(self) -> None:
        """异步阻塞等待关闭完成 (closed)."""
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        """正式启动."""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """运行结束."""
        ...


# --- LoopHealth / LoopStatus --- #

LoopStatus = Literal["running", "stopped", "not_started"]
"""三循环中单一循环的健康状态."""


class LoopHealth(TypedDict):
    """三循环健康状态快照, 三个 key 始终存在."""

    main: LoopStatus
    articulate: LoopStatus
    action: LoopStatus


# --- GhostRuntime --- #

class GhostRuntime(ABC):
    """编排 MossRuntime + Ghost 的生命周期.

    GhostRuntime 持有 MossRuntime, 在其启动前后完成 Ghost 的注册和生命周期管理.
    不实现 MossRuntime ABC — 组合优于伪装.
    ghost + mindflow 的 main loop 作为内部 async 函数, 通过 matrix.create_task 托管,
    Matrix 退出时自动 cancel.

        ghost_runtime.moss          → MossRuntime (全部 moss 能力)
        ghost_runtime.ghost         → Ghost (运行时实例)
        ghost_runtime.meta          → GhostMeta (启动前即可访问)
        ghost_runtime.mindflow      → Mindflow (运行时三循环中枢)
        ghost_runtime.container     → IoCContainer (快捷路径)
    """

    @property
    @abstractmethod
    def moss(self) -> MossRuntime:
        """持有的 MossRuntime. 调用方通过 .moss 访问全部 Moss 能力."""
        ...

    @property
    @abstractmethod
    def ghost(self) -> Ghost:
        """由 GhostMeta.factory(container) 产出的 Ghost 运行时实例."""
        ...

    @property
    @abstractmethod
    def meta(self) -> GhostMeta:
        """Ghost 的元信息. MossHost.run_ghost 时即已发现, 启动前即可访问."""
        ...

    @property
    @abstractmethod
    def mindflow(self) -> Mindflow:
        """GhostRuntime 持有的 Mindflow. 启动后可用, 未启动时抛出 RuntimeError."""
        ...

    @property
    def container(self) -> IoCContainer:
        """快捷路径: moss.matrix.container."""
        return self.moss.matrix.container

    @abstractmethod
    async def __aenter__(self) -> Self:
        """编排生命周期:

        1. 预注入 ghost providers / nuclei manifests → container.
        2. MossRuntime.__aenter__ (matrix → shell → mindflow).
        3. GhostMeta.factory(container) → ghost.
        4. ghost.__aenter__ (注册 main loop 为 matrix.create_task).
        """
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """逆序清理: ghost.__aexit__ → moss.__aexit__."""
        ...

    @abstractmethod
    def is_paused(self) -> bool:
        """查询 pause 状态."""
        ...

    @abstractmethod
    def pause(self, toggle: bool = True, callback: Callable[[], None] | None = None) -> None:
        """急停 — 幂等, 设值. callback 在级联完成后同步 fire (done 语义).

        callback 必须自行保证线程安全 (可能跨 loop 或跨线程调用).
        """
        ...

    def close(self) -> None:
        """发送关闭信号. 委托给 MossRuntime."""
        self.moss.close()

    @abstractmethod
    def inspect_loop_health(self) -> LoopHealth:
        """返回三循环运行状态快照, 供 REPL / debug 脚本观察. 无副作用.

        三个 key 始终存在.
        """
        ...


# --- MossHost --- #

class MossHost(ABC):
    """MOSS (model-oriented operating system shell) 基于环境发现的高阶抽象.

    如果不需要环境发现, 可以直接使用 ghoshell_moss.core.ctml.new_ctml_shell 来实例化 MOSShell.
    Host 用来管理和发现环境, 从环境中创建 Moss 的一切.

    1. 它屏蔽了 shell/interpreter 等内核模块.
    2. 它管理 Shell 的环境发现与运行. 核心目标包含:
      - 基于约定发现能力.
      - 屏蔽生命周期注册逻辑.
      - 屏蔽底层抽象, 只暴露实体.
      - 将 Shell 的高阶封装作为 MossRuntime 提供.
    3. 它通过 Matrix 解决并行思考网络内的通讯体系.
    4. 它缝合 Ghost 和 Shell, 作为一个独立的认知实体架构:
      - 支持 Ghost in MOSShell 的实现, 不与 Shell 直接耦合.
      - 通过 MossRuntime / GhostRuntime 的分层呼应 Ghost In Shells 理念.

    架构拓扑的设计, 延续自 2019~2020 年的实现.
    https://github.com/thirdgerb/chatbot/blob/dba62e1337559c327d27ec4300366cd890a18ebc/src/Host/IHost.php#L4
    """

    @property
    @abstractmethod
    def env(self) -> Environment:
        """环境变量."""
        ...

    @property
    @abstractmethod
    def project(self) -> Project:
        """当前所见的项目."""
        ...

    @classmethod
    def discover(cls, env: Environment | None = None) -> Self:
        """通过环境发现, 使抽象本身基于约定足以使用.

        # 举例
        async with MossHost.discover().run() as moss:
            ...
        """
        from ghoshell_moss.factory import create_host, create_project
        env = env or Environment.discover()
        project = create_project(env)
        project.bootstrap()
        # 使用反范式定义项目的默认约定.
        return create_host(env, project)

    @abstractmethod
    def run(
            self,
            *,
            run_shell: bool = True,
            name: str | None = None,
            description: str | None = None,
    ) -> MossRuntime:
        """启动并返回 MossRuntime.

        :param run_shell: 为 True 时, 在 runtime aenter 时启动 shell.
        :param name: 指定 moss 名字, 否则使用 meta 中的配置.
        :param description: 指定 moss 描述, 否则使用 meta 中的配置.
        """
        ...

    @abstractmethod
    def run_ghost(
            self,
            ghost: str | GhostMeta,
            *,
            run_shell: bool = True,
    ) -> GhostRuntime:
        """启动并返回 GhostRuntime — 编排 MossRuntime + Ghost 的生命周期.

        :param ghost: ghost 名称 (从 all_ghost_manifests 查找) 或 GhostMeta 实例.
                      传入实例时环境无关, 可用于测试.
        :param run_shell: 传递给 MossRuntime.
        """
        ...
