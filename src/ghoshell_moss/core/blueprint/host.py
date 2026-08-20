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
from concurrent.futures import Future
from dataclasses import dataclass
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.session import Session, OutputItem
from ghoshell_moss.core.blueprint.mindflow import Mindflow, Signal
from ghoshell_moss.core.blueprint.project import Project, HostMode
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.message import Message
from ghoshell_moss.contracts import SystemPrompter
from ghoshell_container import IoCContainer
from .shell_trajectory import MShellTrajectory
import logging

__all__ = [
    'IShellRuntime', 'IHost',
    'MossSystemPrompter', 'IGhostRuntime', 'LoopHealth', 'LoopStatus',
    'SafeMode', 'PendingApproval', 'Verdict',
]


# --- MossSystemPrompter --- #

class MossSystemPrompter(SystemPrompter, ABC):
    """MOSS 约定的 instruction 层次 — 命名访问器.
    通过组装的方式, 从环境 (workspace) 中生成 moss 的系统提示词. 通常包含三部分:
    1. Logos: 提示模型输出的 text chunks 用何种方式 (比如 ctml) 驱动它所控制的躯体.
    2. Project: moss workspace 下 MOSS.md 里定义的提示词, 用来告知模型处在 moss 系统内部.
    3. Mode:
    """

    # 约定的 prompt slots.
    LOGOS_SLOT = 'logos'
    PROJECT_SLOT = 'project'
    MODE_SLOT = 'mode'
    MOSS_STATIC_SLOT = 'static'

    def logos_meta_instruction(self) -> str:
        """当前系统所使用的 Logos 语法本身的提示词(通常是 ctml). 是 moss 运行基础."""
        return self.child_instruction(self.LOGOS_SLOT)

    def project_instruction(self) -> str:
        """项目级提示词, 定义在 workspace 的 MOSS.md, 所有模式共享."""
        return self.child_instruction(self.PROJECT_SLOT)

    def mode_instruction(self) -> str:
        """模式级别的提示词. 定义在 workspace 的不同模式中 (MODE.md), 每个模式独有."""
        return self.child_instruction(self.MODE_SLOT)


    def moss_static_instruction(self) -> str:
        """moss 运行时的静态提示词. 来自 shell 构建后的 moss static."""
        return self.child_instruction(self.MOSS_STATIC_SLOT)

    def base_instruction(self) -> str:
        """由 moss mode 决定的基础 instruction, 和运行时 channel 的组装情况无关."""
        return self.linear([
            self.LOGOS_SLOT,  # Logos 使用策略的提示词.
            self.PROJECT_SLOT,  # moss 环境的根提示词.
            self.MODE_SLOT,  # 每个模式下独有的提示词.
        ])

    def full_instruction(self) -> str:
        """
        Moss StaticMessages + DynamicMessages 组合上下文时, 使用的 instruction.
        在 base instruction 之外, 增加了 moss static 讯息, 呈现所有 Channel 不变部分的讯息.
        然后模型下每一帧请求前, 再提供 moss channel 树动态部分的讯息. 这部分信息不进入对话历史.

        对话历史形如 (full_instruction + turns[without dynamic] + dynamic + input.
        依赖 LLM Agent 有能力在每一轮请求时, 将上一轮历史消息中的动态部分拿掉.
        """
        return self.linear([
            self.LOGOS_SLOT,
            self.PROJECT_SLOT,
            self.MODE_SLOT,
            self.MOSS_STATIC_SLOT,
        ])


# --- MossRuntime --- #

class IShellRuntime(ABC):
    """MOSShell 运行时整体
    完成 matrix / shell 等所有模块装线, 提供统一的交互界面.
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
    def instruction(self, with_static: bool = True) -> str:
        """返回所有的 instruction 信息, 可以加入到 agent 的 instruction.

        :param with_static: 是否包含 moss static messages.
        """
        ...

    @abstractmethod
    def static_messages(self) -> str:
        """返回 Shell 包含 Channel 体系在运行时不变的信息. 合适无 cache 的上下文组装.  和 dynamic messages 组成完成讯息."""
        ...

    @abstractmethod
    async def dynamic_messages(self, refresh: bool = True, max_wait: float = 2.0) -> list[Message]:
        """返回 Shell 运行时的变化信息. 适合无 cache 的上下文组装. 每轮变化, 和 static messages 组成完整讯息. """
        ...

    @abstractmethod
    async def refresh_metas(self) -> None:
        """刷新 channel metas 缓存, 让 static / dynamic 消息反映最新状态."""
        ...

    def trajectory(self) -> MShellTrajectory:
        """
        创建 shell 运行时的轨迹讯息. 这是针对 LLM Agent 基于 append only 治理上下文时提供的策略.
        用前缀缓存命中率, 代替动态上下文治理策略.

        通过两部分更新 Shell Channel 树的上下文变化:
        1. trajectory.epoch_start_point: 每次重建当前运行状态时, 返回全量信息. 通常在新上下文, 或 compact 之后刷新 epoch.
        2. trajectory.pop_frame: 适合在多轮交互的每一帧返回 delta (shell 运行时返回值 + shell 状态 + facade 变更)

        使用 trajectory 的场景不需要使用 static + dynamic 方式.
        所有的 frame delta 都应该进入历史, 让 cache 命中.
        需要 async with 的方式启动, 伪代码如下:

        >>> async def append_only_agent_loop(trajectory: MShellTrajectory, llm_agent):
        >>>     async with trajectory:
        >>>         async for epoch in llm_agent:
        >>>             llm_agent.inject(trajectory.epoch_start_point(refresh=True))  # 注入新上下文.
        >>>             async for step_inputs in epoch:  # 周期性拿到请求.
        >>>                 llm_agent.inject(trajectory.pop_frame().project())  # 注入每一帧的上下文.
        >>>                 async for logos in llm_agent.run_step(step_inputs)
        >>>                     yield logos   # 返回对躯体的控制.
        """
        return MShellTrajectory(self.shell)

    @abstractmethod
    async def exec_logos(
            self,
            logos: str,
            call_soon: bool = True,
            wait_done: bool = True,
    ) -> list[Message]:
        """适合函数化地执行 logos. 适合调试, 正常的 logos 用法应该是流式的.
        :param logos: 驱动躯体运行的字符串.
        :param call_soon: 为 True 时立刻中断任何运行中的命令. 为 False 时将 logos 追加到执行序列后.
        :param wait_done: 为 True 时阻塞到所有命令执行结束后.
        :return: logos 的运行结果, 不包含 Shell 的状态.
        """
        ...

    @abstractmethod
    async def observe(
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
    async def interrupt(self) -> list[Message]:
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
        """运行时 logger. 未启动时回退到 project.logger, 启动后用 matrix.logger."""
        if not self.is_running():
            return self.project.logger
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

    def run_until_closed(self) -> None:
        """同步阻塞入口: 管理完整 MossRuntime 生命周期直到 close() 被调用.

        对标 Matrix.run — code as prompt: 调用者无需手写 loop / AsyncExitStack /
        cancel+gather. 内部 = uvloop + runtime.__aenter__ → wait_close → runtime.__aexit__
        + graceful teardown.

        注册 SIGINT handler → self.close() → _closing_event → wait_close() 自然唤醒
        → async with 退出 → __aexit__ teardown. 不走 asyncio.run 的暴力取消,
        __aexit__ 保证跑完.

        适用场景: 命令行无交互运行 (moss-shell log 等).
        """
        import asyncio
        import signal
        import sys

        try:
            import uvloop
        except ImportError:
            uvloop = None

        if sys.platform == 'win32':
            loop = asyncio.new_event_loop()
        elif uvloop is not None:
            loop = uvloop.new_event_loop()
        else:
            loop = asyncio.new_event_loop()

        async def _run() -> None:
            async with self:
                await self.wait_close()

        prev_handler = signal.signal(
            signal.SIGINT,
            lambda signum, frame: self.close(),
        )
        try:
            loop.run_until_complete(_run())
        finally:
            signal.signal(signal.SIGINT, prev_handler)
            loop.close()

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


# --- SafeMode — articulator→action 之间的人工审批闸口 --- #


class PendingApproval(TypedDict):
    """SafeMode 挂起的审批快照, TUI 拉取用. 只读语义, 请勿修改字段.

    - ``uuid``: 本次审批的唯一标识. TUI 提交 approve/reject 时必须回传, 用于 stale 比对.
    - ``logos``: articulator 完整产出的文本, 作为裁决依据. 用户实时读到的流即此文本.
    """

    uuid: str
    logos: str


@dataclass(frozen=True)
class Verdict:
    """SafeMode 裁决结果 — 三态标签, 拦截点据此分派.

    - ``approved``: 通过. 拦截点回放 buffered logos → ``articulator.send_nowait``;
      若 ``message`` 非空 (approve-with-note), 再 ``raise_observe`` 使人类补充意见
      作为下一帧内观.
    - ``rejected``: 否决. 拦截点走 ``articulator.raise_observe(message)``, message
      作为否决理由随下一帧 Reaction 进 moment.
    - ``cancelled``: 撤销. abort 到来时拦截点 finally 幂等 cancel; TUI 不主动产生.

    ``message`` 语义在两态间共享 (决策 12/13): 都走 attention 内观通道, 不是外视 outcome.
    """

    kind: Literal['approved', 'rejected', 'cancelled']
    message: str = ''


class SafeMode(ABC):
    """GhostRuntime 的人工审批闸口 — 懒加载单例, 从 ``GhostRuntime.safe_mode()`` 取.

    局部治理: 只闸 articulator 生成的 logos, 不闸输入 (输入通断是 pause 的职责).
    ``moment.command_logos`` (impulse 反射弧) 绕行 gate. 详见 ghost-runtime-safemode FEATURE.

    生命周期:
      - ``enabled``: 开关. 只影响下一轮 articulation 的模式判定 (生成开始时判定一次),
        不动在途逻辑. 已挂起的 pending 继续等人裁决完.
      - ``pending``: 当前挂起的审批. 审批期间 articulate loop 串行阻塞,
        任意时刻至多一个 pending, 因此 uuid 用于比对而非选择.
    """

    @abstractmethod
    def is_enabled(self) -> bool:
        """开关状态."""
        ...

    @abstractmethod
    def set_enabled(self, enabled: bool = True) -> bool:
        """翻转开关. 返回 True = 状态变更, False = 幂等无变化.

        只影响下一轮 articulation 的模式判定; 已在等待的 pending 不受影响.
        """
        ...

    @abstractmethod
    def pending(self) -> PendingApproval | None:
        """当前挂起的审批. 无 pending 时返回 None. 只读快照, 供 TUI toolbar 拉取."""
        ...

    @abstractmethod
    def submit(self, logos: str) -> Future[Verdict]:
        """拦截侧提交审批 — 由 ``_run_articulator`` 在开启态下调用.

        生成 uuid, 写入 pending, 触发 ``on_pending_changed`` 回调.
        返回的 Future 由 TUI 侧 ``approve``/``reject`` 或内部 ``cancel_current`` 结算.
        articulate loop 通过 ``asyncio.wrap_future(...)`` await; abort 时挂靠
        ``articulator.create_task(...)`` 联动取消.

        任意时刻至多一个 pending — 上一个未结算时禁止再次 submit.
        """
        ...

    @abstractmethod
    def approve(self, uuid: str, note: str = '') -> bool:
        """通过 uuid 匹配的 pending. 返回 True = 生效, False = uuid 不匹配 (stale, no-op).

        决策 8: stale 静默 no-op, 绝不自动顺延到下一帧.
        决策 12: ``note`` 非空时, 拦截点在回放 logos 之后 ``raise_observe(note)``,
        使人类补充的意见作为下一帧内观进入 ghost 感知; 保持默认参数使无 note
        的清洁通过路径不变.
        """
        ...

    @abstractmethod
    def reject(self, uuid: str, reason: str) -> bool:
        """否决 uuid 匹配的 pending. 返回 True = 生效, False = uuid 不匹配 (stale, no-op).

        否决走 ``articulator.raise_observe(reason)`` 反馈, 不 abort attention.
        reason 会随下一帧 Reaction 进 moment, ghost 感知后重新 articulate.
        """
        ...

    @abstractmethod
    def cancel_current(self) -> bool:
        """撤销当前 pending, 结算为 ``cancelled``. 返回 True = 生效, False = 无 pending.

        供拦截点 ``finally: safe_mode.cancel_current()`` 幂等收尾; TUI 不调用.
        """
        ...

    @abstractmethod
    def on_pending_changed(self, callback: Callable[[], None] | None) -> None:
        """注册 pending 变更回调 (进入 pending / 离开 pending 都触发).

        callback 无参; 消费方通过 ``pending()`` 拉取最新快照.
        传入 None 清除回调. callback 必须自行保证线程安全 —
        pending 变更可能来自 articulate loop 或 TUI 线程.
        镜像 pause 的 callback 契约, 不引入队列 / ThreadSafeEvent.
        """
        ...


# --- GhostRuntime --- #

class IGhostRuntime(ABC):
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
    def moss(self) -> IShellRuntime:
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
    def is_running(self) -> bool:
        """是否已完成启动 (__aenter__ 返回). 启动前 False, 启动后 True."""
        ...

    @abstractmethod
    def on_output(self, callback: Callable[[OutputItem], None]) -> None:
        """注册 output 监听 — 生命周期无关, 启动前可注册.

        启动前注册: 缓冲, __aenter__ (matrix 就绪后) 优先装线到 session.
        启动后注册: 直接挂到 session.
        """
        ...

    @abstractmethod
    def on_signal(self, callback: Callable[[Signal], None]) -> None:
        """注册 signal 监听 — 生命周期无关, 启动前可注册. 语义同 on_output."""
        ...

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

    @abstractmethod
    def safe_mode(self) -> SafeMode:
        """SafeMode 懒加载单例入口 — articulator→action 之间的人工审批闸口.

        每个 GhostRuntime 实例持有唯一一个 SafeMode; 未开启时零开销.
        首次调用创建, 后续调用返回同一实例. 详见 ``SafeMode`` ABC 与
        ghost-runtime-safemode FEATURE.
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

class IHost(ABC):
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
    ) -> IShellRuntime:
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
    ) -> IGhostRuntime:
        """启动并返回 GhostRuntime — 编排 MossRuntime + Ghost 的生命周期.

        :param ghost: ghost 名称 (从 all_ghost_manifests 查找) 或 GhostMeta 实例.
                      传入实例时环境无关, 可用于测试.
        :param run_shell: 传递给 MossRuntime.
        """
        ...
