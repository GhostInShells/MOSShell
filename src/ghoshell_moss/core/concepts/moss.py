from abc import ABC, abstractmethod
from typing import Literal, Callable, Coroutine, Iterable
from typing_extensions import Self

from ghoshell_moss.core.blueprint import MutableChannel
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.core.concepts.topic import TopicModel, TopicService
from ghoshell_container import IoCContainer
from ghoshell_moss.message import Message
from pydantic import BaseModel, Field
from pydantic_ai import ToolReturn, UserContent
from enum import Enum
import asyncio

__all__ = [
    'Priority', 'PriorityLevel', 'IgnorePolicy',
    'InputTopic', 'InterruptTopic',
    'Snapshot',
    'IdleHook', 'RespondHook', 'Respond',
    'MOSS', 'MOSSRuntime', 'MOSSToolSet',
]

PriorityLevel = Literal['DEBUG', 'INFO', 'NOTICE', 'WARNING', 'ERROR', 'CRITICAL', 'FATAL', 'DEFAULT']
""" scope of the input messages level, less and equal then """

IgnorePolicy = Literal['drop', 'buffer', 'never']
""" how to handle ignore messages"""


class Priority(int, Enum):
    """
    moss 架构关注的运行信息的默认优先级.
    高于优先级, 会中断 MOSS 运行的 Loop.
    """
    DEBUG = -1
    INFO = 0  # 输入信息作为上下文的一部分, 下一轮思考关键帧时才运行.
    NOTICE = 1  # 默认的输入级别, AI 需要立刻响应.
    WARNING = 2  # 更高的输入级别.
    ERROR = 3  # 更高的输入级别
    CRITICAL = 4  # 更高的输入级别.
    FATAL = 5  # 任何时候都需要响应, 除非正在处理的级别小于这个级别.

    @classmethod
    def new(cls, level: str) -> Self:
        if level in cls.__members__:
            return cls.__members__[level]
        return cls.INFO

    def new_inputs(self, *messages: Message) -> "InputTopic":
        return InputTopic(priority=self.value).with_message(*messages)


class InputTopic(TopicModel):
    """
    MOSS 的输入信息. 可以直接通过 Channel 输入.
    """
    priority: int = Field(
        default=Priority.INFO.value,
        description="输入信息的优先级, 决定是否中断当前运行状态",
    )
    incomplete: dict[str, Message] = Field(
        default_factory=dict,
        description="incomplete messages"
    )
    completed: list[Message] = Field(
        default_factory=list,
        description="completed messages"
    )

    @classmethod
    def new(cls, *messages: Message, priority: int) -> Self:
        return cls(priority=priority).with_message(*messages)

    def with_message(self, *messages: Message) -> Self:
        for msg in messages:
            if msg.is_empty():
                continue
            if msg.meta.incomplete:
                self.incomplete[msg.meta.id] = msg
            else:
                self.completed.append(msg)
        return self

    @classmethod
    def topic_type(cls) -> str:
        return 'moss/InputsTopic'

    @classmethod
    def default_topic_name(cls) -> str:
        return 'moss/inputs'


class MessagesTopic(TopicModel):
    """
    inputs/outputs messages from moss runtime, listen to it for rendering messages
    """
    message: list[Message] = Field(
        description="moss output messages"
    )

    @classmethod
    def topic_type(cls) -> str:
        return 'moss/OutputTopic'

    @classmethod
    def default_topic_name(cls) -> str:
        return 'moss/output'


class InterruptTopic(TopicModel):
    """
    interrupt the moss loop by allmeans
    """
    message: Message | None = Field(
        default=None,
        description="moss interrupt message"
    )

    @classmethod
    def topic_type(cls) -> str:
        return 'moss/InterruptTopic'

    @classmethod
    def default_topic_name(cls) -> str:
        return 'moss/interrupt'


State = Literal['created', 'idle', 'responding', 'executing', 'closed']


# moss 运行时 status 的设计. 坚决不提供底层逻辑.
class Snapshot(BaseModel):
    """
    MOSS 的运行时状态, 可以对 AI 进行呈现.
    """
    cursor: int = Field(
        default=0,
        description='moss snapshot cursor position'
    )
    state: State = Field(
        default='created',
        description='runtime state of the MOSS'
    )
    focus_level: int = Field(
        default=Priority.NOTICE.value,
        description='focus level of the MOSS',
    )
    ignore_method: IgnorePolicy = Field(
        default='buffer',
        description='how to handle ignored messages'
    )

    # -- dynamic runtime messages, always there but changed during time -- #

    runtime_status: list[Message] = Field(
        default_factory=list,
        description="moss current status, include executing/pending/canceled and cleared"
    )

    context: list[Message] = Field(
        default_factory=list,
        description="context messages that can be ignore in history turns"
    )

    incomplete_inputs: list[Message] = Field(
        default_factory=list,
        description="incomplete inputs messages, as part of the context"
    )

    # -- popped after ack, buffering if not handle -- #

    executed: list[Message] = Field(
        default_factory=list,
        description="executed command tasks messages. cleared after each pop"
    )

    inputs: list[Message] = Field(
        default_factory=list,
        description="inputs messages that should be handled"
    )

    def to_messages(
            self,
    ) -> Iterable[Message]:
        yield from self.executed
        yield from self.runtime_status
        yield from self.context
        yield from self.incomplete_inputs
        yield from self.inputs

    def to_user_contents(
            self,
            with_meta: bool = True,
    ) -> Iterable[UserContent]:
        for message in self.to_messages():
            yield from message.as_contents(with_meta=with_meta)


class Respond(ABC):
    """
    在 moss 架构中创建一个 Respond 对象, 用于接收一个完整的运行时信息.
    Respond 正式启动时, 会进入 responding 状态.
    """

    @abstractmethod
    def snapshot(self) -> Snapshot:
        """
        the snapshot before responding.
        """
        pass

    def add(self, token: str) -> None:
        """
        添加待执行的 token.
        :raise InterpreterError: 如果输入信息因为编译问题, 或执行问题而中断, add 时也会抛出异常.
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


RespondHook = Callable[[Respond], Coroutine[None, None, None]]
"""当 MOSS 运行状态被 Interrupt 或有输入时, 执行 loop"""

IdleHook = Callable[[Snapshot], Coroutine[None, None, None]]
"""当 MOSS 没有任何输入, 执行也完毕后, 执行 Idle"""


# 通过下面的 MOSS 实例生成的运行时对象.
# 核心设计思路是配置和运行时分离, 用来拆分关注点和防蠢.
# 简单来说由于 MOSShell 是一个全双工的运行时, 它的物理存在决定了必须是运行时单例.
# 所以需要必要的锁解决资源冲突, 以及提供清晰的 API 用于集成.
class MOSSRuntime(ABC):
    """
    MOSS 的运行时单例.
    """

    # -- 面向 Agent 架构提供的系统函数 -- #

    @property
    @abstractmethod
    def shell(self) -> MOSShell:
        pass

    @abstractmethod
    def meta_instruction(self) -> str:
        """
        return moss meta instruction with its protocol (such as CTML)
        shall put top to other messages of an agent
        """
        pass

    @abstractmethod
    def instructions(self) -> str:
        """
        all the instruction of MOSS channels
        could put it under the meta instruction or other instructions.
        """
        pass

    @abstractmethod
    def snapshot(
            self,
            *,
            context: bool = True,
            inputs: bool = True,
    ) -> Snapshot:
        """
        return snapshot immediately.
        if cursor is not change, always return the newest snapshot.
        if not ack snapshot, the snapshot will not change
        :param context: with context messages
        :param inputs: with popped input messages
        """
        pass

    @abstractmethod
    async def ack_snapshot(self, cursor: int | Snapshot) -> None:
        """
        ack snapshot
        """
        pass

    async def pop_snapshot(
            self,
            *,
            context: bool = True,
            inputs: bool = True,
    ) -> Snapshot:
        """
        generate new snapshot and make sure ack it.
        """
        snapshot = self.snapshot(context=context, inputs=inputs)
        try:
            return snapshot
        finally:
            await self.ack_snapshot(snapshot)

    def add_inputs(
            self,
            *inputs: Message,
            priority: int = Priority.INFO.value,
            creator: str = '',
    ) -> None:
        """
        向运行时提交新的输入. 会立刻按规则影响运行时状态.
        """
        topic = InputTopic.new(*inputs, priority=priority)
        # set the name
        topic.meta.creator = creator or self.shell.name
        self.add_input_topic(topic)

    @abstractmethod
    def add_input_topic(self, topic: InputTopic) -> None:
        """
        just for reflecting the key concepts of topic / InputTopic
        """
        pass

    @property
    def topics(self) -> TopicService:
        """
        获取 topic 实例. 可以在整个 MOSS 体系内完成广播通讯.
        """
        return self.shell.topics()

    @abstractmethod
    async def create_task(self, cor: Coroutine) -> asyncio.Task:
        """
        创建一个 task, 被 MOSS 自身的生命周期所管理.
        可以用在各种技术实现内部.
        """
        pass

    @abstractmethod
    async def refresh_metas(self) -> None:
        pass

    # --- 面向 AI 暴露的控制函数 --- #

    async def observe(self, timeout: float | None = None) -> None:
        """
        进入观察状态, 等待最新的中断行为.
        """
        wait_first_done = []
        try:
            wait_first_done.append(self.create_task(self.wait_inputted()))
            wait_first_done.append(self.create_task(self.wait_interrupted()))
            wait_first_done.append(self.create_task(self.wait_closed()))
            done, pending = await asyncio.wait(
                wait_first_done,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
        except asyncio.CancelledError:
            pass
        finally:
            if len(wait_first_done) > 0:
                for t in wait_first_done:
                    if not t.done():
                        t.cancel()

    @abstractmethod
    async def call_soon(self, commands: str) -> None:
        """
        添加新的输出 commands tokens 到 moss 的运行时.
        commands 遵循 moss 的运行规则, 比如 CTML
        call soon 会立刻中断正在运行中的状态.
        """
        pass

    @abstractmethod
    async def add(self, commands: str) -> None:
        """
        在已经运行的状态中, 追加新的指令. 不会中断已经运行的状态.
        """
        pass

    @abstractmethod
    async def focus(self, level: PriorityLevel, ignore_method: IgnorePolicy = 'buffer') -> None:
        """
        立刻设置 focus 级别.
        :param level: if inputs level > current level, will break the loop
        :param ignore_method: if inputs level <= current level when looping, handle it with the ignore method.
        """
        pass

    @abstractmethod
    async def interrupt(self) -> None:
        """
        立刻终止所有的运行状态.
        """
        pass

    @abstractmethod
    async def wait_compiled(self) -> None:
        """
        等待到最新的指令编译完成.
        """
        pass

    @abstractmethod
    async def wait_idle(self) -> None:
        """
        等待到当前的指令运行结束. 如果没有结束, 立刻返回.
        """
        pass

    @abstractmethod
    async def wait_inputted(self) -> None:
        """
        等待接受到最新的输入.
        """
        pass

    @abstractmethod
    async def wait_interrupted(self) -> None:
        """
        等待运行逻辑被中断. 中断的原因可能有:
        1. 输入了错误的指令.
        2. 等待到了高优的输入, 打断了运行.
        3. 运行时的关键异常, 中断了运行.
        """
        pass

    # --- 生命周期治理 --- #

    @abstractmethod
    async def wait_closed(self) -> None:
        """
        阻塞到系统运行结束.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        发送中断指令, 让 Runtime 进入到 wait_closed 从而退出.
        """
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


class MOSSToolSet(ABC):
    """
    将 MOSS runtime 包装成 tools, 从而可以被作为工具提供给别的框架.
    不过需要目标框架自行兼容 Pydantic AI 的消息协议.
    """

    @property
    @abstractmethod
    def runtime(self) -> MOSSRuntime:
        pass

    def meta_instruction(self) -> str:
        """
        return MOSS meta instruction about what it is.
        """
        return self.runtime.shell.meta_instruction()

    async def moss_instructions(self) -> ToolReturn:
        """
        understand how to use MOSS Runtime.
        """
        instruction_messages = self.runtime.shell.channel_instructions()
        messages = []
        for channel_name, channel_instruction_messages in instruction_messages.items():
            messages.extend(channel_instruction_messages)
        tool_return = ToolReturn(return_value=None, content=None)
        if len(messages) > 0:
            content = []
            for msg in messages:
                content.extend(msg.as_contents())
            tool_return.content = content
        return tool_return

    async def moss_context_messages(self) -> ToolReturn:
        """
        :returns: the context messages of all the channels from MOSS Runtime.
        """
        context_messages = self.runtime.shell.channel_context_messages()
        messages = []
        for channel_name, channel_context_messages in context_messages.items():
            messages.extend(channel_context_messages)
        tool_return = ToolReturn(return_value=None, content=None)
        if len(messages) > 0:
            content = []
            for msg in messages:
                content.extend(msg.as_contents())
            tool_return.content = content
        return tool_return

    async def moss_add(self, commands: str) -> ToolReturn:
        """
        add new commands in MOSS protocol into runtime.
        MOSS Runtime will compile the commands and then return the status immediately.
        :returns: status of the MOSS runtime.
        """
        await self.runtime.add(commands)
        snapshot = await self.runtime.pop_snapshot(inputs=False, context=False)
        return self.snapshot_to_tool_return(snapshot)

    async def moss_call_soon(self, commands: str) -> ToolReturn:
        """
        clear the moss runtime and add new commands in MOSS protocol soon.
        MOSS Runtime will compile the commands then return the status immediately.
        :returns: status of the MOSS runtime.
        """
        await self.runtime.call_soon(commands)
        snapshot = await self.runtime.pop_snapshot(inputs=False, context=False)
        return self.snapshot_to_tool_return(snapshot)

    async def moss_interrupt(
            self,
    ) -> ToolReturn:
        """
        interrupt the execution of MOSS runtime.
        :returns: status of the MOSS runtime. if observe is True, returns the inputs and context messages with it
        """
        await self.runtime.interrupt()
        snapshot = await self.runtime.pop_snapshot(inputs=True, context=True)
        return self.snapshot_to_tool_return(snapshot)

    async def moss_observe(
            self,
            timeout: float | None = None,
    ) -> ToolReturn:
        """
        observe the moss runtime, return when:
        1. new messages that reach the priority level received.
        2. if commands are executing, return when they are executed.
        3. any execution fatal error or command compiling error occurs.
        :returns: context messages, inputs and status of the MOSS runtime.
        """
        await self.runtime.observe(timeout)
        snapshot = await self.runtime.pop_snapshot(context=True, inputs=True)
        return self.snapshot_to_tool_return(snapshot)

    async def moss_focus(
            self,
            level: PriorityLevel,
            policy: IgnorePolicy = 'buffer',
    ) -> None:
        """
        managing MOSS Runtime focus level and policy to handle input messages.
        :param level: you can raise the level to prevent interruption.
        :param policy: you can change the policy to handle any inputs that priority less than the level
        """
        await self.runtime.focus(level, policy)

    @staticmethod
    def snapshot_to_tool_return(
            snapshot: Snapshot,
            *,
            with_meta: bool = True,
    ) -> ToolReturn:
        return ToolReturn(
            return_value=None,
            content=list(snapshot.to_user_contents(with_meta=with_meta)),
        )

    async def __aenter__(self) -> Self:
        await self.runtime.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.runtime.__aexit__(exc_type, exc_val, exc_tb)


# MOSShell 的高级抽象封装, 目的是:
# 1. 屏蔽底层 shell / interpreter 的具体实现.
# 2. 在 Shell 的上层, 针对全双工思考范式, 提供有状态服务. 支持模型的 interactive reasoning.
# 3. 支持以工具的形式接入现有的 Agent 生态, 比如用 mcp 的形式接入.
# 4. 支持 pydantic ai 实现的双工 Agent. 将流式控制范式推进到流式 思考-观察-行动 范式.
#
# 坚持 Facade 思路, 不暴露任何对用户没有用的 API. 降低用户的心智复杂度.
# 用户可以自己读源码了解底层的实现与封装.
class MOSS(ABC):
    """
    MOSS 架构的高阶 interface.
    为 MOSShell 提供和 Agent / MCP / Tool 的集成方式.
    """

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        moss 启动时获取到的全局 IoC 容器.
        可以作为 Pydantic AI 的 Context.deps 使用.
        """
        pass

    @abstractmethod
    def run(self) -> MOSSRuntime:
        """
        完成初始化工程, 返回一个可以使用的 Runtime/
        """
        pass

    @abstractmethod
    def run_as_toolset(self) -> MOSSToolSet:
        """
        将 Runtime 包装成 ToolSet
        可以被注册成 agent tool.
        """
        pass

    @property
    @abstractmethod
    def shell(self) -> MOSShell:
        """
        MOSS 定义阶段的 Shell.
        可以用来注册新的 channel / command 等定制化工作.
        """
        pass

    @property
    def main(self) -> MutableChannel:
        """
        return main channel, main purpose to be able to reflect the current module and return prompt of key classes
        """
        return self.shell.main_channel

    @abstractmethod
    def on_respond(self, hook: RespondHook) -> Self:
        """
        注册 Loop Hook. 为了让 MOSSRuntime 能够同时承载一个 Agent 的生命周期.
        当 MOSS 运行时拿到高优输入/Interrupt/运行时异常时, 已有的 respond 会中断
        会基于瞬时上下文, 提供一个新的 respond.

        respond 本身用于解决流式输出时的 MOSS 指令.
        和 Tool 等不同, respond 可以将 reasoning 或 final answer 的 token 直接按 moss 规则 (CTML) 执行.
        """
        pass

    @abstractmethod
    def on_idle(self, hook: IdleHook) -> Self:
        """
        注册 Idle Hook. 为了让 MOSSRuntime 能够同时承载一个 Agent 的生命周期.
        当一次 Agent 的 respond 结束后, 就进入 Idle 生命周期. 可以用工程方式定义它的行为逻辑.
        最简单的自驱就是在 idle 时就立刻让它思考.
        """
        pass
