from typing import Callable, Coroutine, Protocol, Iterable, AsyncIterator, Optional, Any
from typing_extensions import Self
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, AwareDatetime, ValidationError

from ghoshell_moss.message import Message
from ghoshell_common.helpers import uuid
from PIL.Image import Image
import datetime
import dateutil
import time
import asyncio
import dataclasses

Priority = int
SignalName = str

DEBUG: Priority = -1
INFO: Priority = 0
NOTICE: Priority = 1
WARNING: Priority = 2
ERROR: Priority = 3
CRITICAL: Priority = 4
FATAL: Priority = 5


class Signal(BaseModel):
    name: SignalName = Field(
        description="the signal name, if not match any mind pulse, the signal will be ignore",
    )
    id: str = Field(
        default_factory=uuid,
        description="unique identifier of the signal",
    )
    trace_id: str = Field(
        default='',
        description="the trace id of the signal",
    )
    complete: bool = Field(
        default=True,
        description="whether the signal complete or not",
    )
    max_hop: int = Field(
        default=1,
        description="maximum hop number, 为 0 不传播. ",
    )
    issuer: str = Field(
        default="",
        description="the issuer of the signal, 不需要显示传递, 实际链路发布时会添加.",
    )
    priority: Priority = Field(
        default=0,
        description="信号的优先级, 越大优先级越高. 用于做抢占式调度. 来自边缘系统的输入本身应包含第一轮优先级"
    )
    strength: int = Field(
        default=0,
        description="信号的强度",

    )
    description: str = Field(
        default='',
        description="short description of the signal",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="被处理过的消息体.",
    )
    prompt: str = Field(
        default='',
        description="the prompt to handle the signal",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="meta data of the signal follow the protocol of the name",
    )
    stale_timeout: float = Field(
        default=0,
    )
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.datetime.now(dateutil.tz.gettz()),
    )

    @classmethod
    def new(
            cls,
            name: SignalName,
            *messages: Message,
            priority: int = 0,
            description: str = '',
            metadata: dict[str, Any] | None = None,
            stale_timeout: float = 0,
    ) -> Self:
        return cls(
            name=name,
            messages=list(messages),
            priority=priority,
            description=description,
            metadata=metadata or {},
            stale_timeout=stale_timeout,
        )

    def is_stale(self) -> bool:
        if self.stale_timeout <= 0:
            return False
        delta = time.time() - self.created_at.timestamp()
        return delta > self.stale_timeout

    def to_json(self) -> str:
        return self.model_dump_json(indent=0, exclude_none=True, exclude_defaults=True, ensure_ascii=False)


class SignalMeta(BaseModel, ABC):
    """
    to define a signal protocol.
    """

    @classmethod
    @abstractmethod
    def signal_name(cls) -> SignalName:
        pass

    @classmethod
    def priority(cls) -> Priority:
        return INFO

    @classmethod
    def from_signal(cls, signal: Signal) -> Self | None:
        if cls.signal_name() != signal.name:
            return None
        try:
            metadata = signal.metadata
            return cls.model_validate(metadata)
        except ValidationError:
            return None

    def to_signal(
            self,
            *messages: Message | str | Image,
            description: str = '',
            stale_timeout: float = 0,
            priority: int | None = None,
    ) -> Signal:
        name = self.signal_name()
        wrapped_messages = []
        for msg in messages:
            if isinstance(msg, Image):
                wrapped_messages.append(Message.new().with_content(msg))
            elif isinstance(msg, str):
                wrapped_messages.append(Message.new().with_content(msg))
            elif isinstance(msg, Message):
                wrapped_messages.append(msg)
        priority = self.priority() if priority is None else priority
        return Signal(
            name=name,
            messages=wrapped_messages,
            metadata=self.model_dump(exclude_defaults=True, exclude_none=True),
            description=description,
            stale_timeout=stale_timeout,
            priority=priority,
        )


class InputSignal(SignalMeta):
    """
    basic input.
    """

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'moss/input'


class Impulse(BaseModel):
    """
    the impulse that raise mindflow attention
    """
    id: str = Field(
        default_factory=uuid,
        description="the impulse id",
    )
    source: str = Field(
        default='',
        description="the nucleus source name",
    )
    trace_id: str = Field(
        default='',
        description="the impulse trace id, 向上溯源.",
    )
    priority: Priority = Field(
        default=0,
        description="the impulse priority",
    )
    strength: int = Field(
        default=0,
        description="the impulse 初始强度, 在 attention 中设计强度计算曲线用来解决相同优先级打断机制.",
    )
    on_logos_start: str = Field(
        default='',
        description="the start logos insert into the stream",
    )
    complete: bool = Field(
        default=True,
        description="if the impulse is complete, or just occupy the attention until complete impulse of the same id",
    )
    description: str = Field(
        default='',
        description="the impulse short description",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="the messages of the impulse. if empty, no need to think",
    )
    prompt: str = Field(
        default='',
        description="the prompt to handle the impulse",
    )
    on_logos_done: str = Field(
        default='',
        description="the done logos append to the stream",
    )
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.datetime.now(dateutil.tz.gettz()),
        description="the creation time of the impulse",
    )
    ttl: int = Field(
        default=30,
        description="当一个 impulse 胜出生成 attention 时, 一定需要有过期时间. impulse 的强度也会随着时间调整曲线. ",
    )

    @classmethod
    def from_signal(cls, signal: Signal, belongs_to: str) -> Self:
        """
        一个简单的示例, 直接将 signal 转化成 impulse 不做任何处理.
        """
        return Impulse(
            source=belongs_to,
            trace_id=signal.trace_id or signal.id,
            priority=signal.priority,
            strength=signal.strength,
            messages=signal.messages.copy(),
            description=signal.description,
            prompt=signal.prompt,
            complete=signal.complete,
        )


class Nucleus(ABC):
    """
    并行 感知/思考/决策 单元的统一抽象.
    它接受输入信号, 返回动机.
    在输入场景中, 它是输入信号的治理层, 用于将高频的输入信号治理/加工/降频/加权后, 转化为 Mindflow 可以处理的 Impulse.
    可以拥有各种实现机制, 比如:
    1. lru buffer, 将所有的信号合并
    2. summary, 将信号合并摘要
    3. priory queue, 结合 maxsize 做单一信号量.
    4. arbiter, 加入仲裁者模型做快速校验.
    5. sidecar, 旁路思考, 向主路广播...

    同样, 它可以作为 MultiTasks/Planner/Timer/Ticker/MultiAgent 等各种机制, 通过 signal 和 impulse 两个大一统抽象管理特别复杂的
    异步运行逻辑, 与主交互脑通讯.
    """

    @abstractmethod
    def name(self) -> str:
        """
        用于区分不同的 Nucleus 单元.
        """
        pass

    @abstractmethod
    def signals(self) -> list[SignalName]:
        """
        声明监听的信号类型.
        """
        pass

    @abstractmethod
    def on_signal(self, signal: Signal) -> None:
        """
        接受一个信号量, 在内部开始执行校验逻辑, 生成 impulse.
        没有背压, 应当尽可能快地入队，不执行任何耗时或异步操作。内部应有独立的任务循环消费队列。
        """
        pass

    @abstractmethod
    def with_bus(self, signal_broadcast: Callable[[Signal], None], impulse_notify: Callable[[Impulse], None]) -> None:
        """
        注册总线, 可以广播信号, 或者发送 impulse.
        1. Nucleus 可以广播 signal 给其它监听者.
        2. Nucleus 产生了 Impulse, 可以回调通知, 比如回调 Mindflow.
        注意, Impulse 回调时不能 pop, 如果回调的 Impulse 无法抢占 attention, 应该会收到一个 suppress 信号.
        """
        pass

    @abstractmethod
    def suppress(self, suppress_by: Impulse) -> None:
        """
        如果产生的 impulse 不能被接纳, Nucleus 应该收到一个 suppress 信号
        可以在内部实现加权/降权 逻辑.
        :param suppress_by: 被别的信号压制, 得到别的信号. 未来可以通过决策单元判断是否要加权.
        """
        pass

    @abstractmethod
    def pop_impulse(self) -> Impulse | None:
        """
        吐出最新的 Impulse, 被 Attention 接受.
        """
        pass

    @abstractmethod
    def peek(self) -> Impulse | None:
        """
        查看一下最新的 Impulse.
        方便做 ranking.
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """
        启动 Nucleus 自身的生命周期, 包含异步逻辑, 或者启动子进程.
        """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        退出生命周期.
        """
        pass


@dataclasses.dataclass
class Observation:
    """
    上下文感知的快照.
    """

    context: dict[str, list[Message]]
    """与本轮输入相关的上下文, 只保留 1~n 轮. 作为字典, 相同分类更新覆盖"""
    messages: list[Message]
    """需要思考单元阅读的输入信息, 应该永久保存在历史中. """
    prompt: str
    """提示请求处理逻辑的 prompt, """

    def join(self, observation: Self) -> Self:
        context = self.context.copy()
        context.update(observation.context)
        messages = self.messages.copy()
        messages.extend(observation.messages)
        prompt = observation.prompt
        copied = Observation(
            context=context,
            messages=messages,
            prompt=prompt,
        )
        return copied


class LogosWriter(Protocol):
    """
    接受模型输出的指令流, 将它发送给执行单元.
    """

    @abstractmethod
    def send_nowait(self, delta: str) -> None:
        """
        send logos delta
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """
        start to send.
        """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        commit all
        """
        pass


Logos = AsyncIterator[str]

Articulate = Callable[[Observation], Logos]

class LogosStream(Protocol):
    """
    从 Logos 获取的输出流, 用来控制躯体.
    线程安全的 AsyncIterator[str]
    """

    def __aiter__(self) -> Self:
        return self

    @abstractmethod
    async def __anext__(self) -> str:
        """
        返回输入的 logos 直到输入结束, 或者 Attention 被终止.
        """
        pass


class Flag(Protocol):
    """
    对齐 Event 对应的接口, 不过要实现线程安全.
    """

    @abstractmethod
    async def wait(self) -> None:
        pass

    @abstractmethod
    def set(self) -> None:
        pass

    @abstractmethod
    def is_set(self) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class Attention(ABC):
    """
    一种三循环全双工运行时的资源和状态调度单元.
    它通常是 Impulse 创建出来的实例, 一直到 思考/执行 都结束后退出.
    """

    @abstractmethod
    async def wait_impulse(self) -> Impulse:
        """
        拿到 complete 为 True 的 Impulse.
        举例, ASR 输入的首包 Signal 创建了 complete == False 的 Impulse, 打断行为, 获取了注意力.
        实际上到接受到 complete Impulse 时, 才能正式开始响应. 它只是占据注意力.
        通常到 stale 的时候还没有拿到更新, attention 就会作废.
        """
        pass

    @abstractmethod
    def context(self) -> str:
        """
        形成注意力瞬间, 所有的感知单元的一个快照.
        """
        pass

    @abstractmethod
    def flag(self, name: str) -> Flag:
        """
        声明一个 flag, 用于生命周期通讯, 需要是一个线程安全的可阻塞对象.
        因为未来  躯体/思考/感知 可能运行在三个线程中.
        执行协议可以定义不同的生命周期节点, 方便一些逻辑做具体的阻塞.
        """
        pass

    def on_interpreted(self) -> Flag:
        """
        一个约定的生命周期, 表示 智能体输出的 logos 已经全部被合法解析完毕.
        """
        return self.flag('on_interpreted')

    @abstractmethod
    def logos(self) -> LogosWriter:
        """
        接受指令的通道. 约定是单独一方持有, 不应该是并行持有.
        """
        pass

    @abstractmethod
    def act(self) -> LogosStream:
        """
        接收指令的通道, 拿到 AsyncIterator[str]
        """
        pass

    @abstractmethod
    async def wait_done(self) -> None:
        """
        可用于阻塞到 Attention 生命周期运行结束.
        """
        pass

    @abstractmethod
    def should_preempt(self, impulse: Impulse) -> bool:
        """
        仲裁新的 impulse. 决定自身是否被中断. 调度发起者是 mindflow.
        最基础的仲裁逻辑:
        1. 如果 id 和当前 Impulse 相同, complete 取代 incomplete 并解除 impulse 阻塞. 否则丢弃 (并记录异常).
        2. 挑战的 impulse priory 低于当前 impulse 优先级, 返回 False, 目标 impulse 发起方接受 suppress 回调.
        3. 优先级相同, 应该基于同源提权, 异元降权的原理做强度比较.
        4. 如果挑战者优先级更高, 则挑战一定成功. 当前 Attention 应该 abort.
        5. 如果 priority 为 Fatal, 应该永远被打断.

        这是最简单的规则. Attention 更好的做法是有一个速度极快的仲裁者. 它要具备响应大量讯号挑战的极简算法.
        如果挑战成功, Mindflow 应该实例化新的 Attention 之后, abort 当前的 Attention.
        例如 on_challenge 触发 Mindflow 调度它 abort(reason="preempted")

        :return bool: 是否会被抢占.
        """
        pass

    @abstractmethod
    def start_soon(self, cor: Coroutine) -> asyncio.Future:
        """
        在 Attention 的运行状态中创建一个 Task, 或注册一个 Future. 随 Attention 结束而关闭, 生命周期统一治理.
        底层是一个 task group, 单一任务异常均会导致终止.
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """
        是否已经运行结束.
        """
        pass

    @abstractmethod
    def exception(self) -> Exception | None:
        pass

    @abstractmethod
    def abort(self, error: str | Exception | None) -> None:
        """
        显式声明退出 Attention.
        当 abort 提交时, 它所注册的任务全部会执行结束.
        """
        pass

    @abstractmethod
    async def __aenter__(self):
        """可重入的生命周期, 用来拦截未处理异常. """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """整个生命周期结束"""
        pass


class Mindflow(ABC):
    """
    三循环全双工智能体的思维调度中枢.
    它解决的核心问题是, 如何管理一个全双工三循环系统的运行逻辑.

    三循环: 1. 系统控制者;  2. AI 思考单元. 3. 躯体运行时.
    双工: 1. 躯体输出; 2. 感知输入.
    有复杂的中断逻辑: 0. 强制命令, 比如熔断, 急停. 1. 思考异常; 2. 执行异常; 3. 执行结束; 4. 输入更强的信号, 中断.

    同时有很多个状态和讯号通讯, 而在一个时间片里只有一组行为拥有可运行资源.
    Mindflow 的作用就是统筹所有的实现模块:

    1. nucleus: 感知单元, 接受原始信号量, 通过加工后返回有优先级效果的 Impulse. 解决并行
    2.
    """

    @abstractmethod
    def faculties(self) -> Iterable[Nucleus]:
        """
        持有的并行感知, 思考, 裁决单元.
        """
        pass

    @abstractmethod
    def with_nucleus(self, nucleus: Nucleus) -> Self:
        """
        动态注册新的感知单元.
        """
        pass

    @abstractmethod
    def on_impulse(self, impulse: Impulse) -> None:
        """
        接受一个 impulse, 并进入和当前 attention 的 challenge 仲裁.
        注意, 这里的 on_signal / on_impulse 作为总线提供给 Nucleus 时, 要防止信号成环无限传播.
        似乎没有系统机制可以百分之百预防.
        """
        pass

    @abstractmethod
    def attention(self) -> Attention | None:
        """
        返回当前的 Attention.
        """
        pass

    @abstractmethod
    def set_attention(self, attention: Attention) -> None:
        """
        通过系统操作直接注入 attention, 中断已经执行的 attention.
        绕过决策体系.
        """
        pass

    @abstractmethod
    def set_impulse(self, impulse: Impulse) -> None:
        """
        通过系统操作, 直接将 impulse 定义成 attention, 中断已经执行的 attention.
        绕过了感知决策体系.
        """
        pass

    @abstractmethod
    def pause(self, toggle: bool) -> None:
        """
        急停, 仍然接受 signal/impulse, 但不会分发, 而是直接丢弃. 只有 set_ 系统指令有意义.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        立刻停止.
        """
        pass

    def __aiter__(self) -> Self:
        return self

    @abstractmethod
    async def __anext__(self) -> Attention:
        """
        在生命周期中返回最新的 Attention, 方便定义清晰的 loop.
        每一轮 aborted 的 attention 应该要把异常结果提交给下一轮作为开始.
        """
        pass

    @abstractmethod
    async def __aenter__(self):
        """启动"""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出"""
        pass


if __name__ == "__example__":
    """
    整套实现思路的应用构想. 只是一个举例, 细节未打磨. 
    """


    def articulate(observation: Observation) -> Logos:
        """
        reasoning actions from observation
        generate logos for action.
        """
        pass


    def pop_observation(impulse: Impulse | None) -> Observation:
        """create observation snapshot"""
        pass


    async def conceive_stage(observation: Observation, logos: LogosWriter) -> None:
        """generate logos stage"""
        message = ''
        try:
            async with logos:
                async for delta in articulate(observation):
                    logos.send_nowait(delta)
                    message += delta
        finally:
            update_context(observation, message)


    async def side_thinking_stage(observation: Observation) -> None:
        """just observe and thinking, before attention released"""
        message = ''
        try:
            async for delta in articulate(observation):
                message += delta
        finally:
            update_context(observation, message)


    def allow_side_thinking() -> bool:
        """if the system allow side observation"""
        pass


    def update_context(observation: Observation, message: str) -> None:
        """update context"""
        pass


    async def thinking_loop(attention: Attention) -> None:
        impulse = await attention.wait_impulse()
        observation = pop_observation(impulse)
        logos = attention.logos()
        await conceive_stage(observation, logos)
        if allow_side_thinking():
            observation = pop_observation(None)
            await side_thinking_stage(observation)


    async def interpret(act: LogosStream) -> str:
        """wait interpret done, update the observation by runtime status"""
        pass


    def wait_action_done() -> AsyncIterator[Message]:
        """wait action executed, update the observation by runtime status"""
        pass


    async def action_loop(attention: Attention) -> None:
        output = ""
        try:
            act = attention.act()
            await interpret(act)
            # notify interpreted
            attention.on_interpreted().set()
            async for message in wait_action_done():
                output += message
            attention.abort(None)
        except Exception as e:
            attention.abort(str(e))
        finally:
            # handle output
            pass


    async def mindflow_main_loop(mindflow: Mindflow) -> None:
        async with mindflow:
            async for attention in mindflow:
                # 展开 attention 的异常拦截作用域. 不拦截 fatal
                async with attention:
                    _ = attention.start_soon(thinking_loop(attention))
                    _ = attention.start_soon(action_loop(attention))
                    await attention.wait_done()
