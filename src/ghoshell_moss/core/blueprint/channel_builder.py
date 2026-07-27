"""
how to build a channel — MOSShell channel 构建总入口 (面向开发者模型的全局知识).
the path of this module is ghoshell_moss.core.blueprint.channel_builder

Channel 与模型 (大脑) 之间有三个数据方向, 构建 channel 时先分清信息该走哪条:

- 下行 (result): command 返回值 -> `<result>` 消息. 观察语义三级:
  `return Any` (反馈进上下文, 不触发思考) / `return CommandUtil.observe(...)` (触发 Re-Act)
  / `raise CommandUtil.raise_observe(...)` (中断一切行动, 立刻思考).
- 中行 (progress): `CommandUtil.set_progress`, 长命令执行中途的状态披露.
- 上行 (signal): `CommandUtil.send_signal` / `send_input_signal` / `create_signal_task`,
  command 主动推给大脑, 构成自驱循环 (由 Mindflow 仲裁).

模型看到什么, 以 ChannelMeta 为唯一权威契约 (ghoshell_moss.core.concepts.channel:ChannelMeta):
interface (命令签名自动反射) / instruction / context / memory / states.
本模块是动词 (怎么构建), ChannelMeta 是名词 (暴露成什么), 此处不重复枚举那份契约.

CTML 如何调用 channel: `moss ctml read`.
"""

from abc import ABC, abstractmethod

from PIL import Image
from typing import Union, Callable, Coroutine, Any, Optional, TypeVar, AsyncIterable, Type

from ghoshell_container import IoCContainer
from typing_extensions import Self

from ghoshell_moss.core import ChannelRuntime
from ghoshell_moss.message import Message
from ghoshell_moss.core.concepts.command import Command, Observe, ObserveError
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.topic import TOPIC_MODEL, Publisher, Subscriber
from ghoshell_moss.core.blueprint.mindflow import Signal
import asyncio

__all__ = [
    "Channel", "ChannelFactory",
    "CommandFunction", "MessageFunction", "StringType", "LifecycleFunction",
    "Message",
    "MessageType",
    "Builder",
    "MutableChannel",
    "new_channel", "new_command",
    "CommandUtil",
    "Observe", "ObserveError",

    # 作为示例用的实现风格.
    "ChannelInterface", "ChannelCreator",
]

ChannelFactory = Callable[[IoCContainer], Channel | None]

CommandFunction = Union[Callable[..., Coroutine], Callable[..., Any]]
"""
用于描述一个本地的 python 函数 (或者类的 method) 可以被注册到 Channel 中变成一个 command. 
"""

MessageType = Message | str | Image.Image
MessageFunction = Union[
    Callable[[], Coroutine[None, None, list[MessageType]]],
    Callable[[], list[MessageType]],
]
"""
可以生成消息体的函数. 这种函数注册到 Channel 中, 可以用来动态地生成 Context Messages 与 Memory Messages.
AI 通过双工通讯, 在每个关键帧思考的瞬间, 提取对应的消息体替换到上下文中.

注: Memory Messages 的字段契约已在 ChannelMeta.memory 就位, 但 Builder 尚无挂载点
(memory_messages 钩子未提供). 设计已锁定, 见 workstream channel-meta-dyn-static.
"""

StringType = Union[
    str,
    Callable[[], str],
    Callable[[], Coroutine[None, None, str]],
]

LifecycleFunction = Union[Callable[..., Coroutine[None, None, None]], Callable[..., None]]
"""
用于描述一个本地的 python 函数 (或者类的 method), 可以用来定义 channel 自身生命周期行为. 

一个 Channel 运行的生命周期设计是: 

- [on startup] : channel 启动时
- [on idle] : 闲时, 没有任何命令输入
- [on close] : channel 关闭时
- [on running] : start < running < close

另有 refresh_meta 钩子, 属于刷新节奏而非运行生命周期: 每个 refresh 周期重新生成 metas 前调用.
见 Builder.refresh_meta.

举一个典型的例子: 数字人在执行动画 command 时, 运行轨迹动画; 执行完毕后, 没有命令输入时, 需要返回呼吸效果 (on_idle)
"""

_ChannelName = str

INSTANCE = TypeVar("INSTANCE", bound=object)


class CommandUtil:
    """
    在 Command 内部使用的工具, 仅在 Command | Channel Lifecycle Function 被执行时可以使用.
    通过 contextlib ctx 获取调用者能力.
    包含各种 Command 函数内需要的常用 API.
    """

    @classmethod
    def force_get_contract(cls, contract: type[INSTANCE]) -> INSTANCE:
        """
        force get contract from ioc Container.
        raise Error if the contract is not registered.
        combine with moss manifests to know existing contracts
        """
        # dig deeper only when necessary
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        return ChannelCtx.get_contract(contract)

    @classmethod
    def get_contract(cls, contract: type[INSTANCE]) -> INSTANCE | None:
        """
        if contract is not registered, return None
        """
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        runtime = ChannelCtx.runtime()
        return runtime.container.get(contract)

    @classmethod
    def runtime(cls) -> ChannelRuntime:
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        runtime = ChannelCtx.runtime()
        if runtime is None:
            cls.raise_observe('CommandUtil is not in the runtime context')
        return runtime

    @classmethod
    def enabled(cls) -> bool:
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        runtime = ChannelCtx.runtime()
        return runtime is not None

    @classmethod
    def topic_publisher(cls, model: Type[TOPIC_MODEL]) -> Publisher[TOPIC_MODEL]:
        """
        return a topic publisher, use it in with statement:

        async with CommandUtil.topic_publisher(model) as publisher:
            await publisher.publish(topic_model: TOPIC_MODEL)

        or managed within startup -> close lifecycle of the channel
        """
        runtime = cls.runtime()
        return runtime.topic_publisher(model)

    @classmethod
    def topic_subscriber(cls, model: Type[TOPIC_MODEL], *, maxsize: int = 1000) -> Subscriber[TOPIC_MODEL]:
        """
        return a topic subscriber, use it in with statement:

        async with CommandUtil.topic_subscriber(model) as subscriber:
            model = await subscriber.poll()

        or managed within startup -> close lifecycle of the channel
        """
        runtime = cls.runtime()
        return runtime.topic_subscriber(model, maxsize=maxsize)

    @classmethod
    def logger(cls):
        """返回日志模块 logging.Logger, 只保留基础的记录函数. """
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        from ghoshell_moss.contracts import LoggerItf, get_moss_logger
        return ChannelCtx.container().get(LoggerItf) or get_moss_logger()

    @classmethod
    def set_progress(cls, progress: str) -> None:
        """set progress of the current command task (ONLY when needed to do so)"""
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        task = ChannelCtx.task()
        if task is not None:
            task.set_progress(progress)

    @classmethod
    def observe(cls, value: str) -> 'str | Observe':
        """返回一个需要立刻观察的信息. 实际上返回的是 Observe 对象, 但可以在 command 返回值定义为 str"""
        return Observe(messages=[Message.new().with_content(value)])

    @classmethod
    def raise_observe(cls, value: str) -> None:
        """通过 raise 来在 command 中返回一个可中断其它逻辑的观察信息. """
        raise cls.observe_error(value)

    @classmethod
    def observe_error(cls, value: str) -> 'ObserveError':
        from ghoshell_moss.core.concepts.command import ObserveError
        return ObserveError(value)

    @classmethod
    def send_signal(cls, signal: Signal) -> None:
        """
        在 command 内发送信号给自己的大脑. 构成自驱循环.
        需要发送不同类型的 Signal, 可参考服务发现的 SignalMeta 协议.
        """
        from ghoshell_moss.core.blueprint.session import Session
        session = cls.force_get_contract(Session)
        if isinstance(signal, Signal):
            session.add_signal(signal)
        else:
            raise TypeError(f"only Signal or str is accepted")

    @classmethod
    def create_task(cls, coroutine) -> asyncio.Task:
        """
        create an asyncio task in channel lifecycle.
        useful for some task going on after command itself done
        """
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        runtime = ChannelCtx.runtime()
        return runtime.create_asyncio_task(coroutine)

    @classmethod
    def is_task_done(cls) -> bool:
        """
        判断触发当前 command 执行的 task 是否已经完成.
        方便同步函数里做状态清理.
        """
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        task = ChannelCtx.task()
        return task.done()

    @classmethod
    def get_task_context(cls) -> dict[str, Any]:
        """
        返回 task 创建时从环境传入的参数.
        """
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        task = ChannelCtx.task()
        return task.context

    @classmethod
    def send_input_signal(cls, content: str, *, description: str = '') -> None:
        """发送标准的请求信号给 ghost. """
        from ghoshell_moss.core.blueprint.session import Session
        session = cls.force_get_contract(Session)
        session.add_input_signal(content, description=description)

    @classmethod
    async def create_signal_task(
            cls,
            *,
            closure: Callable[[], Coroutine[None, None, Signal | str]],
    ) -> None:
        """
        在 Command 内创建一个异步的 Signal 回调 task, 不阻塞 Command 返回.
        当 closure 异步执行完毕后, 结果的 Signal 会发送给 ghost.
        """
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        task = ChannelCtx.task()
        caller = task.caller_name()
        task_ctx = task.context

        async def _send_signal_after_task_done() -> None:
            nonlocal closure, task_ctx, caller
            signal = await closure()
            if isinstance(signal, Signal):
                cls.send_signal(signal)
            elif isinstance(signal, str):
                cls.send_input_signal(signal)
            else:
                cls.logger().error(
                    "signal task returns invalid signal type: %s, task %s, task context %s,",
                    signal, caller, task_ctx
                )

        runtime = ChannelCtx.runtime()
        runtime.create_asyncio_task(_send_signal_after_task_done())


def new_command(
        func: CommandFunction,
        *,
        name: str = "",
        doc: Optional[StringType] = None,
        comments: Optional[StringType] = None,
        interface: Optional[StringType | Callable[[...], Coroutine[None, None, Any]]] = None,
        available: Optional[Callable[[], bool]] = None,
        # --- 高级参数 --- #
        blocking: bool = True,
        call_soon: bool = False,
        priority: int = 0,
        always_observe: bool = False,
        timeout: Optional[float] = None,
        visible: bool = True,
) -> Command:
    """
    定义一个 Command. 逻辑与 Builder.command 相同.

    在 own_commands 之类的场景要反射 python 函数为 Command 时, 优先使用它 (等价 PyCommand 但实现可替换)
    """
    from ghoshell_moss.core.concepts.command import PyCommand
    return PyCommand(
        func=func,
        name=name,
        doc=doc,
        comments=comments,
        interface=interface,
        available=available,
        blocking=blocking,
        call_soon=call_soon,
        priority=priority,
        always_observe=always_observe,
        timeout=timeout,
        visible=visible,
    )


class Builder(ABC):
    """
    用来动态构建一个 Channel 的通用接口.

    Builder 有唯一 id. builder 是有副作用的.
    一个实例应该只使用一次.
    """

    # ---- decorators ---- #
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def available(self, func: Callable[[], bool]) -> Callable[[], bool]:
        """
        decorator
        注册一个函数, 用来动态生成整个 Channel 的 available 状态.
        Channel 每次刷新状态时, 都会从这个函数取值. 否则默认为 True.
        >>> async def building(chan: MutableChannel) -> None:
        >>>     chan.build.available(lambda: True)
        """
        pass

    @abstractmethod
    def instruction(self, func: StringType) -> StringType:
        """
        decorator
        注册字符串或者函数, 用来生成当前 channel 提供的 instruction / system prompt. 只生成一次.

        Channel as Context Components 思想:
            直接将 Channel 作为上下文的组件, 提供模块化的上下文讯息.
            讯息应该足够简洁, 高效, 同时注意 token 用量. 具体裁剪和压缩由 Agent 工程决定.
            由于 Channel 持有的 Command 可以影响自身的运行时状态, 所以 Channel 提供了完整的上下文反身性.
            结合后续的 StatefulChannel 实现, 同时提供渐进式披露的能力.

        红线: instruction 里绝不要重复罗列本 channel 有哪些 command ——
        命令签名已由 interface 自动反射给模型 (Code as Prompt), 手写重列必然随代码漂移成谎言.
        instruction 只写 interface 表达不了的东西: 整体用法, 协作约定, 状态语义.

        channel 对模型暴露的完整上下文字段契约, 以 ChannelMeta 为唯一权威
        (ghoshell_moss.core.concepts.channel:ChannelMeta).

        注意! Channel 仅在特别有必要的时候, 才需要提供上下文讯息. 大部分 channel 完全不用提供.
        """
        pass

    @abstractmethod
    def context_messages(self, func: MessageFunction, reset: bool = False) -> MessageFunction:
        """
        decorator
        注册一个上下文生成函数. 用来生成 channel 运行时动态的上下文.
        举个例子, 如果是视觉模块, 则可以把当前瞬间看见的图片, 和视觉模块的简单描述作为 context messages.

        这部分上下文会出现在模型上下文的 inputs 之前或之后.
        当 channel 每次刷新后, 都会通过它生成动态的上下文消息体.
        通常只有具备感知功能的模块, 需要提供动态的 context messages.

        >>> async def building(chan: MutableChannel) -> None:
        >>>     async def context() -> list[Message]:
        >>>         return [
        >>>             Message.new().with_content("dynamic information")
        >>>         ]
        >>>     chan.build.context_messages(context)
        """
        pass

    def content_command(
            self,
            func: Callable[[AsyncIterable[str]], Coroutine[None, None, None | str]],
            doc: Optional[str] = None,
            override: bool = True,
    ) -> Command[None]:
        """
        register a special function for channel's content method.
        """
        name = ChannelRuntime.__content__.__name__
        return self.command(
            name=name,
            # 允许覆盖说明.
            doc=doc,
            # use __content__ as interface, override the docstring if need.
            interface=ChannelRuntime.__content__,
            override=override,
            return_command=True,
        )(func)

    @abstractmethod
    def add_command(
            self,
            command: Command,
            *,
            override: bool = True,
            name: Optional[str] = None,
    ) -> None:
        """
        添加一个 Command 对象.
        """
        pass

    @abstractmethod
    def command(
            self,
            *,
            name: str = "",
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            tags: Optional[list[str]] = None,
            interface: Optional[StringType | Callable[[...], Coroutine[None, None, Any]]] = None,
            available: Optional[Callable[[], bool]] = None,
            override: bool = True,
            # --- 高级参数 --- #
            blocking: bool = True,
            call_soon: bool = False,
            priority: int = 0,
            return_command: bool = False,
            always_observe: bool = False,
            timeout: float | None = None,
            visible: bool = True,
    ) -> Callable[[CommandFunction], CommandFunction | Command]:
        """
        decorator
        将一个 Python 函数或类的 method 注册到 Channel 上, 成为 Channel 的一个 Command.
        函数会自动反射出 signature, 作为给大模型查看的讯息.
        大模型只会看到函数的签名和注释, 不会看到原始代码.

        :param name: 不为空, 则改写这个函数的名称.
        :param doc: 重定义函数的docstring, 如果传入的是一个函数, 则会在每次刷新时, 动态调用这个函数, 生成它的 docstring.
        :param comments: 改写函数的 body 部分, 用注释形式提供的字符串. 每行前会自动添加 '#'. 不用手动添加.
                         Comments 最直接的用处是写使用的案例, 说明, 执行逻辑等. 辅助 AI 理解.

        :param interface: 大模型看到的函数代码形式. 一旦定义了这个, doc, name, comments 就都会失效.
                支持三种传参方式:
                - str: 直接用字符串来定义模型看到的函数签名.
                    注意, 必须写成 Python Async 的形式.
                    async def foo(...) -> ...:
                      '''docstring'''
                      # comments
                - callalble[[], str]: 生成模型签名的函数
                - async function: 直接反射这个 function, 来生成一个模型签名的字符串. 可以定义虚拟函数作为 interface.
        :param override: override existing one
        :param tags: 标记函数的分类. 可以让使用者用来过滤和筛选.
        :param available: 通过一个 Available 函数, 定义这个命令的状态. 当这个函数返回 False 时, Command 会动态地变成不可用.
                这种方式, 可以结合状态机逻辑, 动态定义一个 Channel 上的可用函数.
        :param blocking: 这个函数是否会阻塞 channel. 为 None 的话跟随 channel 的默认定义.
                blocking = True 类型的 Command 执行完毕前, 会阻塞后续 Command 执行, 通常是在机器人等需要时序规划的场景中.
                blocking = False 类型则会并发执行. 对于没有先后顺序的工具, 可以设置并行.
        :param call_soon: 决定这个函数进入轨道后, 会第一时间执行 (不等待调度), 还是等待排队执行到自身时.
                如果是 (blocking and call_soon) == True, 会在入队时立刻清空队列.

        :param priority: 命令优先级, <0 时, 有新的命令加入, 就会被自动取消. >0 时, 之前所有优先级比自己低的都会立刻取消.
                高级功能, 不理解的情况下请不要改动它.

        :param return_command: 为真的话, 返回的不是原函数, 而是一个可以视作该函数的 Command 对象. 通常用于测试.
        :param always_observe: 为 True 的话, 不需要特别声明, command 的返回值总是会标记需要下一轮观察思考.
        :param timeout: if not None, set default timeout for the command.
        :param visible: 命令是否对模型可见. 不可见, 通常因为这个命令是模型认知协议的一部分, 以至于可以省略它.

        CommandFunction 最佳实践是:
        >>> # 原始函数是 async, 从而有能力根据真实运行的时间, 阻塞 Channel 后续命令.
        >>> # 参数和返回值有明确的类型约束, 类型约束也是 prompt 的一部分.
        >>> # 使用可序列化对象作为入参和出参
        >>> # 依赖线程安全的逻辑, 定义为 sync 函数.
        >>> async def func(arg: type) -> Any:
        >>>     '''有清晰的说明'''
        >>>     try
        >>>         # 执行逻辑, 不能有线程阻塞, 否则会阻塞全局.
        >>>         # CommandUtil.create_task
        >>>         ...
        >>>         # return None # 仅表示执行结束, 不需要特别观察
        >>>         # return Any  # 返回讯息反馈给上下文, 但不需要触发 Re-Act. 或配置 always_observe 使之触发思考.
        >>>         # return CommandUtil.observe('xxx')  记模型需要观察和思考的结果, 会触发 Re-Act.
        >>>         # raise CommandUtil.raise_observe(...)  中断所有的行动, 立刻触发思考.
        >>>     except asyncio.CancelledError:
        >>>         # 命令可以被调度层正常取消, 有取消的行为. 通常 AI 可以随时取消一个运行的 Command.
        >>>         ...
        >>>     except Exception as e:
        >>>         # 正确处理异常
        >>>         ...
        >>>     finally:
        >>>         # 有运行结束逻辑.
        >>>         ...

        async 函数支持 cancel 生命周期, 所以 command 是一个拥有 done / cancel / exception 完整语义的单元.
        如果用同步函数定义, 需要通过 CommandUtil.is_task_done 来管理中间中断和清空状态逻辑, 避免阻塞行为因为同步函数未完成而冲突.
        """
        pass

    @abstractmethod
    def idle(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        decorator
        注册一个生命周期函数, 当 Channel 运行 policy 时, 会执行这个函数.

        生命周期的最佳实践是:

        >>> # 原始函数是 async, 从而有能力根据真实运行的时间, 阻塞 Channel 后续命令.
        >>> async def func() -> None:
        >>>     # 可以获取执行这个 command 的真实 runtime
        >>>     try
        >>>         # 通过全局的 IoC 容器获取依赖, 可以拿到运行时的依赖注入.
        >>>         contract = CommandUtil.force_get_contract(...)
        >>>         ...
        >>>     except asyncio.CancelledError:
        >>>         # 生命周期函数随时会被 Channel Runtime 调度取消
        >>>         ...
        >>>     except Exception as e:
        >>>         # 正确处理异常
        >>>         ...
        >>>     finally:
        >>>         # 有运行结束逻辑.
        >>>         ...
        """
        pass

    @abstractmethod
    def startup(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        启动时执行的生命周期函数
        """
        pass

    @abstractmethod
    def close(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        关闭时执行的生命周期函数
        """
        pass

    @abstractmethod
    def running(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        在整个 Channel Runtime is_running 时间里运行的逻辑. 只会被调用一次.
        注意, 这个函数和 idle / executing 是并行的.
        """
        pass

    @abstractmethod
    def refresh_meta(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        decorator
        注册一个 async 回调, 在每个 refresh 周期重新生成 metas 前调用.

        用于需要在 refresh 时做 I/O 的场景 —— alive_cells 查询, proxy 缓存更新,
        外部状态同步等. 与 get_virtual_children() 配合: 这里更新缓存 (async),
        get_virtual_children() 返回缓存 (sync, 快速).

        >>> async def func() -> None:
        >>>     alive = await matrix.alive_cells()
        >>>     ...  # 更新 proxy 缓存, 供下轮 get_virtual_children() 使用
        """
        pass

    @abstractmethod
    def with_binding(self, contract: type[INSTANCE], instance: INSTANCE) -> Self:
        """
        注册一个依赖, 在 Channel 实例化时注入 IoC 容器.
       可以通过 CommandCtx.get_contract 获取.
        依赖注入完全是可选的, 可以通过模块实例化/全局工厂等替代.
        """
        pass

    @abstractmethod
    def with_contract_factory(
            self,
            contract: type[INSTANCE],
            factory: Callable[[...], INSTANCE],
            *,
            singleton: bool = True,
            override: bool = False,
    ) -> Self:
        """
        注册一个依赖的工厂方法.
        这个工厂方法当 Channel 实例化 Runtime 时, 会把 contract 的工厂函数注册到 IoC 容器.
        """
        pass

    @abstractmethod
    def import_channels(
            self,
            *children: Channel | ChannelFactory | tuple[Channel | ChannelFactory, _ChannelName],
    ) -> Self:
        """
        add sustain channels to the channel.
        """
        pass


class MutableChannel(Channel, ABC):
    """
    一个约定, 用来描述拥有动态构建能力的 Channel.
    """

    def import_channels(
            self,
            *children: Channel | ChannelFactory | tuple[Channel | ChannelFactory, _ChannelName],
    ) -> Self:
        """
        添加子 Channel 到当前 Channel. 形成树状关系.
        效果可以比较 python 的 import module as name
        """
        self.build.import_channels(*children)
        return self

    @property
    @abstractmethod
    def build(self) -> Builder:
        """
        支持通过 Builder 动态构建一个 Channel.
        """
        pass

    @abstractmethod
    def children(self) -> dict[_ChannelName, Channel]:
        """
        return all the static imported channel
        """
        pass

    @abstractmethod
    def virtual_children(self) -> dict[_ChannelName, Channel]:
        """
        return the virtual children channels
        """
        pass


def new_channel(name: str, description: str = "", uid: str | None = None) -> MutableChannel:
    """
    Create a new Mutable/Stateful Channel object with builder.
    Able to define all kinds of channels.
    Use this tool to build your own channel object.
    """
    from ghoshell_moss.core.py_channel import PyChannel
    # if uid is None, auto generate uuid for it.
    return PyChannel(name=name, description=description, uid=uid)


class ChannelCreator(ABC):
    """
    示例, 如果不用 new_channel 等面向组合风格创建 Channel
    仍然可以用这种面向对象的风格来创建.
    """

    @classmethod
    @abstractmethod
    def factory(cls, container: IoCContainer) -> Channel:
        """创建 channel 的函数本身可以注册到 channel. """
        pass


class ChannelInterface(ChannelCreator, ABC):
    """
    面向对象更激进的 Channel 实现风格.
    不仅定义 factory, 还提前定义好函数.
    """

    @classmethod
    @abstractmethod
    def new(cls, container: IoCContainer) -> Self:
        """确保可以在 ioc 中实例化自己. """
        pass

    @abstractmethod
    def as_channel(self) -> Channel:
        """实例化后的自己, 可以生产 channel 对象."""
        pass

    @classmethod
    def factory(cls, container: IoCContainer) -> Channel:
        self_instance = cls.new(container)
        return self_instance.as_channel()


if __name__ == "__build_channel_by_channel_interface_example__":
    # 通过 ChannelCreator 的风格创建面向对象的抽象/实现的方式.
    main = new_channel(name="__main__")


    class FooInterface(ChannelInterface):

        @abstractmethod
        async def foo(self) -> str:
            pass

        @classmethod
        @abstractmethod
        def new(cls, container: IoCContainer) -> Self:
            pass

        def as_channel(self) -> Channel:
            channel = new_channel(name="foo")
            channel.build.command(
                interface=FooInterface.foo,
            )(self.foo)
            return channel


    class FooImpl(FooInterface):

        def __init__(self, c: IoCContainer):
            self.c = c

        async def foo(self) -> str:
            return self.c.name

        @classmethod
        def new(cls, container: IoCContainer) -> Self:
            return cls(container)


    # 直接将工厂方法注入到通道中.
    main.import_channels(FooImpl.factory)


async def test_channel(
        *channels: Channel,
        ctml: str,
        timeout: float | None = None,
):
    """Convenience wrapper around ctml_shell_test for app channel testing.

    This is a baseline helper — it covers the common case of sending CTML to a
    single channel and checking results.  Before using it you should understand
    CTML syntax::

        moss ctml read

    For more complex scenarios (scopes, until=any/all, observe, cancel, nested
    channels) this wrapper is not enough.  Read the source of ``ctml_shell_test``,
    search for how it is used in existing tests, and consult the CTML syntax
    (``moss ctml read``).

    Usage in app tests::

        from ghoshell_moss.core.blueprint.channel_builder import new_channel, test_channel

        chan = new_channel(name="my_app")

        @chan.build.command()
        async def greet(name: str) -> str:
            return f"Hello, {name}"

        tasks = await test_channel(chan, ctml='<apps.my_app:greet name="world" />')
        assert len(tasks) == 1
        assert await tasks[0] == "Hello, world"
    """
    from ghoshell_moss.core.ctml import ctml_shell_test
    return await ctml_shell_test(*channels, ctml=ctml, timeout=timeout)
