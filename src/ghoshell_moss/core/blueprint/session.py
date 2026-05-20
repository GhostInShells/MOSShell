from typing import Callable, AsyncIterator, AsyncGenerator, Protocol, NamedTuple
from typing_extensions import Self
from ghoshell_moss.contracts.workspace import Storage
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.blueprint.mindflow import Signal, SignalMeta, InputSignal
from typing import Iterable, Literal
from abc import ABC, abstractmethod
from ghoshell_moss.message import Message
from pydantic import BaseModel, Field
from PIL.Image import Image

Role = Literal['system', 'logos', 'log', 'error', 'task']


class OutputItem(BaseModel):
    """
    可以用于输出的原子化数据结构.

    是整个系统的对外输出.
    以 Message 为基础.
    """
    role: str | Role = Field(
        default='log',
        description="消息的类型.",
    )
    log: str = Field(
        default="",
        description="some log information.",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description='messages',
    )

    @classmethod
    def new(cls, role: Role | str, *messages: Message, log: str = '') -> Self:
        if isinstance(role, str):
            return cls.model_construct(role=role, messages=[], log=log).with_messages(*messages)
        else:
            return cls(role=role, log=log).with_messages(*messages)

    def messages_string(self) -> str:
        """how to convert all messages into string without none-string type"""
        if len(self.messages) > 0:
            contents = []
            for msg in self.messages:
                contents.append(msg.to_content_string())
            return "\n".join(contents)
        return ""

    def with_messages(self, *messages: Message | str) -> Self:
        for msg in messages:
            # 接受字符串处理后的消息.
            if isinstance(msg, str):
                self.messages.append(Message.new().with_content(msg))
            else:
                self.messages.append(msg.compact())
        return self


class Sample(NamedTuple):
    """stream 协议返回的结果. 未来可能要扩展"""
    relative_key: str
    payload: bytes


class StreamSubscriber(Protocol):
    """
    session stream 订阅的控制句柄.

    >>> async def consume(stream: StreamSubscriber):
    >>>       async with stream:
    >>>         async for msg in stream:
    >>>             print(msg)
    """

    @abstractmethod
    def full_key(self) -> str:
        """底层协议完整的 key"""
        pass

    @abstractmethod
    def relative_key(self) -> str:
        """在 session 中创建 key 的相对路径"""
        pass

    @abstractmethod
    async def __aenter__(self) -> 'StreamSubscriber':
        """先进入生命周期管理才能使用. """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """确保有明确的退出信号."""
        pass

    def __aiter__(self) -> AsyncIterator[Sample]:
        """支持异步迭代器, 阻塞获取后续数据."""
        return self

    @abstractmethod
    async def __anext__(self) -> Sample:
        """
        :raise StopAsyncIteration:
        """
        pass


class OutputBuffer(ABC):
    """
    用于实现一个 OutputItem 的 先消费, 后使用缓冲区.
    """

    @abstractmethod
    def close(self) -> None:
        """关闭 buffer"""
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """是否关闭"""
        pass

    @abstractmethod
    def add_output(self, item: OutputItem) -> None:
        """添加 item, 需要实现线程安全. """
        pass

    @abstractmethod
    def values(self) -> Iterable[OutputItem]:
        """返回所有的 items. 会生成一个线程安全的快照. """
        pass

    @abstractmethod
    def updated_at(self) -> float:
        """最后更新的 timestamp"""
        pass


class Session(ABC):
    """
    MOSS 运行时当前会话的通讯总线.

    各种不同的组件通过总线进行通讯.

    默认提供多种通讯路径:
      - output: 结构化消息 (OutputItem), 适合 event 和状态通知. 属于全局的消息输出协议. 是系统对外单向输出协议.
      - signal: Mindflow 感知信号, 驱动三循环, 用于驱动 MOSS 中运行的 Ghost. 详见 Mindflow. 是并行信号输入协议.
      - file: 基于 Session 级别的文件夹, 可以做文件级别读写通讯.
      - stream: 字节流 pub/sub, 适合 logos 等实时流式数据, 可以自定义协议. 原则上是单一有序发布, 多端接收.
      - topic service: 基于可用的 Topic 强类型广播协议通讯. 是原子化的 n * m  广播总线.

    todo:
        1. 实现共享的 parameters. 类似分布式中心协议.
        2. 实现可注册的基于 key 的函数. actor 协议.
    """

    LOGOS_KEY = 'logos'
    """logos stream 的 key 前缀. 完整 key 需要通过 stream key 获取."""

    @property
    @abstractmethod
    def session_scope(self) -> str:
        """
        所属的会话 scope
        """
        pass

    @property
    @abstractmethod
    def session_id(self) -> str:
        """
        session id
        """
        pass

    @abstractmethod
    def add_signal(self, signal: Signal) -> None:
        """
        input a mindflow signal to the Session
        """
        pass

    def add_input_signal(
            self,
            *values: str | Image | Message,
            description: str = '',
            priority: int | None = None,
            meta: SignalMeta | None = None,
            stale_timeout: float = 0,
    ) -> None:
        """
        easy way to add a default input signal to the Mindflow
        """
        meta = meta or InputSignal()
        signal = meta.to_signal(
            *values,
            description=description,
            priority=priority,
            stale_timeout=stale_timeout,
        )
        self.add_signal(signal)

    @abstractmethod
    def on_signal(self, callback: Callable[[Signal], None]) -> None:
        """
        listen to the MOSS input signal
        """
        pass

    @property
    @abstractmethod
    def topics(self) -> TopicService:
        """
        基于 Topic 协议的服务.
        """
        pass

    @property
    @abstractmethod
    def storage(self) -> Storage:
        """
        session 专属的 storage.
        需要的话可以将文件作为通讯方式.
        """
        pass

    @abstractmethod
    def output(self, role: str | Role, *messages: Message | str, log: str = '') -> None:
        """
        输出消息给 moss 共享 session 的终端.
        不应有线程阻塞
        :param role: 输出角色分类
        :param messages: 消息体，无消息时可通过 log 单独描述
        :param log: 单行摘要，verbose 场景下供展示使用
        """
        pass

    @abstractmethod
    def on_output(self, callback: Callable[[OutputItem], None]) -> None:
        """
        输出回调监听 conversation item.
        可以用来做个什么渲染.
        """
        pass

    @abstractmethod
    def output_buffer(
            self,
            maxsize: int = 100,
    ) -> OutputBuffer:
        """
        生产一个 OutputBuffer
        """
        pass

    # ── stream 协议 ──────────────────────────────

    @abstractmethod
    def is_running(self) -> bool:
        """session 是否在运行中. """
        pass

    @abstractmethod
    def self_explain(self) -> str:
        """
        Session 协议自解释: 通讯协议, key 命名空间, stream 约定等等.
        方便调试和运行时检查.
        """
        pass

    @abstractmethod
    def sub_stream(
            self,
            relative_key: str,
            callback: Callable[[Sample], None],
    ) -> Callable[[], None]:
        """
        订阅字节流. callback 接收解码后的 payload. 返回 StreamHandle 管理生命周期.
        通过自己定义的 Key 来对齐协议约定.

        :param relative_key: Session 层面的 key.
        :param callback: 需要注意线程安全.
        :return: 返回 Stop 句柄, 取消监听.
        """
        # Session 是跨进程共享的, 进程之间可以用自身约定的协议进行通讯. 用 bytes 作为传输包, 协议自行定义.
        # Session 底层需要提供通讯的基础建设, 目前默认实现是 zenoh. 支持用通配符做 key.
        pass

    @abstractmethod
    def pub_stream_delta(self, relative_key: str, delta: bytes) -> None:
        """
        向 Session Stream 总线中广播 payload.
        relative_key 为 session 层面的 key, 实现层负责转换为完整路径.
        """
        pass

    @abstractmethod
    def get_stream(
            self, relative_key: str, *, maxsize: int = 0,
    ) -> StreamSubscriber:
        """
        阻塞的方式获取字节流. maxsize=0 无限缓冲. 调用方 async for 消费.
        """
        # 底层
        pass

    @abstractmethod
    def stream_key_expr(self, relative_key: str) -> str:
        """生成完整的 stream key 路径. 子类可覆盖."""
        # session 需要定义并且暴露自身的 stream key 实现, 方便被调研和理解.
        pass

    # ── logos stream ──────────────────────────────

    def pub_logos(
            self,
            *deltas: str,
            session_id: str | None = None,
    ) -> None:
        """
        发送模型生产的 logos 片段 (默认是 ctml 流, 详见 Mindflow) 到总线.

        :param deltas: 流式数据的片段.
        :param session_id: 默认使用当前会话的 Session id 传递 logos 流.

        技术上需要实现有序.
        """
        sid = session_id or self.session_id
        for delta in deltas:
            self.pub_stream_delta(
                f"{self.LOGOS_KEY}/{sid}",
                delta.encode('utf-8'),
            )

    async def get_logos(
            self, *, session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        基于约定的协议, 获取广播的 Stream 流.
        """
        sid = session_id or self.session_id
        stream = self.get_stream(f"{self.LOGOS_KEY}/{sid}")
        async with stream:
            async for delta in stream:
                yield delta.payload.decode('utf-8')

    @abstractmethod
    async def __aenter__(self) -> Self:
        """session 需要定义自身的生命周期, 方便 matrix 统一治理. """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
