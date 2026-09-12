"""Session — the communication bus of the current MOSS runtime session.

``Session`` carries the duplex traffic of a running runtime: ``OutputItem`` is the atomic
output structure, ``OutputBuffer`` buffers it for consume-then-use, ``Sample`` is a
stream-protocol result, and ``StreamSubscriber`` is the control handle for subscribing to
the session stream.
"""

from typing import Callable, AsyncIterator, AsyncGenerator, Protocol, NamedTuple
from typing_extensions import Self
from ghoshell_moss.contracts.workspace import Storage
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.concepts.qa import QAManager
from ghoshell_moss.core.blueprint.mindflow import Signal, SignalMeta, InputSignalMeta
from typing import Iterable, Literal
from abc import ABC, abstractmethod
from ghoshell_moss.message import Message
from pydantic import BaseModel, Field
from PIL.Image import Image

Role = Literal['system', 'logos', 'log', 'error', 'task']


class OutputItem(BaseModel):
    """
    The atomic output structure of the system.

    It is the system's outward output, built on Message.
    """
    role: str | Role = Field(
        default='log',
        description="The kind of message.",
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
            # Accept a message after string conversion.
            if isinstance(msg, str):
                self.messages.append(Message.new().with_content(msg))
            else:
                self.messages.append(msg.compact())
        return self


class Sample(NamedTuple):
    """Result returned by the stream protocol. May be extended in the future."""
    relative_key: str
    payload: bytes


class StreamSubscriber(Protocol):
    """
    Control handle for subscribing to a session stream.

    >>> async def consume(stream: StreamSubscriber):
    >>>       async with stream:
    >>>         async for msg in stream:
    >>>             print(msg)
    """

    @abstractmethod
    def full_key(self) -> str:
        """The full key in the underlying protocol"""
        pass

    @abstractmethod
    def relative_key(self) -> str:
        """Relative path of the key created inside the session"""
        pass

    @abstractmethod
    async def __aenter__(self) -> 'StreamSubscriber':
        """Must be entered before use — this is where the lifecycle starts. """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Guarantees an explicit exit signal."""
        pass

    def __aiter__(self) -> AsyncIterator[Sample]:
        """Async iterator: blocks to obtain subsequent data."""
        return self

    @abstractmethod
    async def __anext__(self) -> Sample:
        """
        :raise StopAsyncIteration:
        """
        pass


class OutputBuffer(ABC):
    """
    A consume-then-use buffer of OutputItem.
    """

    @abstractmethod
    def close(self) -> None:
        """Close the buffer"""
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """Whether it is closed"""
        pass

    @abstractmethod
    def add_output(self, item: OutputItem) -> None:
        """Add an item; implementations must be thread-safe. """
        pass

    @abstractmethod
    def values(self) -> Iterable[OutputItem]:
        """Return all items as a thread-safe snapshot. """
        pass

    @abstractmethod
    def updated_at(self) -> float:
        """Timestamp of the last update"""
        pass


class Session(ABC):
    """
    The communication bus of the running MOSS matrix.

    Every component on the network talks through this bus. A session is the matrix's shared
    communication state: all cells of the same session scope share it, and no address constraint
    binds a session to a single cell.

    The bus carries these paths by default:
      - output: structured messages (OutputItem), suited to events and status notifications. It
        is the system's global one-way outward output protocol.
      - signal: Mindflow perception signals driving the three loops, used to drive the Ghost
        running inside MOSS. See Mindflow. It is the parallel signal input protocol.
      - file: a Session-level folder for file-level read/write communication.
      - stream: byte-stream pub/sub for real-time streaming data such as logos, with a
        user-defined protocol. In principle one ordered publisher, many receivers.
      - topic service: strongly typed broadcast over the available Topic protocol; an atomic
        n * m broadcast bus.
    """

    LOGOS_KEY = 'logos'
    """Key prefix of the logos stream. The full key comes from the stream key."""

    LOGOS_END = "\x00"
    """End-of-utterance marker (EOF) for the logos stream.

    The publisher emits it as a standalone delta when ``pub_logos(end=True)``; a consumer flushes
    the tail and renders an interval when it sees it. NUL never occurs in natural text or CTML,
    so it serves as a dedicated control sentinel (unlike ``\\n\\n``, which is a content value).
    """

    @property
    @abstractmethod
    def session_scope(self) -> str:
        """
        Session scope — the re-enterable composite identity of a session:
        ``mode-{mode}-ghost-{ghost}-network-{network}``.

        Processes sharing this scope share storage and the logos stream, and re-entering it on a
        later run re-attaches to the same storage. The network scope (a communication subspace)
        is deliberately not part of the key — it partitions transport, not storage.
        """
        pass

    @property
    @abstractmethod
    def run_id(self) -> str:
        """
        Run id of this process.

        Every Session instantiated in a process carries a different run id. It exists for
        cell-local records and logs only; nothing shared across processes keys on it — shared
        identity is ``session_scope``.
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
        meta = meta or InputSignalMeta()
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
        Service built on the Topic protocol.
        """
        pass

    @property
    @abstractmethod
    def qa(self) -> QAManager:
        """
        QA broadcast question/answer protocol — the cross-cell ask/answer bus.
        """
        pass

    @abstractmethod
    def output(self, role: str | Role, *messages: Message | str, log: str = '') -> None:
        """
        Output a message to the terminals sharing this moss session.
        Implementations must not block the thread.
        :param role: output role classification
        :param messages: message bodies; when there are none, ``log`` may describe the output alone
        :param log: one-line summary, shown in verbose scenarios
        """
        pass

    @abstractmethod
    def on_output(self, callback: Callable[[OutputItem], None]) -> None:
        """
        Listen for output callbacks carrying conversation items — e.g. to drive some rendering.
        """
        pass

    @abstractmethod
    def output_buffer(
            self,
            maxsize: int = 100,
    ) -> OutputBuffer:
        """
        Produce an OutputBuffer.
        """
        pass

    # ── stream protocol ──────────────────────────────

    @abstractmethod
    def is_running(self) -> bool:
        """Whether the session is running. """
        pass

    @abstractmethod
    def self_explain(self) -> str:
        """
        Self-explanation of the session: transport, key namespaces, stream conventions and so on.
        For debugging and runtime inspection.
        """
        pass

    @abstractmethod
    def sub_stream(
            self,
            relative_key: str,
            callback: Callable[[Sample], None],
    ) -> Callable[[], None]:
        """
        Subscribe to a byte stream. The callback receives the decoded payload; the returned
        handle owns the lifecycle. Callers align on a protocol through keys they define.

        :param relative_key: a Session-level key.
        :param callback: must be thread-safe.
        :return: a stop handle that cancels the subscription.
        """
        # A Session is shared across processes: processes talk using protocols they define
        # themselves, with bytes as the transport packet. The Session provides the communication
        # foundation; the default implementation is zenoh. Wildcards are supported in keys.
        pass

    @abstractmethod
    def pub_stream_delta(self, relative_key: str, delta: bytes) -> None:
        """
        Broadcast a payload onto the Session stream bus.
        ``relative_key`` is a Session-level key; the implementation converts it into a full path.
        """
        pass

    @abstractmethod
    def get_stream(
            self, relative_key: str, *, maxsize: int = 0,
    ) -> StreamSubscriber:
        """
        Obtain a byte stream lazily. ``maxsize=0`` means unbounded buffering. The caller consumes
        it with ``async for``.
        """
        # underlying layer
        pass

    @abstractmethod
    def stream_key_expr(self, relative_key: str) -> str:
        """Build the full stream key path. Subclasses may override."""
        # The session defines and exposes its own stream key implementation so it can be
        # inspected and understood.
        pass

    # ── logos stream ──────────────────────────────

    def pub_logos(
            self,
            *deltas: str,
            stream_id: str | None = None,
            end: bool = False,
    ) -> None:
        """
        Send logos fragments produced by the model (the ctml stream by default; see Mindflow)
        onto the bus.

        :param deltas: fragments of streaming data.
        :param stream_id: a stream id isolating different logos streams.
        :param end: mark the end of a logos utterance, sending the {LOGOS_END} EOF sentinel.

        Ordering is required by construction.
        """
        sid = stream_id or self.session_scope
        key = f"{self.LOGOS_KEY}/{sid}"
        for delta in deltas:
            self.pub_stream_delta(key, delta.encode('utf-8'))
        if end:
            self.pub_stream_delta(key, self.LOGOS_END.encode('utf-8'))

    async def get_logos(
            self, *, stream_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Obtain the broadcast stream under the agreed protocol.
        """
        sid = stream_id or self.session_scope
        stream = self.get_stream(f"{self.LOGOS_KEY}/{sid}")
        async with stream:
            async for delta in stream:
                yield delta.payload.decode('utf-8')

    # --- session storage spaces, isolating at different levels, usable as a file channel --- #

    @property
    @abstractmethod
    def storage(self) -> Storage:
        """
        Storage owned by this session scope.
        Files may serve as a communication channel when needed. Only meaningful inside the project.
        Conventionally at ``[ws]/runtime/sessions/[session-scope]``.
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """A session defines its own lifecycle so Matrix can govern it uniformly. """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
