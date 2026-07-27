from typing import Callable
from typing_extensions import Self

from ghoshell_moss.message import Message
from pathlib import Path

from ghoshell_moss.contracts import Storage, LocalStorage, get_moss_logger
from ghoshell_moss.contracts.cache import Cache
from ghoshell_moss.core.cache import SqliteCache
from ghoshell_moss.core.parameter import SessionParameterStore
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.parameter import ParameterStore
from ghoshell_moss.core.blueprint.project import Project
from ghoshell_moss.core.blueprint.session import (
    Session, Signal, Role, OutputBuffer, OutputItem, StreamSubscriber,
    Sample
)
from ghoshell_moss.tools.zenoh_helper import MatrixNamespace, MatrixEnvNamespace
from ghoshell_moss.depends import depend_matrix
from ghoshell_moss.message import unique_id
from ghoshell_moss.core.session.utils import SimpleOutputBuffer

depend_matrix()
from .zenoh_stream_subscriber import ZenohStreamSubscriber
from pydantic import BaseModel
import zenoh
import logging
import asyncio

__all__ = [
    'MossSessionWithZenoh',
    'SimpleOutputBuffer',
    'ProjectZenohSession',
]


class SessionMetadata(BaseModel):
    session_scope: str
    session_id: str
    cell_address: str
    parent_cell_address: str


class MossSessionWithZenoh(Session):
    """
    Session implementation for host
    """

    def __init__(
            self,
            *,
            session_scope: str,
            namespace: MatrixNamespace,
            zenoh_session: zenoh.Session,
            topic_service: TopicService,
            sessions_storage_dir: Path,
            sessions_tmp_storage_dir: Path,
            logger: logging.Logger | None = None,
            cell_address: str = '',
            parent_cell_address: str = '',
            session_id: str | None = None,
    ):
        """
        :param session_scope: Moss Matrix 运行时, 所有通讯都围绕同一个 session scope.
        :param logger: 日志模块.
        :param zenoh_session: 依赖 zenoh 通讯.
        :param topic_service: session 持有 topic service. 未来应该是 session 构建它.
        """
        self._namespace = namespace
        self._session_scope = session_scope
        self._session_id = session_id or unique_id()
        # 用于写入 session scope.
        self._metadata = SessionMetadata(
            session_scope=self._session_scope,
            session_id=self._session_id,
            cell_address=cell_address,
            parent_cell_address=parent_cell_address,
        )

        # 子类继承可重写.
        self._output_key_expr = self._namespace.output_ns
        self._input_signal_expr = self._namespace.signal_ns
        self._stream_key_expr_prefix = self._namespace.stream_ns
        self._received_signal_index: int = 0

        self._zenoh_session = zenoh_session
        if zenoh_session.is_closed():
            raise RuntimeError(f'HostSession receive Zenoh session but closed')

        self._output_sub = zenoh_session.declare_subscriber(self._output_key_expr, self._on_zenoh_output)
        self._input_sub = zenoh_session.declare_subscriber(self._input_signal_expr, self._on_zenoh_signal_input)
        self._logger = logger or get_moss_logger()
        self._log_prefix = f'<Session cls={self.__class__} scope={session_scope} id={self.session_id}>'

        # 注意内存泄漏.
        self._output_listeners: list[Callable[[OutputItem], None]] = []
        # 与生命周期绑定有限个. 这个方法没有解绑的机制. 要考虑未来支持一个最小生命周期 handler.
        self._on_signal_callbacks: list[Callable[[Signal], None]] = []
        self._topic_service = topic_service
        self._closing_event = ThreadSafeEvent()
        # --- lazy 懒启动 --- #
        self._cache: Cache | None = None
        self._parameters: ParameterStore | None = None

        self._sessions_storage_dir = sessions_storage_dir
        self._sessions_tmp_storage_dir = sessions_tmp_storage_dir
        self._session_tmp_storage: Storage | None = None
        self._session_scope_storage: Storage | None = None

    @classmethod
    def make_session_scope(cls, env: Environment) -> str:
        mode = cls._normalize(env.mode_name)
        ghost = cls._normalize(env.ghost_name)
        network = cls._normalize(env.network)
        return f"mode-{mode}-ghost-{ghost}-network-{network}"

    @classmethod
    def _normalize(cls, name: str) -> str:
        return (name.replace('.', '_').replace('\\', '_').
                replace(' ', '_').replace('/', '_'))

    @property
    def session_scope(self) -> str:
        return self._session_scope

    @property
    def storage(self) -> Storage:
        if self._session_scope_storage is None:
            self._session_scope_storage = self._make_session_storage(self._sessions_storage_dir)

        return self._session_scope_storage

    @property
    def tmp_storage(self) -> Storage:
        # tmp storage 应该要在每次运行完后删除.
        if self._session_tmp_storage is None:
            self._session_tmp_storage = self._make_session_storage(self._sessions_tmp_storage_dir)
        return self._session_tmp_storage

    def _session_storage_dir_name(self) -> str:
        return f"session-{self.session_scope}"

    def _make_session_storage(self, root: Path) -> Storage:
        storage_path = root / self._session_storage_dir_name()
        if not storage_path.exists():
            storage_path.mkdir(parents=True, exist_ok=True)
        storage = LocalStorage(storage_path)
        return storage

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def topics(self) -> TopicService:
        return self._topic_service

    @property
    def cache(self) -> Cache:
        if self._cache is None:
            db_path = Path(self.tmp_storage.abspath()) / 'cache.db'
            self._cache = SqliteCache(db_path)
        return self._cache

    @property
    def parameters(self) -> ParameterStore:
        self._check_running()
        if self._parameters is None:
            self._parameters = SessionParameterStore(self)
        return self._parameters

    def _check_running(self) -> None:
        if self._zenoh_session.is_closed():
            raise RuntimeError(f'HostSession is closed')

    def add_signal(self, signal: Signal) -> None:
        """向 session 总线发布信号。

        调用方负责控制发送频率。本方法不做限频——连续高频调用会直接打满 zenoh
        发布通道，淹没下游 subscriber 回调链。限频应在 Mindflow 的 signal
        ingestion 层实现，而非 transport 层。
        """
        self._check_running()
        js = signal.to_json()
        self._zenoh_session.put(self._input_signal_expr, js)

    def on_signal(self, callback: Callable[[Signal], None]) -> None:
        self._on_signal_callbacks.append(callback)

    def _on_zenoh_signal_input(self, sample: zenoh.Sample) -> None:
        if len(self._on_signal_callbacks) == 0:
            return None
        try:
            signal = Signal.model_validate_json(sample.payload.to_bytes())
            self._received_signal_index += 1
            # 在 session 内流转的都分配一个隐藏的 参数方便 debug. 没有额外性能开销.
            signal.metadata['_session_signal_index'] = self._received_signal_index
        except Exception as e:
            self._logger.error(
                f"%s failed to handle received signal sample %s: %s",
                self._log_prefix, sample.payload.to_string(), e,
            )
            return None
        # 回调感知接口.
        for callback in self._on_signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                self._logger.exception(
                    "%s failed to callback received signal on %s: %s",
                    self._log_prefix, callback, e
                )
        return None

    def output(self, role: str | Role, *messages: Message | str, log: str = '') -> None:
        item = OutputItem.new(role, *messages, log=log)
        js = item.model_dump_json(indent=0, ensure_ascii=False, exclude_none=True, exclude_defaults=True)
        self._zenoh_session.put(self._output_key_expr, js)

    def output_buffer(self, maxsize: int = 100) -> OutputBuffer:
        buffer = SimpleOutputBuffer(maxsize)

        def _output_add_to_buffer(item: OutputItem) -> None:
            nonlocal buffer
            if buffer.is_closed():
                return
            buffer.add_output(item)

        self.on_output(_output_add_to_buffer)
        return buffer

    def _on_zenoh_output(self, sample: zenoh.Sample) -> None:
        if len(self._output_listeners) == 0:
            return
        try:
            item = OutputItem.model_validate_json(sample.payload.to_bytes())
        except Exception as e:
            self._logger.error(
                "%s failed to send output %s: %s",
                self._log_prefix, sample.payload.to_string(), e,
            )
            item = OutputItem.new('error', Message.new().with_content("receive invalid output: %s" % e))
        for listener in self._output_listeners:
            try:
                listener(item)
            except Exception as e:
                self._logger.error(
                    "%s failed to send output %s: %s",
                    self._log_prefix, item.id, e,
                )

    def on_output(self, callback: Callable[[OutputItem], None]) -> None:
        self._output_listeners.append(callback)

    # ── stream 协议 ──────────────────────────────

    def is_running(self) -> bool:
        return not self._zenoh_session.is_closed()

    def self_explain(self) -> str:
        return (
            f"Session:"
            f"  scope: {self._session_scope}\n"
            f"  session_id: {self._session_id}\n"
            f"  transport: zenoh\n"
            f"  output key: {self._output_key_expr}\n"
            f"  signal key: {self._input_signal_expr}\n"
            f"  stream key prefix: {self._stream_key_expr_prefix}\n"
        )

    def sub_stream(
            self, relative_key: str, callback: Callable[[Sample], None],
    ) -> Callable[[], None]:
        self._check_running()

        stream_key = self.stream_key_expr(relative_key)

        def _on_sample(_sample: zenoh.Sample) -> None:
            if not self.is_running():
                return

            _relative_key = self._parse_stream_relative_key(str(_sample.key_expr))
            if _relative_key is None:
                self._logger.warning(
                    "%s stream subscriber received sample with unexpected key: %s (prefix: %s)",
                    self._log_prefix, str(_sample.key_expr), self._stream_key_expr_prefix,
                )
                return
            _moss_sample = Sample(
                relative_key=_relative_key,
                payload=_sample.payload.to_bytes(),
            )
            callback(_moss_sample)

        sub = self._zenoh_session.declare_subscriber(stream_key, _on_sample)

        def _release():
            nonlocal sub
            if not self.is_running() or self._zenoh_session.is_closed():
                return
            try:
                sub.undeclare()
            except Exception:
                return

        return _release

    def _parse_stream_relative_key(self, sample_key: str) -> str | None:
        if sample_key.startswith(self._stream_key_expr_prefix):
            return sample_key[len(self._stream_key_expr_prefix) + 1:]
        return None

    def pub_stream_delta(self, relative_key: str, delta: bytes) -> None:
        self._check_running()
        self._zenoh_session.put(self.stream_key_expr(relative_key), delta)

    def stream_key_expr(self, relative_key: str) -> str:
        return "/".join([
            self._stream_key_expr_prefix,
            relative_key.strip('/')
        ])

    def get_stream(self, relative_key: str, *, maxsize: int = 0) -> StreamSubscriber:
        return ZenohStreamSubscriber(
            key_expr_prefix=self._stream_key_expr_prefix,
            relative_key=relative_key,
            maxsize=maxsize,
            zenoh_session=self._zenoh_session,
            session_stop_event=self._closing_event,
        )

    async def __aenter__(self) -> Self:
        self._logger.info("%s session started", self._log_prefix)
        # Eager-init parameter store in thread pool — SQLite WAL + Zenoh
        # sub are synchronous and would block the event loop.
        await asyncio.to_thread(lambda: self.parameters)
        # 记录 jsonl
        await asyncio.to_thread(self.storage.append_model, "sessions", self._metadata)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closing_event.set()
        if self._parameters is not None:
            self._parameters.close()
        self._logger.info("%s session closed", self._log_prefix)


class ProjectZenohSession(MossSessionWithZenoh):

    def __init__(
            self,
            *,
            project: Project,
            zenoh_session: zenoh.Session,
            topic_service: TopicService,
            logger: logging.Logger | None = None,
    ):
        session_scope = MossSessionWithZenoh.make_session_scope(project.env)
        session_id = project.env.session_id
        namespace = MatrixEnvNamespace(project.env)

        super().__init__(
            session_scope=session_scope,
            session_id=session_id,
            namespace=namespace,
            zenoh_session=zenoh_session,
            topic_service=topic_service,
            logger=logger,
            cell_address=project.env.this_cell_address,
            parent_cell_address=project.env.parent_cell_address,
            sessions_storage_dir=project.sessions_dir,
            sessions_tmp_storage_dir=project.tmp
        )
