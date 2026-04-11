from typing import Callable, Iterable, Type

from ghoshell_moss.contracts import Storage, LoggerItf, Workspace
from ghoshell_moss.host.abcd import ConversationItem
from ghoshell_moss.host.abcd.session import Session
from ghoshell_container import IoCContainer, Provider
from threading import Event
import zenoh
import orjson

__all__ = [
    'HostSession',
    'WorkspaceSessionProvider',
]


class HostSession(Session):
    """
    Session implementation for host
    """

    def __init__(
            self,
            session_id: str,
            session_storage: Storage,
            logger: LoggerItf,
            zenoh_session: zenoh.Session,
    ):
        self._session_id = session_id
        self._output_key_expr = f"moss/{session_id}/outputs"
        self._session_storage = session_storage
        self._closing_event = Event()
        self._output_listeners: list[Callable[[ConversationItem], None]] = []
        self._zenoh_session = zenoh_session
        if zenoh_session.is_closed():
            raise RuntimeError(f'HostSession receive Zenoh session but closed')
        _ = zenoh_session.declare_subscriber(self._output_key_expr, self._on_zenoh_output)
        self._logger = logger
        self._log_prefix = f'<Session cls={self.__class__} id={session_id}>'

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def storage(self) -> Storage:
        return self._session_storage

    def output(self, *items: ConversationItem) -> None:
        if self._zenoh_session.is_closed():
            return
        for item in items:
            js = item.model_dump_json(indent=0, ensure_ascii=False, exclude_defaults=True, exclude_none=True)
            self._zenoh_session.put(self._output_key_expr, js)

    def _on_zenoh_output(self, sample: zenoh.Sample) -> None:
        if len(self._output_listeners) == 0:
            return
        try:
            data = orjson.loads(sample.payload.to_bytes())
            item = ConversationItem(**data)
            for listener in self._output_listeners:
                try:
                    listener(item)
                except Exception as e:
                    self._logger.error(
                        "%s failed to send output %s: %s",
                        self._log_prefix, item.id, e,
                    )
        except Exception as e:
            self._logger.error(
                "%s failed to send output %s: %s",
                self._log_prefix, sample.payload.to_string(), e,
            )

    def on_output(self, callback: Callable[[ConversationItem], None]) -> None:
        self._output_listeners.append(callback)


class WorkspaceSessionProvider(Provider[Session]):
    """
    make session instance from workspace
    """

    def __init__(
            self,
            session_id: str,
            *,
            session_path: str = 'sessions',
            session_id_prefix: str = 'session-',
    ):
        self._session_id = session_id
        self._session_path = session_path
        self._session_id_prefix = session_id_prefix

    def singleton(self) -> bool:
        return True

    def contract(self) -> type:
        return Session

    def aliases(self) -> Iterable[Type]:
        yield HostSession

    def factory(self, con: IoCContainer) -> HostSession:
        ws = con.force_fetch(Workspace)
        zenoh_session = con.force_fetch(zenoh.Session)
        logger = con.get(LoggerItf)
        session_storage_path = self._session_id_prefix + self._session_id
        storage = ws.runtime().sub_storage('session').sub_storage(session_storage_path)
        return HostSession(
            session_id=self._session_id,
            session_storage=storage,
            logger=logger,
            zenoh_session=zenoh_session,
        )
