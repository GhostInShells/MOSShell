from typing import Iterable, Type

from ghoshell_moss.contracts import LoggerItf, Workspace
from ghoshell_moss.core.concepts.session import Session
from ghoshell_container import IoCContainer, Provider
from ghoshell_moss.depends import depend_zenoh
from ghoshell_moss.host.abcd.environment import Environment

depend_zenoh()
import zenoh
from ghoshell_moss.core.session.zenoh_session import MossSessionWithZenoh

__all__ = [
    'WorkspaceSessionProvider',
]


class WorkspaceSessionProvider(Provider[Session]):
    """
    make session instance from workspace
    """

    def __init__(
            self,
            session_scope: str | None = None,
            *,
            session_path: str = 'sessions',
            session_id_prefix: str = 'session-',
    ):
        self._session_scope = session_scope
        self._session_path = session_path
        self._session_id_prefix = session_id_prefix

    def singleton(self) -> bool:
        return True

    def contract(self) -> type:
        return Session

    def aliases(self) -> Iterable[Type]:
        yield MossSessionWithZenoh

    def factory(self, con: IoCContainer) -> MossSessionWithZenoh:
        ws = con.force_fetch(Workspace)
        zenoh_session = con.force_fetch(zenoh.Session)
        logger = con.get(LoggerItf)
        session_scope = self._session_scope
        if session_scope is None:
            env = con.force_fetch(Environment)
            session_scope = env.session_scope
        session_storage_path = self._session_id_prefix + session_scope
        storage = ws.runtime().sub_storage('session').sub_storage(session_storage_path)
        session = MossSessionWithZenoh(
            session_scope=self._session_scope,
            session_storage=storage,
            logger=logger,
            zenoh_session=zenoh_session,
        )
        # always clear during the container shutdown.
        con.add_shutdown(session.clear)
        return session
