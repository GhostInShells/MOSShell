from typing import Iterable, Type

from ghoshell_moss.contracts import LoggerItf, Workspace
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_container import IoCContainer, Provider
from ghoshell_moss.depends import depend_zenoh
from ghoshell_moss.core.blueprint.environment import Environment

depend_zenoh()
import zenoh
from ghoshell_moss.matrix.session.zenoh_session import MossSessionWithZenoh

__all__ = [
    'HostSessionProvider',
]

# todo: 删除
class HostSessionProvider(Provider[Session]):
    """
    make session instance from workspace
    """

    def __init__(
            self,
            *,
            session_path: str = 'sessions',
            session_tmp_path: str = 'sessions-tmp',
    ):
        self._session_path = session_path
        self._session_tmp_path = session_tmp_path

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
        env = con.force_fetch(Environment)
        session_scope = env.network_scope
        session_id = env.session_id
        # session 的根目录.
        session_root_storage = ws.runtime().sub_storage(self._session_path)
        # session 的临时信息的根目录.
        session_tmp_root_storage = ws.runtime().sub_storage(self._session_tmp_path)
        topics = con.force_fetch(TopicService)
        session = MossSessionWithZenoh(
            session_scope=session_scope,
            session_root_storage=session_root_storage,
            session_tmp_root_storage=session_tmp_root_storage,
            logger=logger,
            zenoh_session=zenoh_session,
            topic_service=topics,
            session_id=session_id,
        )
        return session
