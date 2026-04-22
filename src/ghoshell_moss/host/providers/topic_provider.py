from typing import Iterable, Type
from ghoshell_moss.core.topic.zenoh_topics import ZenohTopicService
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer, INSTANCE

from ghoshell_moss.host.abcd import Matrix
from ghoshell_moss.host.abcd.environment import Environment
from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

__all__ = ['ZenohTopicServiceProvider']


class ZenohTopicServiceProvider(Provider[TopicService]):
    """
    zenoh topic service provider
    """

    def __init__(
            self,
            *,
            session_scope: str = '',
            cell_address: str = '',
    ):
        self.session_scope = session_scope
        self.cell_address = cell_address

    def singleton(self) -> bool:
        return True

    def aliases(self) -> Iterable[Type]:
        yield ZenohTopicService

    def factory(self, con: IoCContainer) -> INSTANCE:
        session_scope = self.session_scope
        cell_address = self.cell_address
        if not session_scope:
            env = con.force_fetch(Environment)
            session_scope = env.session_scope
        if not cell_address:
            matrix = con.force_fetch(Matrix)
            cell_address = matrix.this.address

        session = con.force_fetch(zenoh.Session)
        logger = con.get(LoggerItf)

        return ZenohTopicService(
            session_scope=session_scope,
            session=session,
            address=cell_address,
            logger=logger,
        )
