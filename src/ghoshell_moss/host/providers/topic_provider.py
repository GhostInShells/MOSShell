from typing import Iterable, Type

from ghoshell_moss.topic.zenoh_topics import ZenohTopicService
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer, INSTANCE
import zenoh

__all__ = ['ZenohTopicServiceProvider']


class ZenohTopicServiceProvider(Provider[TopicService]):
    """
    zenoh topic service provider
    """

    def __init__(
            self,
            *,
            session_id: str,
            node_name: str,
    ):
        self.session_id = session_id
        self.node_name = node_name

    def singleton(self) -> bool:
        return True

    def aliases(self) -> Iterable[Type]:
        yield ZenohTopicService

    def factory(self, con: IoCContainer) -> INSTANCE:
        session = con.force_fetch(zenoh.Session)
        logger = con.get(LoggerItf)

        return ZenohTopicService(
            session_id=self.session_id,
            session=session,
            node_name=self.node_name,
            logger=logger,
        )
