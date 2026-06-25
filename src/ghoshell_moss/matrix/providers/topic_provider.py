from typing import Iterable, Type
from ghoshell_moss.matrix.topics.zenoh_topics import ZenohTopicService
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer, INSTANCE

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.tools.zenoh_helper import MatrixEnvNamespace
from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

__all__ = ['ZenohTopicServiceProvider']


class ZenohTopicServiceProvider(Provider[TopicService]):
    """
    zenoh topic service provider
    """

    def singleton(self) -> bool:
        return True

    def aliases(self) -> Iterable[Type]:
        yield ZenohTopicService

    def factory(self, con: IoCContainer) -> INSTANCE:
        env = con.force_fetch(Environment)
        session = con.force_fetch(zenoh.Session)
        logger = con.get(LoggerItf)
        namespace = MatrixEnvNamespace(env)

        return ZenohTopicService(
            network_scope=env.network_scope,
            session=session,
            address=env.this_cell_address,
            namespace=namespace,
            logger=logger,
        )
