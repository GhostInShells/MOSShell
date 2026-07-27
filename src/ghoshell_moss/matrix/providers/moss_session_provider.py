from typing import Iterable, Type, TYPE_CHECKING

from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts import LoggerItf
from ghoshell_moss.core.blueprint.project import Project

if TYPE_CHECKING:
    from ghoshell_moss.matrix.session.zenoh_session import ProjectZenohSession, MossSessionWithZenoh

__all__ = [
    'ProjectZenohSessionProvider',
]


class ProjectZenohSessionProvider(Provider[Session]):
    """
    make session instance from workspace
    """

    def singleton(self) -> bool:
        return True

    def contract(self) -> type:
        return Session

    def aliases(self) -> Iterable[Type]:
        from ghoshell_moss.matrix.session.zenoh_session import ProjectZenohSession, MossSessionWithZenoh
        yield MossSessionWithZenoh
        yield ProjectZenohSession

    def factory(self, con: IoCContainer) -> 'MossSessionWithZenoh':
        from ghoshell_moss.depends import depend_matrix
        depend_matrix()
        import zenoh
        from ghoshell_moss.matrix.session.zenoh_session import ProjectZenohSession
        logger = con.get(LoggerItf)
        project = con.force_fetch(Project)
        topic_service = con.force_fetch(TopicService)
        zenoh_session = con.force_fetch(zenoh.Session)

        return ProjectZenohSession(
            project=project,
            logger=logger,
            topic_service=topic_service,
            zenoh_session=zenoh_session,
        )
