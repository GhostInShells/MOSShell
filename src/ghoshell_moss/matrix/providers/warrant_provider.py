from typing import Iterable, Type

from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.warrant import Warrant
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_container import Provider, IoCContainer, INSTANCE

__all__ = ['SessionWarrantProvider']


class SessionWarrantProvider(Provider[Warrant]):
    """Provide a Warrant wired to the cell's Session, split by host/non-host.

    host cell -> SessionWarrant (写 storage 模式, owns storage).
    non-host cell -> TopicWarrant (topic 模式, cache + write-request topic, v8).

    Warrant is an optional capability (KD7): consumers use ``con.get(Warrant)``
    and allow when absent.  If Matrix is absent we fall back to host (KD7
    fail-open) — the old write-storage behaviour.
    """

    def singleton(self) -> bool:
        return True

    def contract(self) -> type:
        return Warrant

    def aliases(self) -> Iterable[Type]:
        from ghoshell_moss.matrix.warrant import SessionWarrant, TopicWarrant
        yield SessionWarrant
        yield TopicWarrant

    def factory(self, con: IoCContainer) -> INSTANCE:
        from ghoshell_moss.matrix.warrant import SessionWarrant, TopicWarrant
        session = con.force_fetch(Session)
        matrix = con.get(Matrix)
        is_host = matrix.is_host if matrix is not None else True
        if not is_host:
            return TopicWarrant(session=session)
        states_dir = session.storage.sub_storage("warrants").abspath()
        return SessionWarrant(session=session, states_dir=states_dir)
