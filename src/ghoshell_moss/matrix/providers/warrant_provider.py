from typing import Iterable, Type

from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.warrant import Warrant
from ghoshell_container import Provider, IoCContainer, INSTANCE

__all__ = ['SessionWarrantProvider']


class SessionWarrantProvider(Provider[Warrant]):
    """Provide a SessionWarrant wired to the cell's Session (storage + QA).

    Warrant is an optional capability (KD7): consumers use ``con.get(Warrant)``
    and allow when absent.  This provider only wires the concrete instance —
    fail-open is the consumer's contract, not the provider's.
    """

    def singleton(self) -> bool:
        return True

    def contract(self) -> type:
        return Warrant

    def aliases(self) -> Iterable[Type]:
        from ghoshell_moss.matrix.warrant import SessionWarrant
        yield SessionWarrant

    def factory(self, con: IoCContainer) -> INSTANCE:
        from ghoshell_moss.matrix.warrant import SessionWarrant
        session = con.force_fetch(Session)
        states_dir = session.storage.sub_storage("warrants").abspath()
        return SessionWarrant(session=session, states_dir=states_dir)
