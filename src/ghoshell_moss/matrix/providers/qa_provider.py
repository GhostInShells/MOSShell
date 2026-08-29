from typing import Iterable, Type
from ghoshell_moss.core.concepts.qa import QAManager
from ghoshell_moss.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer, INSTANCE

from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.matrix.zenoh_helper import MatrixEnvNamespace
from ghoshell_moss.depends import depend_matrix

depend_matrix()
import zenoh

__all__ = ['ZenohQAManagerProvider']


class ZenohQAManagerProvider(Provider[QAManager]):
    """Provide ZenohQAManager wired to the cell's zenoh session and namespace."""

    def singleton(self) -> bool:
        return True

    def aliases(self) -> Iterable[Type]:
        from ghoshell_moss.matrix.qa.zenoh_qa import ZenohQAManager
        yield ZenohQAManager

    def factory(self, con: IoCContainer) -> INSTANCE:
        from ghoshell_moss.matrix.qa.zenoh_qa import ZenohQAManager
        env = con.force_fetch(Environment)
        session = con.force_fetch(zenoh.Session)
        logger = con.get(LoggerItf)
        namespace = MatrixEnvNamespace(env)
        prefix = f"{namespace}/qa"

        return ZenohQAManager(
            issuer=env.this_cell_address,
            prefix=prefix,
            session=session,
            logger=logger,
        )
