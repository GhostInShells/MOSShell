from typing import TYPE_CHECKING
from typing_extensions import Self

from ghoshell_moss.core.blueprint.conversation import ModelContext
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import Logos
from ghoshell_container import IoCContainer

if TYPE_CHECKING:
    from ._meta import AtomMeta


class Atom(Ghost):

    def __init__(
            self,
            meta: "AtomMeta",
    ):
        self._meta = meta

    @classmethod
    def factory(cls, meta: "AtomMeta", container: IoCContainer) -> Self:
        pass

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    def system_prompt(self) -> str:
        pass

    def articulate(self, context: ModelContext) -> Logos:
        pass

    async def __aenter__(self) -> Self:
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
