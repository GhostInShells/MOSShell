from typing import TYPE_CHECKING
from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.ghost import GhostMeta
from ghoshell_moss.core.blueprint.mindflow import NucleusFactory
from ._runtime import Atom

__all__ = ["AtomMeta"]


class AtomMeta(GhostMeta):
    """
    Atom 原型的基本配置.
    """

    def __init__(
            self,
            name: str,
            description: str,
            nuclei_metas: list[NucleusFactory],
    ):
        self._name = name
        self._description = description
        self._nuclei_metas = nuclei_metas

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def nuclei_manifests(self) -> list[NucleusFactory]:
        return self._nuclei_metas

    def factory(self, container: IoCContainer) -> "Ghost":
        pass
