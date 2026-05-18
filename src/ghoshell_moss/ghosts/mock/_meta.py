from ghoshell_container import IoCContainer, Contracts, Provider
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta

__all__ = ["MockGhostMeta"]


class MockGhostMeta(GhostMeta):
    """Mock GhostMeta — 所有返回值可随时替换，用于测试 GhostRuntime 体系."""

    def __init__(
        self,
        name: str = "mock",
        description: str = "Mock ghost for testing GhostRuntime without real model calls.",
        nuclei_metas: list[NucleusMeta] | None = None,
        providers: list[Provider] | None = None,
    ):
        self._name = name
        self._description = description
        self._nuclei_metas = nuclei_metas or []
        self._providers = providers or []

    # ── GhostMeta ABC ──────────────────────────────

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def nuclei_metas(self) -> list[NucleusMeta]:
        return self._nuclei_metas

    def providers(self) -> list[Provider]:
        return self._providers

    def contracts(self) -> Contracts:
        return Contracts([])

    def factory(self, container: IoCContainer) -> Ghost:
        from ._runtime import MockGhost

        return MockGhost(meta=self)

    # ── helpers ─────────────────────────────────────

    def set_name(self, name: str) -> None:
        self._name = name

    def set_description(self, description: str) -> None:
        self._description = description

    def set_nuclei_metas(self, nuclei_metas: list[NucleusMeta]) -> None:
        self._nuclei_metas = nuclei_metas

    def set_providers(self, providers: list[Provider]) -> None:
        self._providers = providers
