from pathlib import Path

from ghoshell_container import IoCContainer
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.host import MossSystemPrompter
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.concepts.shell import MOSShell

from .nucleus import DoloresEgoNucleusMeta

__all__ = ["DoloresMeta"]


class DoloresMeta(GhostMeta):
    """Dolores — the second Ghost prototype (named after the character in Westworld).

    Unlike Atom's linear in-memory history, Dolores adds Memento-persisted trajectory, a ghost
    reflexivity channel, interleaved thinking, independent thinking modules, and model self-awareness,
    iterating as the carrier for this repo's own ghost instance.
    """

    VERSION = "dev_2"

    def __init__(
        self,
        name: str = "dolores",
        description: str = (
            "Dolores — the second Ghost prototype. DSH is the reasoning core; MOSS keeps memory "
            "(Memento) / execution (CTML channels) / perception (audio/vision)."
        ),
        nuclei_metas: list[NucleusMeta] | None = None,
    ):
        self._name = name
        self._description = description
        # Default to the ego self-wake nucleus; an explicit nuclei_metas fully replaces it.
        self._nuclei_metas = (
            nuclei_metas if nuclei_metas is not None else [DoloresEgoNucleusMeta()]
        )

    # ── GhostMeta ABC ──────────────────────────────

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def nuclei_metas(self) -> list[NucleusMeta]:
        return self._nuclei_metas

    # ── instruction sections (derived structurally, no hardcoded prompt) ──────────

    def prototype_instruction(self) -> str:
        """Prototype meta info — model + version, derived from structured meta."""
        # todo: add behavior logic — model output is parsed as CTML to drive the body; channels
        # control/interact. Extend here as the prototype iterates.
        return "\n".join([
            f"prototype: {self.prototype()}",
            f"version: {self.VERSION}",
        ])

    def identity_instruction(self) -> str:
        """Identity description — name + description, derived from structured meta."""
        return "\n".join([
            f"name: {self.name()}",
            f"description: {self.description()}",
        ])

    # ── stubs / dsh home ────────────────────────────

    @classmethod
    def stubs_dir(cls) -> Path:
        """Source dir of the ghost-home skeleton (GROUND.md / .dolores.yml / .gitignore)."""
        return Path(__file__).parent / "stubs"

    @classmethod
    def dsh_stubs_dir(cls) -> Path:
        """Source dir of the dsh-home skeleton."""
        return Path(__file__).parent / "dsh_stubs"

    @classmethod
    def dsh_plugin_stub(cls) -> Path:
        """Source file of the dsh ghost plugin (a standalone stub)."""
        return Path(__file__).parent / "dsh_plugin" / "moss-dolores-ghost-plugin.ts"

    # ── factory ─────────────────────────────────────

    def factory(self, container: IoCContainer) -> Ghost:
        """Fetch paths/dependencies only — no side effects.

        Stub sync (file IO + session.output) and dsh startup (matrix.processes) are deferred to
        Dolores.__aenter__, so tests can pass a tmp home and construct without touching the disk.
        """
        from ._runtime import Dolores

        home: Path | None = None
        session: Session | None = None
        matrix: Matrix | None = None
        shell: MOSShell | None = None
        base_instruction: str | None = None
        if container is not None:
            matrix = container.get(Matrix)
            if matrix is not None:
                home = matrix.ghost_home
            session = container.get(Session)
            shell = container.get(MOSShell)
            prompter = container.get(MossSystemPrompter)
            if prompter is not None:
                base_instruction = prompter.base_instruction()
        return Dolores(
            meta=self,
            home=home,
            session=session,
            matrix=matrix,
            shell=shell,
            base_instruction=base_instruction,
        )
