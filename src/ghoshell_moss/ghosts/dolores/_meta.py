from dataclasses import dataclass, field
from pathlib import Path

from ghoshell_container import IoCContainer
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.host import MossSystemPrompter
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.concepts.shell import MOSShell

from .nucleus import DoloresEgoNucleusMeta

__all__ = ["DoloresMeta", "DoloresProviderSpec"]


@dataclass(frozen=True)
class DoloresProviderSpec:
    """A pi-ai provider declaration — the shape Dolores projects into dsh at **seed time**.

    Every field is a *name* (env var, route key), never a secret. The provider block is written
    once into ``<ghost_home>/.dsh/profiles/web/cordis.patch.yml`` on first boot; after that, the
    ghost home owns the file and the Python declaration is no longer the source of truth.

    - ``route`` — pi-ai provider route id (``anthropic`` / ``openai`` / ``deepseek`` …).
    - ``api_key_env`` — process env var that holds the credential at runtime (``apiKeyEnv``).
    - ``base_url_env`` — process env var holding a base-URL override. At seed time Dolores reads
      this env var and writes the resolved URL literally into the (gitignored) patch file; the
      meta carries only the *name*, so repo-visible code never leaks a vendor endpoint.
    - ``models`` — optional explicit catalog override.
    - ``display_name`` — optional label.
    """

    route: str
    api_key_env: str
    base_url_env: str | None = None
    models: list[str] | None = None
    display_name: str | None = None


class DoloresMeta(GhostMeta):
    """Dolores — the second Ghost prototype (named after the character in Westworld).

    Unlike Atom's linear in-memory history, Dolores adds Memento-persisted trajectory, a ghost
    reflexivity channel, interleaved thinking, independent thinking modules, and model self-awareness,
    iterating as the carrier for this repo's own ghost instance.
    """

    VERSION = "0.1.0-dev3"

    def __init__(
            self,
            name: str = "dolores",
            description: str = (
                    "Dolores — the second Ghost prototype. DSH is the reasoning core; MOSS keeps memory "
                    "(Memento) / execution (CTML channels) / perception (audio/vision)."
            ),
            identity: str | None = None,
            nuclei_metas: list[NucleusMeta] | None = None,
            providers: list[DoloresProviderSpec] | None = None,
            default_model: tuple[str, str] | None = None,
    ):
        self._name = name
        self._description = description
        # optional one-line social identity ("how others know it"); None means no identity line.
        self._identity = identity
        # Default to the ego self-wake nucleus; an explicit nuclei_metas fully replaces it.
        self._nuclei_metas = (
            nuclei_metas if nuclei_metas is not None else [DoloresEgoNucleusMeta()]
        )
        # Per-ghost dsh provider seed — projected **once** into ghost_home's cordis.patch.yml on
        # first boot. After the ghost home is materialized the patch file belongs to the ghost
        # (and lives inside the gitignored ghost_home), so later edits here do *not* chase the
        # old home. Empty list = don't touch the patch's provider section.
        self._providers: list[DoloresProviderSpec] = list(providers or [])
        # Default (provider_route, model_id) — seeded **once** into settings.yaml's
        # ``agent-default-model`` section on first boot. None = leave settings.yaml alone and let
        # dsh fall back to its own default. The ghost home can change this value after seeding.
        self._default_model: tuple[str, str] | None = default_model

    @property
    def llm_providers(self) -> list[DoloresProviderSpec]:
        # Named ``llm_providers`` (not ``providers``) because GhostMeta.providers() is the
        # IoC-provider contract — a different concept. Dolores declares pi-ai routes here.
        return list(self._providers)

    @property
    def default_model(self) -> tuple[str, str] | None:
        return self._default_model

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
            "# Ghost Prototype",
            f"prototype: {self.prototype()}",
            f"version: {self.VERSION}",
        ])

    def identity_instruction(self) -> str:
        """Identity section — name + optional identity line + technical description."""
        lines = [f"name: {self.name()}"]
        if self._identity:
            lines.append(f"identity: {self._identity}")
        lines.append(f"description: {self.description()}")
        return "# Your Identity\n" + "\n".join(lines)

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
        base_instruction: str | None = None
        if container is None:
            raise RuntimeError(f"IoCContainer {container} is invalid")
        matrix = container.get(Matrix)
        if matrix is not None:
            home = matrix.ghost_home
        session = container.get(Session)
        shell = container.get(MOSShell)
        prompter = container.get(MossSystemPrompter)
        return Dolores(
            meta=self,
            moss_prompter=prompter,
            home=home,
            session=session,
            matrix=matrix,
            shell=shell,
        )
