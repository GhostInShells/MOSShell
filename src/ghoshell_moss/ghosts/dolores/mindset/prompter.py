"""Prompt manifest — the ghost's file-based prompt registry.

A prompt is addressed by a dotted key (``a.b.c``) and stored as a markdown file per version
under ``<ghost_home>/.dolores/prompts/<a-b-c>/<version>.md``. The manifest
(``PromptsConfig``) is the index: which keys exist, their slots, their default version. The
active version is a pointer in the same config; creating a version never activates it.

Key / path bijection: the key's dots become hyphens, one-to-one. Segments are
``[A-Za-z0-9_]`` with no hyphens inside, so the mapping cannot collide. A version is the
file stem and never contains a slash.

Reads are strict about the manifest (an unknown key is an error) and about slots (declared
slots must be provided; undeclared placeholders are an error). Writes go through the manager
only — callers never receive a writable path.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from ghoshell_container import INSTANCE, IoCContainer, Provider
from pydantic import BaseModel, Field, field_validator, model_validator

from ghoshell_moss.contracts import ConfigType
from ghoshell_moss.core.blueprint.ghost import GhostHome
from ghoshell_moss.ghosts.dolores.configs import DoloresConfigStore
from ghoshell_moss.ghosts.dolores.constants import (
    DOLORES_PROMPTS_DIRECTORY,
    PROMPT_MEMENTO_CHAT,
    PROMPT_MEMENTO_NOTE,
)

__all__ = [
    "PromptError",
    "PromptManager",
    "PromptManagerProvider",
    "PromptPathError",
    "PromptSpec",
    "PromptVersion",
    "PromptVersionNotFoundError",
    "PromptsConfig",
    "SlotMismatchError",
    "UnknownPromptError",
    "default_specs",
]

_KEY_SEGMENT = re.compile(r"[A-Za-z0-9_]+")
_SLOT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SLOT = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class PromptError(RuntimeError):
    """Base class for prompt-registry errors — a caller may catch it and render a hint."""


class UnknownPromptError(PromptError):
    """The key is not in the manifest."""


class PromptVersionNotFoundError(PromptError):
    """The key exists, but the requested version has no file."""


class SlotMismatchError(PromptError):
    """The declared slots and the render values/content disagree."""


class PromptPathError(PromptError):
    """A key/version would leave the prompts root (or is otherwise unsafe)."""


def _validate_key(key: str) -> str:
    segments = key.split(".")
    if not key or any(
        not segment or _KEY_SEGMENT.fullmatch(segment) is None for segment in segments
    ):
        raise ValueError(f"prompt key must be dotted segments of [A-Za-z0-9_], got {key!r}")
    return key


def _validate_version(version: str) -> str:
    if not version or _VERSION.fullmatch(version) is None:
        raise ValueError(
            f"prompt version must match [A-Za-z0-9][A-Za-z0-9._-]*, got {version!r}"
        )
    return version


def _key_to_dir_name(key: str) -> str:
    return key.replace(".", "-")


def _atomic_write(path: Path, content: str) -> None:
    """Write text through a same-directory temp file — a crashed write never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class PromptSpec(BaseModel):
    """One manifest entry — the key's identity and its contract."""

    key: str = Field(description="Dotted key, e.g. `memento.note` (dsh-compatible).")
    description: str = Field(default="", description="What this prompt is for.")
    slots: list[str] = Field(
        default_factory=list,
        description="`{{name}}` placeholders the caller must provide; declared, not inferred.",
    )
    default_version: str = Field(
        default="v1",
        description="Version used when `active` has no override for this key.",
    )

    @field_validator("key")
    @classmethod
    def _key_shape(cls, value: str) -> str:
        return _validate_key(value)

    @field_validator("default_version")
    @classmethod
    def _version_shape(cls, value: str) -> str:
        return _validate_version(value)

    @field_validator("slots")
    @classmethod
    def _slots_are_unique_names(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        for value in values:
            if _SLOT_NAME.fullmatch(value) is None:
                raise ValueError(f"slot name must match [A-Za-z_][A-Za-z0-9_]*, got {value!r}")
            if value in seen:
                raise ValueError(f"duplicate slot {value!r}")
            seen.add(value)
        return values


class PromptVersion(BaseModel):
    """One loaded prompt file — the content plus where it came from."""

    key: str = Field(description="The manifest key.")
    version: str = Field(description="The version (the file stem).")
    content: str = Field(description="The markdown body, as stored.")
    path: Path = Field(description="The absolute file path — read-only inspection.")


class PromptsConfig(ConfigType):
    """The manifest: which prompts exist, and which version is active."""

    specs: list[PromptSpec] = Field(default_factory=list)
    active: dict[str, str] = Field(
        default_factory=dict,
        description="key -> active version; an absent key falls back to the spec's default_version.",
    )

    @classmethod
    def conf_name(cls) -> str:
        return "prompts"

    @model_validator(mode="after")
    def _manifest_is_consistent(self) -> PromptsConfig:
        seen: set[str] = set()
        for spec in self.specs:
            if spec.key in seen:
                raise ValueError(f"duplicate prompt key {spec.key!r}")
            seen.add(spec.key)
        for key, version in self.active.items():
            _validate_key(key)
            _validate_version(version)
            if key not in seen:
                raise ValueError(f"active prompt {key!r} has no manifest spec")
        return self

    def spec(self, key: str) -> PromptSpec | None:
        for spec in self.specs:
            if spec.key == key:
                return spec
        return None


class PromptManager:
    """The only code surface over the prompt registry.

    The manager computes every path from the key and version; it never hands out a writable
    path. Callers create versions, activate one, and render through it. A key that is not in
    the manifest is an error at every entry point.
    """

    def __init__(
        self,
        store: DoloresConfigStore,
        prompts_root: Path,
        *,
        defaults: list[PromptSpec] | None = None,
    ) -> None:
        self._store = store
        self._root = Path(prompts_root)
        self._config = store.get_or_create(PromptsConfig(specs=defaults or []))

    # -- reads --------------------------------------------------------- #

    @property
    def config(self) -> PromptsConfig:
        return self._config

    @property
    def root(self) -> Path:
        """The prompts root — a read-only view for surfaces and tests."""
        return self._root

    def specs(self) -> list[PromptSpec]:
        return list(self._config.specs)

    def spec(self, key: str) -> PromptSpec:
        spec = self._config.spec(key)
        if spec is None:
            raise UnknownPromptError(f"prompt {key!r} is not in the manifest")
        return spec

    def path_of(self, key: str, version: str) -> Path:
        """The file for one version — computed and contained; never a write API."""
        _validate_key(key)
        _validate_version(version)
        path = self._root / _key_to_dir_name(key) / f"{version}.md"
        root = self._root.resolve()
        if not path.resolve().is_relative_to(root):
            raise PromptPathError(f"prompt path escapes the prompts root: {path}")
        return path

    def versions(self, key: str) -> list[str]:
        """Every version file for one key, sorted; empty when the key has none yet."""
        self.spec(key)
        directory = self._root / _key_to_dir_name(key)
        if not directory.is_dir():
            return []
        return sorted(
            path.stem for path in directory.glob("*.md")
            if path.is_file() and _VERSION.fullmatch(path.stem) is not None
        )

    def get(self, key: str) -> PromptVersion:
        """The active version's content (or the spec's default when no override)."""
        spec = self.spec(key)
        version = self._config.active.get(key, spec.default_version)
        path = self.path_of(key, version)
        if not path.is_file():
            raise PromptVersionNotFoundError(
                f"prompt {key!r} has no version {version!r} (expected {path})"
            )
        return PromptVersion(
            key=key, version=version, content=path.read_text(encoding="utf-8"), path=path,
        )

    def render(self, key: str, **values: str) -> str:
        """Render the active prompt; declared slots must be provided, placeholders declared.

        Extra values are ignored (a caller may pass a wider context); a declared slot the
        content does not use is allowed (the spec is a superset).
        """
        spec = self.spec(key)
        prompt = self.get(key)
        declared = set(spec.slots)
        missing = declared - set(values)
        if missing:
            raise SlotMismatchError(f"prompt {key!r} is missing slots: {', '.join(sorted(missing))}")
        unknown = set(_SLOT.findall(prompt.content)) - declared
        if unknown:
            raise SlotMismatchError(
                f"prompt {key!r}@{prompt.version} uses undeclared slots: "
                f"{', '.join(sorted(unknown))}"
            )
        return _SLOT.sub(lambda match: str(values[match.group(1)]), prompt.content)

    # -- writes -------------------------------------------------------- #

    def register(self, spec: PromptSpec) -> PromptSpec:
        """Add a new manifest entry; an existing key must be changed explicitly, not silently."""
        if self._config.spec(spec.key) is not None:
            raise PromptError(f"prompt key {spec.key!r} is already registered")
        self._config.specs.append(spec)
        self._save()
        return spec

    def create_version(self, key: str, content: str, version: str) -> PromptVersion:
        """Write a new version file; never activates it. Existing versions are immutable."""
        self.spec(key)
        path = self.path_of(key, version)
        if path.exists():
            raise PromptError(
                f"prompt {key!r} version {version!r} already exists; versions are immutable"
            )
        _atomic_write(path, content)
        return PromptVersion(key=key, version=version, content=content, path=path)

    def activate(self, key: str, version: str) -> None:
        """Point the key at one existing version."""
        self.spec(key)
        path = self.path_of(key, version)
        if not path.is_file():
            raise PromptVersionNotFoundError(f"prompt {key!r} has no version {version!r}")
        self._config.active[key] = version
        self._save()

    def reset(self, key: str) -> bool:
        """Drop the active override, falling back to the spec's default_version."""
        self.spec(key)
        if self._config.active.pop(key, None) is None:
            return False
        self._save()
        return True

    def reload(self) -> PromptsConfig:
        """Re-read the manifest from disk after an external edit."""
        self._store.invalidate(PromptsConfig)
        self._config = self._store.get_or_create(PromptsConfig())
        return self._config

    def _save(self) -> None:
        self._store.save(self._config)


def default_specs() -> list[PromptSpec]:
    """The stable mechanism prompts. Bodies ship with the first consumer (memento note/chat)."""
    return [
        PromptSpec(
            key=PROMPT_MEMENTO_NOTE,
            description="Write the next memento commit note from the span's context.",
            slots=["coord", "prior"],
        ),
        PromptSpec(
            key=PROMPT_MEMENTO_CHAT,
            description="Talk one round with a commit's context.",
            slots=[],
        ),
    ]


class PromptManagerProvider(Provider[PromptManager]):
    """Provide the prompt registry as an IoC singleton — the ghost does not hold it.

    Lazy: the factory needs the bound ghost-home contract, so the first fetch happens after
    the runtime binds it. Contracts instantiation at boot is what makes a broken manifest
    fail startup rather than the first use.
    """

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        ghost_home = con.force_fetch(GhostHome)
        store = con.force_fetch(DoloresConfigStore)
        return PromptManager(
            store,
            ghost_home.home_meta_dir() / DOLORES_PROMPTS_DIRECTORY,
            defaults=default_specs(),
        )
