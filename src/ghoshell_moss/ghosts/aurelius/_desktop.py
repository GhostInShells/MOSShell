"""Aurelius-owned lifecycle adapter for the existing Ground implementation."""

from __future__ import annotations

from pathlib import Path

from ghoshell_moss.contracts.desktop import Ground, PathOutsideRootError, Pin, UpdateResult
from ghoshell_moss.core.desktop import DefaultGrounds

__all__ = ["AureliusDesktop"]


class AureliusDesktop:
    """Keep Ground as current-frame working memory, separate from Memento."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        default_root: str | Path | None = None,
        enabled: bool = True,
    ) -> None:
        self._workspace_root = Path(workspace_root).resolve()
        self._default_root = Path(default_root).resolve() if default_root is not None else self._workspace_root
        self._enabled = enabled
        self._grounds = DefaultGrounds(workspace_root=self._workspace_root)
        self._primary_label: str | None = None
        self._entered = False
        self._startup_error = ""

    @property
    def workspace_root(self) -> Path:
        return self._workspace_root

    @property
    def primary(self) -> Ground | None:
        if self._primary_label is None:
            return None
        return self._grounds.get(self._primary_label)

    def active(self) -> dict[str, Ground]:
        return self._grounds.active()

    async def open(self, directory: str | Path = ".", *, label: str | None = None) -> Ground:
        target = self._resolve(directory)
        ground = await self._grounds.open(target, label=label)
        if self._primary_label is None:
            self._primary_label = ground.label
        return ground

    async def close(self, label: str) -> None:
        await self._grounds.close(label)
        if label == self._primary_label:
            self._primary_label = next(iter(self._grounds.active()), None)

    def pin(self, label: str, addr: str, note: str = "") -> Pin:
        return self._grounds.pin(label, addr, note)

    def unpin(self, label: str, addr: str) -> None:
        self._grounds.unpin(label, addr)

    async def update(self, label: str, addr: str) -> UpdateResult:
        return await self._grounds.update(label, addr)

    async def frame(self, label: str) -> str:
        return await self._grounds.frame(label)

    def instruction(self) -> str:
        blocks = [ground.instruction().strip() for ground in self._grounds.active().values()]
        return "\n\n".join(block for block in blocks if block)

    async def context(self) -> str:
        blocks = [await ground.context() for ground in self._grounds.active().values()]
        return "\n\n".join(block for block in blocks if block)

    def inspect(self) -> dict[str, object]:
        return {
            "enabled": self._enabled,
            "workspace_root": str(self._workspace_root),
            "primary_label": self._primary_label,
            "startup_error": self._startup_error,
            "active": [
                {
                    "label": ground.label,
                    "root": str(ground.root),
                    "pins": len(ground.pins()),
                }
                for ground in self._grounds.active().values()
            ],
        }

    async def __aenter__(self) -> AureliusDesktop:
        await self._grounds.__aenter__()
        self._entered = True
        if self._enabled:
            if self._default_root.is_dir():
                ground = await self.open(self._default_root, label="workspace")
                self._primary_label = ground.label
            else:
                self._startup_error = f"default Ground root is not a directory: {self._default_root}"
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._entered:
            await self._grounds.__aexit__(exc_type, exc_val, exc_tb)
            self._entered = False

    def _resolve(self, directory: str | Path) -> Path:
        target = Path(directory)
        if not target.is_absolute():
            target = self._workspace_root / target
        target = target.resolve()
        try:
            target.relative_to(self._workspace_root)
        except ValueError as error:
            raise PathOutsideRootError(
                f"ground {target} escapes Aurelius workspace {self._workspace_root}"
            ) from error
        if not target.is_dir():
            raise NotADirectoryError(target)
        return target
