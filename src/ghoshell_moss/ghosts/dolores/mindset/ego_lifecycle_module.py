"""EgoLifecycleModule — modules that join the ego's lifecycle.

The registry is a catalog of modules that have been *registered*, usually by providers. It
does not run them at registration: ``start(lifecycle)`` is the one activation point, called
by the runtime at the phase it chooses. A module is constructed cheaply; ``__aenter__`` is
where it resolves late dependencies and does its boot work, ``listen`` registers its
listeners, and ``__aexit__`` takes both back.

A module is not an event listener. The listener surface is :class:`EgoLifecycle`; a module
owns a lifetime and may register several listeners. Blocking is still a property of the
phase, not of the module.
"""

from __future__ import annotations

from typing import Protocol

from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger

from .ego_lifecycle import EgoLifecycle

__all__ = ["EgoLifecycleModule", "EgoLifecycleRegistry"]


class EgoLifecycleModule(Protocol):
    """A module with a lifetime, wired into the ego's lifecycle at start time."""

    async def listen(self, lifecycle: EgoLifecycle) -> None:
        """Register this module's listeners. Called once, after ``__aenter__``."""
        ...

    async def __aenter__(self) -> Self:
        """Resolve late dependencies and do boot work. Cheap registries stay cheap."""
        ...

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Release what ``__aenter__`` took. Listener discards belong to the module."""
        ...


class EgoLifecycleRegistry:
    """Register modules first; start them once, at the runtime's chosen phase.

    Registration is intentionally separate from activation: a provider can populate the
    registry during container bootstrap, while the runtime decides when the ghost is ready
    for the modules to run. ``start`` enters every module in registration order, then calls
    ``listen``; failures are logged per module. A non-critical module that fails is skipped;
    a critical one stops the ones already started and propagates.
    """

    def __init__(self, *, logger: LoggerItf | None = None) -> None:
        self._logger = logger or get_moss_logger()
        self._modules: dict[str, tuple[EgoLifecycleModule, bool]] = {}
        self._started: list[tuple[str, EgoLifecycleModule]] = []
        self._state = "registered"  # registered -> started -> stopped

    def register(self, name: str, module: EgoLifecycleModule, *, critical: bool = False) -> None:
        """Add a module to the catalog. No constructor side effect, no activation."""
        if self._state != "registered":
            raise RuntimeError("ego lifecycle modules: register() is only valid before start()")
        name = name.strip()
        if not name:
            raise ValueError("ego lifecycle modules: a module name cannot be empty")
        if name in self._modules:
            raise ValueError(f"ego lifecycle modules: duplicate module name {name!r}")
        self._modules[name] = (module, critical)

    async def start(self, lifecycle: EgoLifecycle) -> None:
        """Activate every registered module; idempotent while already started."""
        if self._state == "started":
            return
        if self._state == "stopped":
            raise RuntimeError("ego lifecycle modules: a stopped registry cannot start again")
        self._state = "started"
        for name, (module, critical) in self._modules.items():
            entered = False
            try:
                await module.__aenter__()
                entered = True
                await module.listen(lifecycle)
            except Exception:
                self._logger.exception("ego lifecycle module failed: %s", name)
                if entered:
                    await self._exit(name, module)
                if critical:
                    await self.stop()
                    raise
                continue
            self._started.append((name, module))

    async def stop(self) -> None:
        """Deactivate started modules in reverse order; idempotent before/after start."""
        if self._state != "started":
            return
        self._state = "stopped"
        for name, module in reversed(self._started):
            await self._exit(name, module)
        self._started.clear()

    async def _exit(self, name: str, module: EgoLifecycleModule) -> None:
        try:
            await module.__aexit__(None, None, None)
        except Exception:
            self._logger.exception("ego lifecycle module failed to stop: %s", name)
