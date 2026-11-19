"""EgoLifecycleHook — modules that join the ego's lifecycle.

The registry is a catalog of hooks that have been *registered*, usually by providers. It
does not run them at registration: ``start(lifecycle)`` is the one activation point, called
by the runtime at the phase it chooses. A hook is constructed cheaply; ``__aenter__`` is
where it resolves late dependencies and does its boot work, ``listen`` registers its
listeners, and ``__aexit__`` takes both back.

A hook is not an event listener. The listener surface is :class:`EgoLifecycle`; a hook is a
module that owns a lifetime and may register several listeners. Blocking is still a
property of the phase, not of the hook.
"""

from __future__ import annotations

from typing import Protocol

from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger

from .ego_lifecycle import EgoLifecycle

__all__ = ["EgoLifecycleHook", "EgoLifecycleRegistry"]


class EgoLifecycleHook(Protocol):
    """A module with a lifetime, wired into the ego's lifecycle at start time."""

    async def listen(self, lifecycle: EgoLifecycle) -> None:
        """Register this module's listeners. Called once, after ``__aenter__``."""
        ...

    async def __aenter__(self) -> Self:
        """Resolve late dependencies and do boot work. Cheap registries stay cheap."""
        ...

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Release what ``__aenter__`` took. Listener discards belong to the hook."""
        ...


class EgoLifecycleRegistry:
    """Register hooks first; start them once, at the runtime's chosen phase.

    Registration is intentionally separate from activation: a provider can populate the
    registry during container bootstrap, while the runtime decides when the ghost is ready
    for the hooks to run. ``start`` enters every hook in registration order, then calls
    ``listen``; failures are logged per hook. A non-critical hook that fails is skipped; a
    critical one stops the ones already started and propagates.
    """

    def __init__(self, *, logger: LoggerItf | None = None) -> None:
        self._logger = logger or get_moss_logger()
        self._hooks: dict[str, tuple[EgoLifecycleHook, bool]] = {}
        self._started: list[tuple[str, EgoLifecycleHook]] = []
        self._state = "registered"  # registered -> started -> stopped

    def register(self, name: str, hook: EgoLifecycleHook, *, critical: bool = False) -> None:
        """Add a hook to the catalog. No constructor side effect, no activation."""
        if self._state != "registered":
            raise RuntimeError("ego lifecycle hooks: register() is only valid before start()")
        name = name.strip()
        if not name:
            raise ValueError("ego lifecycle hooks: a hook name cannot be empty")
        if name in self._hooks:
            raise ValueError(f"ego lifecycle hooks: duplicate hook name {name!r}")
        self._hooks[name] = (hook, critical)

    async def start(self, lifecycle: EgoLifecycle) -> None:
        """Activate every registered hook; idempotent while already started."""
        if self._state == "started":
            return
        if self._state == "stopped":
            raise RuntimeError("ego lifecycle hooks: a stopped registry cannot start again")
        self._state = "started"
        for name, (hook, critical) in self._hooks.items():
            entered = False
            try:
                await hook.__aenter__()
                entered = True
                await hook.listen(lifecycle)
            except Exception:
                self._logger.exception("ego lifecycle hook failed: %s", name)
                if entered:
                    await self._exit(name, hook)
                if critical:
                    await self.stop()
                    raise
                continue
            self._started.append((name, hook))

    async def stop(self) -> None:
        """Deactivate started hooks in reverse order; idempotent before/after start."""
        if self._state != "started":
            return
        self._state = "stopped"
        for name, hook in reversed(self._started):
            await self._exit(name, hook)
        self._started.clear()

    async def _exit(self, name: str, hook: EgoLifecycleHook) -> None:
        try:
            await hook.__aexit__(None, None, None)
        except Exception:
            self._logger.exception("ego lifecycle hook failed to stop: %s", name)
