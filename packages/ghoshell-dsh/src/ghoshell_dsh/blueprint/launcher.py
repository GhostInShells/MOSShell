"""The launcher contract — what "dsh is open" means, said without saying how it was opened.

A launcher owns a **lifecycle**, not a transport. dsh can be opened in more than one way — a
web server on a port, a headless process on a pipe, whatever comes next — and none of those
ways belongs in this file. What a consumer asks for is: start it, hand me something usable,
stop it when I am done. Which dsh, which port, which profile, which probe path is the concrete
launcher's business, decided by its own config or by whoever wired it in an IoC container.

That is the point of blueprint programming here: a consumer holding only these types can do
the basic thing — open a dsh, drive it, close it — without knowing which generation or which
transport answered.

**Not everything is promised.** `start()` may hand back a full-duplex `Connection`, or only
something that can create sessions. Demanding every ability from every launcher is how a
contract stops fitting the next transport.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # what `start()` hands back; `Connection` is a `SessionSource` with more on it
    from .agent import SessionSource

__all__ = ["Exit", "Launcher"]


@dataclass(frozen=True, slots=True)
class Exit:
    """How the dsh process ended.

    Kept deliberately small: a caller that wants to report or retry needs the code and what
    dsh last said, and nothing else. `self_shutdown` distinguishes "dsh decided to stop" from
    "something killed it", which is the difference between a restart and a hunt.
    """

    exit_code: int | None
    stderr: str
    self_shutdown: bool = False


class Launcher(ABC):
    """The lifetime of one dsh.

    Three operations and one attribute of state. Everything else a launcher knows — argv,
    environment, port, readiness probe — is inside the implementation, because a consumer
    that had to supply it would no longer be transport-agnostic.
    """

    @abstractmethod
    async def start(self) -> SessionSource:
        """Open dsh and return something usable.

        **Returning means usable.** A launcher that cannot become usable raises — `TimeoutError`
        when its deadline passes — instead of returning a half-open thing that fails later at a
        random call site. Calling `start()` while already running returns the same object
        rather than opening a second one.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Close whatever `start()` opened.

        Idempotent: stopping twice is not an error, and stopping a launcher that never started
        is a no-op. This is the method that must not leak a process — an orphan outliving its
        owner is the failure this contract exists to prevent.
        """

    @abstractmethod
    def is_running(self) -> bool:
        """Whether the thing opened by `start()` is still alive right now."""

    def exit_info(self) -> Optional[Exit]:
        """How it ended, once it has; `None` while running, before starting, or when the
        launcher does not track exit at all."""
        return None

    def on_exit(self, callback: Callable[[Exit], None]) -> None:
        """Register a callback for the end of the process.

        Optional capability: a launcher that has no process to watch raises
        `NotImplementedError` rather than silently never calling back.
        """
        raise NotImplementedError(f"{type(self).__name__} does not report exit")

    async def __aenter__(self) -> SessionSource:
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
