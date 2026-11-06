"""The launcher contract — what "dsh is open" means, said without saying how it was opened.

A launcher owns a **lifecycle**, not a transport. dsh can be opened in more than one way — a
web server on a port, a headless process on a pipe, whatever comes next — and none of those
ways belongs in this file. What a consumer asks for is: start it, hand me something usable,
stop it when I am done. Which dsh, which port, which profile, which probe path is the concrete
launcher's business, decided by its own config or by whoever wired it in an IoC container.

That is the point of blueprint programming here: a consumer holding only these types can do the
basic thing — open a dsh, drive it, close it — without knowing which generation or which
transport answered.

### A launcher exposes; it does not hand back

`start()` returns nothing. Once the dsh is up, the launcher **exposes** what is running —
`connection()` for dsh's own face, `controller()` for the reinforced one — and a consumer takes
whichever it needs from the runtime.

The reason is independence, and it is not hypothetical: **a `Connection` must be usable without
a launcher at all.** Pointing at a dsh somebody else started is an ordinary case — a web
connection needs a URL, not a process. If `start()` returned the connection, the only way to
obtain one would be to own its process, and that is simply false. A launcher is one way to
arrive at a connection; it is not the way.

**Not every face is promised.** A launcher may expose a connection and no controller, or the
reverse; an accessor returning `None` while `is_running()` is true means this launcher does not
provide that face.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from .connection import Connection
from .controller import DshController

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
    environment, port, readiness probe — is inside the implementation, because a consumer that
    had to supply it would no longer be transport-agnostic.
    """

    @abstractmethod
    async def start(self) -> None:
        """Bring the dsh up.

        **Returning means usable.** A launcher that cannot become usable raises — `TimeoutError`
        when its deadline passes — instead of returning a half-open thing that fails later at a
        random call site. Calling `start()` while already running is a no-op, not a second dsh.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Bring it down.

        Idempotent: stopping twice is not an error, and stopping a launcher that never started is
        a no-op. This is the method that must not leak a process — an orphan outliving its owner
        is the failure this contract exists to prevent.
        """

    @abstractmethod
    def is_running(self) -> bool:
        """Whether the thing `start()` brought up is still alive right now."""

    # ---- what is running, taken from the runtime ---- #

    def connection(self) -> Optional[Connection]:
        """dsh's own face, once it is up — `None` when there is nothing to hand out.

        `None` covers two situations and `is_running()` tells them apart: not started (or already
        stopped), or a launcher that does not expose this face.
        """
        return None

    def controller(self) -> Optional[DshController]:
        """The reinforced face, once it is up — same `None` rule as `connection()`."""
        return None

    def exit_info(self) -> Optional[Exit]:
        """How it ended, once it has; `None` while running, before starting, or when the launcher
        does not track exit at all."""
        return None

    def on_exit(self, callback: Callable[[Exit], None]) -> None:
        """Register a callback for the end of the process.

        Optional capability: a launcher that has no process to watch raises
        `NotImplementedError` rather than silently never calling back.
        """
        raise NotImplementedError(f"{type(self).__name__} does not report exit")

    async def __aenter__(self) -> Launcher:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
