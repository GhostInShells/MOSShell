"""DoloresEgoNucleus — the ego self-wake nucleus.

A specialized nucleus: it receives self-wake signals from the Dolores ego's turn/start watcher and
produces a low-key BACKGROUND challenge impulse with an empty body; when attended, the impulse is
upgraded to INFO to wake a silent mindflow into a normal challenge round.
"""

from collections.abc import Iterable
from typing import Callable

from ghoshell_container import IoCContainer
from pydantic import Field
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.mindflow import (
    Impulse,
    Nucleus,
    NucleusMeta,
    Priority,
    Signal,
    SignalMeta,
    SignalName,
)
from ghoshell_moss.message import ContextType, Message

__all__ = [
    "NAME",
    "SIGNAL_NAME",
    "DoloresEgoSignalMeta",
    "DoloresEgoNucleus",
    "DoloresEgoNucleusMeta",
    "new_dolores_ego_signal",
]

NAME = "dolores_ego_nucleus"
"""nucleus name — also used as the impulse source name."""

SIGNAL_NAME = "dolores/ego"
"""self-wake signal name — emitted by the ego's turn/start watcher."""


class DoloresEgoSignalMeta(SignalMeta):
    """Self-wake channel signal meta — one turn/start observation is one self-wake signal.

    Extended with the startup boot protocol: ``kind`` distinguishes the ordinary step wake
    (empty) from the startup boot wake, which may carry a ``command`` (spoken logos) and an
    ``instruction`` (preheat text). All three default empty — an unextended signal keeps the
    original empty-body wake behavior.
    """

    kind: str = Field(
        default='',
        description="wake kind — '' (step activity wake) | 'startup' (boot wake)",
    )
    command: str = Field(
        default='',
        description="CTML logos to execute directly (speak) on startup",
    )
    instruction: str = Field(
        default='',
        description="preheat instruction text, wrapped into a <startup> handshake by the mechanism",
    )

    @classmethod
    def signal_name(cls) -> SignalName:
        return SIGNAL_NAME

    @classmethod
    def priority(cls) -> Priority:
        return Priority.INFO


class DoloresEgoNucleus(Nucleus):
    """Self-wake nucleus — emits a BACKGROUND challenge (fire-and-forget); attended upgrades it to INFO to wake attention."""

    NAME = NAME

    def __init__(
        self,
        *,
        name: str = NAME,
        description: str = "wake the ghost when the dolores ego agent is running",
        logger: LoggerItf | None = None,
    ):
        self._name = name
        self._description = description
        self._target_signal = SIGNAL_NAME
        self._logger = logger or get_moss_logger()
        self._impulse: Impulse | None = None
        self._broadcast_cb: Callable[[Signal], None] | None = None
        self._notify_cb: Callable[[Impulse], None] | None = None
        self._index = 0
        self._running = False

    # -- Nucleus ABC --

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def status(self) -> str:
        return ""

    def signals(self) -> list[SignalName]:
        return [self._target_signal]

    def clear(self) -> None:
        self._impulse = None

    def with_bus(
        self,
        signal_broadcast: Callable[[Signal], None],
        fire_impulse: Callable[[Impulse], None],
    ) -> None:
        self._broadcast_cb = signal_broadcast
        self._notify_cb = fire_impulse

    def add_signal(self, signal: Signal) -> None:
        if not self.is_running():
            return
        if signal.name != self._target_signal:
            return
        meta = DoloresEgoSignalMeta.from_signal(signal)
        # challenge: BACKGROUND — low-key, only wins initial when mindflow is idle; suppressed when there is attention.
        self._index += 1
        messages: list[Message] = []
        logos = ""
        if meta is not None and meta.kind == "startup":
            # boot wake: preheat via instruction (handshake), speak via command (reflex-arc logos).
            if meta.instruction:
                messages.append(Message.new(tag="startup").with_content(
                    f"<startup>\n本 instruction 来自文档:\n---\n{meta.instruction}\n</startup>"
                ))
            if meta.command:
                logos = meta.command
        self._impulse = Impulse(
            source=self._name,
            source_idx=self._index,
            id=signal.id,
            priority=Priority.BACKGROUND,
            messages=messages,  # empty body for step wake — handling is defined at the thinking/enter layer.
            logos=logos,
            description="",
            complete=True,  # default (empty) mode = normal arbitration (not silent)
        )
        if self._notify_cb is not None:
            self._notify_cb(self._impulse)

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        # fire-and-forget: dropped on preemption failure — no reraise, no cooldown.
        self._impulse = None
        return

    def attended(self, impulse: Impulse) -> Impulse | None:
        # running package: upgraded to INFO (normal run strength), distinct from the BACKGROUND challenge. Empty body kept.
        if not self.is_running():
            return None
        self._impulse = None
        return impulse.model_copy(update={"priority": Priority.INFO})

    def peek(self, no_stale: bool = True) -> Impulse | None:
        if self._impulse is None:
            return None
        impulse = self._impulse
        self._impulse = None  # fire-and-forget: peek clears; single-consumption, no cache.
        if no_stale and impulse.is_stale():
            return None
        return impulse

    def is_running(self) -> bool:
        return self._running

    async def __aenter__(self) -> Self:
        self._running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._running = False


class DoloresEgoNucleusMeta(NucleusMeta):
    """Factory meta — makes the ego self-wake nucleus discoverable via ``moss manifests nuclei``."""

    def __init__(
        self,
        *,
        name: str = NAME,
        description: str = "wake the ghost when the dolores ego agent is running",
    ):
        self._name = name
        self._description = description

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield DoloresEgoSignalMeta

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return DoloresEgoNucleus(name=self._name, description=self._description, logger=logger)


def new_dolores_ego_signal(
    *messages: ContextType,
    priority: Priority = Priority.INFO,
    description: str = "",
    hint: str = "",
    kind: str = "",
    command: str = "",
    instruction: str = "",
) -> Signal:
    """Helper — build a self-wake signal for the ego's turn/start watcher to emit.

    ``kind`` / ``command`` / ``instruction`` extend the signal for the startup boot
    protocol; leaving them empty keeps the ordinary step wake.
    """
    meta = DoloresEgoSignalMeta(kind=kind, command=command, instruction=instruction)
    return meta.to_signal(
        *messages,
        description=description,
        priority=priority,
        hint=hint,
    )
