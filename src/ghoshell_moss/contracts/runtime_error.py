"""RuntimeErrorLog — bounded runtime error collector, pullable by the model.

A logging.Handler on the moss logger capturing error-level+ records into a bounded
FIFO (``pull`` returns the latest n, oldest dropped on overflow) plus a never-drop
CRITICAL buffer (``pull_critical``). The channel built on it lets the model
self-diagnose startup/runtime failures — the counterpart to graceful provider
degradation (NullSpeech / NullASR): those degrade silently, this surfaces the
degradations so the model can see *why* a capability is missing.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ghoshell_common.contracts import LoggerItf

__all__ = ["RuntimeErrorLog", "RuntimeErrorRecord"]


@dataclass
class RuntimeErrorRecord:
    """One captured error — the minimal, renderable surface of a log record."""

    levelname: str
    message: str
    created: float
    logger: str = ""
    location: str = ""  # "filename:lineno"

    def render(self) -> str:
        return f"[{self.levelname}] {self.message} ({self.location})"


class RuntimeErrorLog(ABC):
    """Runtime error collector — bounded ERROR+ tail, pullable by the model.

    Attached to the moss logger as a ``logging.Handler``. Captures ERROR+ records
    into a bounded FIFO (``pull`` returns the latest n, newest last) plus a
    never-drop CRITICAL buffer (``pull_critical``). ``total`` counts every error
    seen since attach (including dropped), so the consumer can tell "N errors,
    showing the last m".

    Lifecycle: instantiated at project init *before* providers, attached to the
    logger after logging config but before container bootstrap — so provider
    bootstrap errors (e.g. ASR degraded to NullASR) are captured.
    """

    @abstractmethod
    def attach(self, logger: LoggerItf) -> None:
        """Attach this collector to a logger — starts capturing ERROR+ records."""

    @abstractmethod
    def pull(self, n: int) -> list[RuntimeErrorRecord]:
        """Pull the latest n error records (ERROR+), newest last."""

    @abstractmethod
    def pull_critical(self) -> list[RuntimeErrorRecord]:
        """Pull all CRITICAL records — never dropped, independent of the FIFO."""

    @abstractmethod
    def total(self) -> int:
        """Total ERROR+ records seen since attach (including dropped)."""
