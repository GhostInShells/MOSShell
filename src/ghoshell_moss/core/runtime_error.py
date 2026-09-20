"""RuntimeErrorLogImpl — concrete runtime error collector (a logging.Handler).

Bounded FIFO (ERROR+) + never-drop CRITICAL buffer. Attached to the moss logger
before provider bootstrap so startup degradations are captured.
"""
import logging
import threading
from collections import deque

from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.contracts.runtime_error import RuntimeErrorLog, RuntimeErrorRecord

__all__ = ["RuntimeErrorLogImpl"]


class RuntimeErrorLogImpl(RuntimeErrorLog, logging.Handler):
    """Concrete collector — a ``logging.Handler`` wired to two bounded stores.

    Handler level is ERROR, so ``emit`` only receives ERROR+ records. ERROR+ go into
    a bounded FIFO (``pull`` latest n); CRITICAL additionally go into a never-drop
    buffer (``pull_critical``). A lock guards the stores — logging is multi-threaded.
    """

    def __init__(self, *, fifo_capacity: int = 64) -> None:
        logging.Handler.__init__(self, level=logging.ERROR)
        self._lock = threading.Lock()
        self._fifo: deque[RuntimeErrorRecord] = deque(maxlen=fifo_capacity)
        self._critical: list[RuntimeErrorRecord] = []
        self._total = 0

    def attach(self, logger: LoggerItf) -> None:
        if self not in logger.handlers:
            logger.addHandler(self)

    def emit(self, record: logging.LogRecord) -> None:
        entry = RuntimeErrorRecord(
            levelname=record.levelname,
            message=record.getMessage(),
            created=record.created,
            logger=record.name,
            location=f"{record.filename}:{record.lineno}",
        )
        with self._lock:
            self._total += 1
            self._fifo.append(entry)
            if record.levelno >= logging.CRITICAL:
                self._critical.append(entry)

    def pull(self, n: int) -> list[RuntimeErrorRecord]:
        with self._lock:
            return list(self._fifo)[-n:]

    def pull_critical(self) -> list[RuntimeErrorRecord]:
        with self._lock:
            return list(self._critical)

    def total(self) -> int:
        with self._lock:
            return self._total
