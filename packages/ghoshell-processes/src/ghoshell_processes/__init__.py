"""ghoshell_processes — spawn and govern child processes that never outlive their owner."""

from .contracts import (
    CaptureSpec,
    ErrorInfo,
    ManagedProcess,
    ProcessMeta,
    ProcessOutput,
    Subprocesses,
    SubprocessFacade,
)
from .impl import SubprocessesImpl
from .utils import killpg

__all__ = [
    "CaptureSpec",
    "ErrorInfo",
    "ManagedProcess",
    "ProcessMeta",
    "ProcessOutput",
    "SubprocessFacade",
    "Subprocesses",
    "SubprocessesImpl",
    "killpg",
]
