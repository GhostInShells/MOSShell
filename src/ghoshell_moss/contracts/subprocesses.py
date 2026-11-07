"""Subprocesses contract — re-exported from :mod:`ghoshell_processes`.

机制层契约已迁到独立包 ``ghoshell_processes``（不和 moss 同生的消费者也能复用）。这里保留
原路径与同名符号，既有 import、类型注解与 ``isinstance`` 检查不需要改动。
"""

from ghoshell_processes import (
    CaptureSpec,
    ErrorInfo,
    ManagedProcess,
    ProcessMeta,
    ProcessOutput,
    SubprocessFacade,
    Subprocesses,
)

__all__ = [
    "CaptureSpec",
    "ErrorInfo",
    "ManagedProcess",
    "ProcessMeta",
    "ProcessOutput",
    "SubprocessFacade",
    "Subprocesses",
]
