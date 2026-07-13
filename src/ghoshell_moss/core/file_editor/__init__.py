"""File editor — MOSS 结构化文件编辑动作层 (concrete).

公开契约: ``ghoshell_moss.contracts.file_editor``.
Vendor: openhands-aci editor 子集, 见 ``_openhands/UPSTREAM.md``.

进程内默认实现: ``DefaultFileEditor``.
"""

from ghoshell_moss.contracts.file_editor import (
    FileEditor,
    FileEditorError,
    FileEditorResult,
    FileValidationError,
    NoEditHistoryError,
    ParameterInvalidError,
    ParameterMissingError,
)

from ghoshell_moss.core.file_editor._default import DefaultFileEditor

__all__ = [
    # 契约 re-export (便利导入)
    "FileEditor",
    "FileEditorResult",
    "FileEditorError",
    "ParameterMissingError",
    "ParameterInvalidError",
    "FileValidationError",
    "NoEditHistoryError",
    # 实现
    "DefaultFileEditor",
]
