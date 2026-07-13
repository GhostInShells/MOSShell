"""Default FileEditor implementation — thin adapter over the vendored OHEditor.

Layout discipline (K18 three-layer): contract (ABC) → this concrete → future
channel. This module *only* wraps the vendor engine and translates its
exceptions/results into the MOSS-facing types from
``ghoshell_moss.contracts.file_editor``.

**No vendor types leak out.** Callers see ``FileEditorResult`` and
``FileEditorError`` subclasses; ``ToolError`` and ``CLIResult`` stay inside
this module.
"""

from __future__ import annotations

from pathlib import Path

from ghoshell_moss.contracts.file_editor import (
    FileEditor,
    FileEditorError,
    FileEditorResult,
    FileValidationError,
    NoEditHistoryError,
    ParameterInvalidError,
    ParameterMissingError,
)

from ._openhands.editor import Command, OHEditor
from ._openhands.exceptions import (
    EditorToolParameterInvalidError,
    EditorToolParameterMissingError,
)
from ._openhands.exceptions import (
    FileValidationError as _VendorFileValidationError,
)
from ._openhands.exceptions import (
    ToolError as _VendorToolError,
)
from ._openhands.results import CLIResult


class DefaultFileEditor(FileEditor):
    """OHEditor-backed FileEditor.

    Construction params mirror OHEditor's for simplicity but are exposed as
    MOSS-native names:

    - ``max_file_size_mb`` — hard cap on read/edit (default 10 MB, vendor
      value). Exceeding raises ``FileValidationError``.
    - ``workspace_root`` — informational only. Vendor uses it to add
      "did you mean <abs path>?" hints when the caller passes a relative
      path. We intentionally do **not** enforce a spatial boundary here
      (that's the job of Grounds in the desktop contract, if the caller
      wires FileEditor under one). Passing None disables the hint and
      still accepts absolute paths.
    """

    def __init__(
        self,
        *,
        max_file_size_mb: int | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self._engine = OHEditor(
            max_file_size_mb=max_file_size_mb,
            workspace_root=str(workspace_root) if workspace_root else None,
        )

    def view(
        self,
        path: str | Path,
        view_range: list[int] | None = None,
    ) -> FileEditorResult:
        return self._invoke("view", path, view_range=view_range)

    def create(
        self,
        path: str | Path,
        file_text: str,
    ) -> FileEditorResult:
        return self._invoke("create", path, file_text=file_text)

    def str_replace(
        self,
        path: str | Path,
        old_str: str,
        new_str: str,
    ) -> FileEditorResult:
        return self._invoke(
            "str_replace", path, old_str=old_str, new_str=new_str
        )

    def insert(
        self,
        path: str | Path,
        insert_line: int,
        new_str: str,
    ) -> FileEditorResult:
        return self._invoke(
            "insert", path, insert_line=insert_line, new_str=new_str
        )

    def undo_edit(self, path: str | Path) -> FileEditorResult:
        return self._invoke("undo_edit", path)

    def _invoke(
        self, command: Command, path: str | Path, **kwargs
    ) -> FileEditorResult:
        """Call the engine + translate exceptions/results.

        Any vendor exception becomes a matching FileEditorError subclass;
        any success becomes a FileEditorResult. This is the only place where
        vendor types are observed.
        """
        try:
            result = self._engine(command=command, path=str(path), **kwargs)
        except EditorToolParameterMissingError as e:
            raise ParameterMissingError(e.command, e.parameter) from None
        except EditorToolParameterInvalidError as e:
            # Vendor keeps Path objects in `value`; stringify for cleaner messages.
            value = str(e.value) if isinstance(e.value, Path) else e.value
            raise ParameterInvalidError(e.parameter, value) from None
        except _VendorFileValidationError as e:
            raise FileValidationError(e.path, e.reason) from None
        except _VendorToolError as e:
            # NoEditHistoryError disguises as ToolError in vendor —
            # detect by message shape (vendor doesn't type this specially).
            msg = str(e)
            if "No edit history found" in msg:
                raise NoEditHistoryError(str(path)) from None
            # Everything else is an IO / parse / write error — surface as
            # generic FileValidationError with the vendor message as reason.
            # We deliberately do not invent a new exception class; the
            # contract is "read/edit failed" and that is what this signals.
            raise FileValidationError(str(path), msg) from None

        return _to_result(result)


def _to_result(cli: CLIResult) -> FileEditorResult:
    """Vendor CLIResult → MOSS FileEditorResult.

    Vendor's ``error`` field is unused by the current code paths (used to
    be populated by the removed directory-view shell branch). We ignore it
    intentionally — any real error already surfaced as an exception.
    """
    return FileEditorResult(
        output=cli.output or "",
        path=cli.path or "",
        prev_exist=cli.prev_exist,
        old_content=cli.old_content,
        new_content=cli.new_content,
    )
