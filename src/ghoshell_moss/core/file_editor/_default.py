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

import re
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
        # v2 read-side verbs (file_list / glob / grep) don't go through the
        # vendor engine — they need the same construction params directly.
        self._workspace_root = Path(workspace_root) if workspace_root else None
        self._max_file_size_mb = max_file_size_mb

    def view(
        self,
        path: str | Path,
        view_range: list[int] | None = None,
    ) -> FileEditorResult:
        return self._invoke("view", path, view_range=view_range)

    def file_list(
        self,
        path: str | Path = ".",
    ) -> FileEditorResult:
        target = self._resolve(path)
        if not target.exists():
            raise FileValidationError(str(target), "no such file or directory")
        if not target.is_dir():
            raise ParameterInvalidError(
                "path", str(target), "is not a directory — use view for files"
            )
        entries = sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        lines = [f"Directory: {target}"]
        if not entries:
            lines.append("(empty)")
        for e in entries:
            kind = "dir" if e.is_dir() else "symlink" if e.is_symlink() else "file"
            try:
                size = _human_size(e.stat().st_size)
            except OSError:
                size = "?"
            lines.append(f"  {e.name:<40} {size:>8}  {kind}")
        return FileEditorResult(output="\n".join(lines), path=str(target), prev_exist=True)

    def glob(self, pattern: str) -> FileEditorResult:
        if not pattern or not pattern.strip():
            raise ParameterMissingError("glob", "pattern")
        p = Path(pattern)
        if p.is_absolute():
            base, rel = Path("/"), str(p).lstrip("/")
        else:
            base = self._workspace_root or Path.cwd()
            rel = pattern
        matches = sorted(base.glob(rel))
        lines = [f"Glob {pattern!r} ({len(matches)} match{'es' if len(matches) != 1 else ''}):"]
        if not matches:
            lines.append("  (none)")
        else:
            for m in matches[: _GLOB_CAP]:
                lines.append(f"  {m}")
            if len(matches) > _GLOB_CAP:
                lines.append(f"  ... ({len(matches) - _GLOB_CAP} more omitted)")
        return FileEditorResult(output="\n".join(lines), path=str(base), prev_exist=True)

    def grep(
        self,
        pattern: str,
        path: str | Path,
        **options: object,
    ) -> FileEditorResult:
        if not pattern or not pattern.strip():
            raise ParameterMissingError("grep", "pattern")
        unknown = set(options) - _GREP_OPTIONS
        if unknown:
            raise ParameterInvalidError(
                "options",
                sorted(unknown),
                f"unknown grep option(s); recognized: {sorted(_GREP_OPTIONS)}",
            )
        case_sensitive = bool(options.get("case_sensitive", True))
        max_results = options.get("max_results", 100)
        if not isinstance(max_results, int) or max_results <= 0:
            raise ParameterInvalidError(
                "max_results", max_results, "must be a positive int"
            )

        target = self._resolve(path)
        if not target.exists():
            raise FileValidationError(str(target), "no such file or directory")
        if not target.is_file():
            raise ParameterInvalidError(
                "path", str(target), "is not a file — grep is single-file"
            )
        if self._max_file_size_mb and target.stat().st_size > self._max_file_size_mb * 1024 * 1024:
            raise FileValidationError(
                str(target),
                f"file exceeds max_file_size_mb={self._max_file_size_mb}",
            )
        if _is_binary(target):
            raise FileValidationError(str(target), "binary file — grep skipped")

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            rx = re.compile(pattern, flags)
        except re.error as e:
            raise ParameterInvalidError("pattern", pattern, f"invalid regex: {e}") from None

        matches: list[str] = []
        total = 0
        try:
            with target.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    if rx.search(line):
                        total += 1
                        if len(matches) < max_results:
                            matches.append(f"{lineno}: {line.rstrip()}")
        except UnicodeDecodeError:
            raise FileValidationError(str(target), "not valid utf-8 text") from None

        lines = [f"{target} ({total} matching line{'s' if total != 1 else ''}):"]
        if not matches:
            lines.append("  (no matches)")
        else:
            lines.extend(f"  {m}" for m in matches)
            if total > len(matches):
                lines.append(f"  ... ({total - len(matches)} more omitted)")
        return FileEditorResult(output="\n".join(lines), path=str(target), prev_exist=True)

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

    def _resolve(self, path: str | Path) -> Path:
        """Absolute paths as-is; relative resolved against workspace_root
        (falling back to process cwd). K4 stance: a hint, not an enforced
        boundary — spatial confinement is the caller's job."""
        p = Path(path)
        if not p.is_absolute():
            p = (self._workspace_root or Path.cwd()) / p
        return p.resolve()


def _is_binary(path: Path) -> bool:
    """K9 stance: read first 8 KB, reject on a NUL byte."""
    with path.open("rb") as fh:
        return b"\x00" in fh.read(8192)


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}K"
    return f"{size / (1024 * 1024):.1f}M"


_GLOB_CAP = 100
_GREP_OPTIONS = frozenset({"case_sensitive", "max_results"})


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
