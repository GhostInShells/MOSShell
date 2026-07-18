"""结构化文件编辑 — Anthropic text_editor tool 血统 | 集成 | alpha

Example:
    # workspace-integrated: registered via manifests providers
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.file_editor_channel import build_file_editor_channel
    main = new_shell_main_channel()
    main.import_channels(build_file_editor_channel)

    # direct composition (tests / hand-wired scripts)
    from ghoshell_moss.channels.file_editor_channel import new_file_editor_channel
    from ghoshell_moss.core.file_editor import DefaultFileEditor
    main.import_channels(new_file_editor_channel(DefaultFileEditor()))
"""

from __future__ import annotations

import json

from ghoshell_container import IoCContainer

from ghoshell_moss.contracts.file_editor import (
    FileEditor,
    FileEditorError,
    ParameterInvalidError,
    ParameterMissingError,
)
from ghoshell_moss.core.blueprint.channel_builder import (
    CommandUtil,
    MutableChannel,
    new_channel,
)
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.file_editor import DefaultFileEditor

__all__ = ["new_file_editor_channel", "build_file_editor_channel"]


# -- factory (IoC integration, core API) -------------------------------------


def build_file_editor_channel(
    container: IoCContainer,
    *,
    channel_name: str = "file_editor",
) -> Channel:
    """IoC-integrated factory. Registered in workspace manifests providers.

    Resolves FileEditor from the container; falls back to DefaultFileEditor()
    if no provider is registered. The fallback is intentional — file editor
    is designed as "zero-config default; special cases override via provider".
    Undo scope, workspace boundary, size caps are all decided by the
    provider (whoever registers a FileEditor into which scope of container).

    :param container: IoC container (session / mode / process — determined
        by whoever imported this factory).
    :param channel_name: CTML tag name (default ``file_editor``). Change
        if renaming under a parent channel via ``import_channels``.
    """
    editor = container.get(FileEditor) or DefaultFileEditor()
    return new_file_editor_channel(editor, channel_name=channel_name)


# -- composition primitive (contract consumer, no IoC knowledge) ------------


def new_file_editor_channel(
    editor: FileEditor,
    *,
    channel_name: str = "file_editor",
) -> MutableChannel:
    """Compose a file editor channel over a FileEditor contract.

    Pure composition primitive — knows nothing about IoC. Tests wire
    ``DefaultFileEditor()`` or a mock directly. Production wiring goes
    through :func:`build_file_editor_channel`.

    Editor identity determines undo history scope: one editor instance =
    one undo history dict. Sharing an editor across multiple channels
    shares the undo stack; separate editors don't.

    Contract errors (``FileEditorError``) raise ``ObserveError`` and abort
    remaining commands in the same CTML channel scope — str_replace_editor
    semantics are "edit-then-observe", a failed edit should stop dependent
    commands from running blind.

    :param editor: FileEditor contract implementation.
    :param channel_name: CTML tag name.
    """
    chan = new_channel(
        name=channel_name,
        description=(
            "Structured file editor — view / create / str_replace / insert / "
            "undo_edit. Absolute paths only. For directory listing use bash/glob."
        ),
    )

    # -- view (nonblocking, always_observe) --------------------------------

    @chan.build.command(name="view", blocking=False, always_observe=True)
    async def view(path: str, view_range: str = "") -> str:
        """Read a file, optionally a line range.

        :param path: absolute file path
        :param view_range: ``"start,end"`` 1-based inclusive (e.g. ``"1,50"``).
            Empty = whole file. CTML parses attribute values via
            ``ast.literal_eval``; ``"1,50"`` arrives as tuple ``(1, 50)``,
            ``"[1, 50]"`` as list, ``view_range:str="1,50"`` as str.
            All three forms accepted.

        Returns ``cat -n`` style snippet. Directory paths error (use
        bash/glob). Binary / oversized files error.
        """
        try:
            parsed = _parse_view_range(view_range)
            result = editor.view(path, view_range=parsed)
            return result.output
        except FileEditorError as e:
            _raise_observe(e)

    # -- create (blocking, always_observe) ---------------------------------

    @chan.build.command(name="create", blocking=True, always_observe=True)
    async def create(path: str, text__: str = "") -> str:
        """Create a new file with the given content.

        :param path: absolute file path (MUST NOT already exist)
        :param text__: full file content, passed via CTML body with CDATA.

        Example::

            <file_editor:create path="/tmp/hello.py"><![CDATA[
            def hello():
                print("hi")
            ]]></file_editor:create>

        Parent directory must exist — no auto-mkdir. Errors if path exists;
        overwrite via ``str_replace``.
        """
        try:
            result = editor.create(path, text__)
            return result.output
        except FileEditorError as e:
            _raise_observe(e)

    # -- str_replace (blocking, always_observe) ----------------------------

    @chan.build.command(name="str_replace", blocking=True, always_observe=True)
    async def str_replace(path: str, text__: str = "") -> str:
        """Replace an exact unique substring.

        :param path: absolute file path
        :param text__: JSON body ``{"old_str": ..., "new_str": ...}`` wrapped
            in CDATA. ``new_str`` optional (empty = delete). Multi-line
            strings use ``\\n`` inside JSON (native escaping, no XML
            attribute pain).

        Example::

            <file_editor:str_replace path="/tmp/hello.py"><![CDATA[
            {"old_str": "def hello():\\n    print(\\"hi\\")",
             "new_str": "def hello():\\n    print(\\"hello world\\")"}
            ]]></file_editor:str_replace>

        ``old_str`` MUST match exactly once. Multiple matches: error
        (add surrounding context to disambiguate). Zero matches:
        whitespace-stripped retry, then error. ``new_str == old_str``:
        error. Snippet with 4 lines of context returned. Enters undo stack.
        """
        try:
            old_str, new_str = _parse_str_replace_args(text__)
            result = editor.str_replace(path, old_str, new_str)
            return result.output
        except FileEditorError as e:
            _raise_observe(e)

    # -- insert (blocking, always_observe) ---------------------------------

    @chan.build.command(name="insert", blocking=True, always_observe=True)
    async def insert(path: str, insert_line: int, text__: str = "") -> str:
        """Insert text AFTER the given line.

        :param path: absolute file path
        :param insert_line: anchor. ``0`` = before file start. ``N`` (file
            line count) = end of file. Out-of-range errors.
        :param text__: text to insert, via CDATA body. Multi-line split
            by ``\\n``.

        Example::

            <file_editor:insert path="/tmp/hello.py" insert_line="0"><![CDATA[
            # -- header comment --
            ]]></file_editor:insert>

        Enters the undo stack.
        """
        try:
            result = editor.insert(path, insert_line, text__)
            return result.output
        except FileEditorError as e:
            _raise_observe(e)

    # -- undo_edit (blocking, always_observe) ------------------------------

    @chan.build.command(name="undo_edit", blocking=True, always_observe=True)
    async def undo_edit(path: str) -> str:
        """Undo the most recent edit to this file.

        :param path: absolute file path

        Errors ``NoEditHistoryError`` when no edits recorded for this file
        in the current session (history is in-memory, per-editor instance,
        cleared on process restart). Undo does not itself enter the stack.
        """
        try:
            result = editor.undo_edit(path)
            return result.output
        except FileEditorError as e:
            _raise_observe(e)

    return chan


# -- helpers -----------------------------------------------------------------


def _parse_view_range(spec) -> list[int] | None:
    """Accept ``"1,50"`` / ``(1, 50)`` / ``[1, 50]`` / ``""`` → normalized.

    CTML runs ``ast.literal_eval`` on attribute values, so ``view_range="1,50"``
    lands here as tuple ``(1, 50)``, not str. We accept all three shapes so
    the model doesn't have to remember the ``:str`` suffix.
    """
    if spec is None or spec == "":
        return None
    if isinstance(spec, (list, tuple)):
        if len(spec) != 2:
            raise ParameterInvalidError(
                "view_range", spec, "expected [start, end] (2 ints)"
            )
        try:
            return [int(spec[0]), int(spec[1])]
        except (TypeError, ValueError):
            raise ParameterInvalidError(
                "view_range", spec, "start,end must be integers"
            ) from None
    if isinstance(spec, str):
        parts = spec.split(",")
        if len(parts) != 2:
            raise ParameterInvalidError(
                "view_range",
                spec,
                'expected "start,end" (1-based inclusive), e.g. "1,50"',
            )
        try:
            return [int(parts[0].strip()), int(parts[1].strip())]
        except ValueError:
            raise ParameterInvalidError(
                "view_range", spec, "start,end must be integers"
            ) from None
    raise ParameterInvalidError(
        "view_range", spec, "expected string, list, or tuple"
    )


def _parse_str_replace_args(text: str) -> tuple[str, str]:
    """Parse the CDATA-wrapped JSON body of str_replace.

    Returns ``(old_str, new_str)``. Missing ``old_str`` → ParameterMissingError,
    bad JSON / wrong shape → ParameterInvalidError. Two fields only — Pydantic
    is overkill; upgrade if the schema grows (e.g. regex, count).
    """
    if not text:
        raise ParameterMissingError("str_replace", "text__")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParameterInvalidError(
            "text__", text[:80], f"expected JSON body: {e.msg}"
        ) from None
    if not isinstance(payload, dict):
        raise ParameterInvalidError(
            "text__", type(payload).__name__,
            'expected JSON object {"old_str": ..., "new_str": ...}',
        )
    if "old_str" not in payload:
        raise ParameterMissingError("str_replace", "old_str")
    old_str = payload["old_str"]
    new_str = payload.get("new_str", "")
    if not isinstance(old_str, str):
        raise ParameterInvalidError(
            "old_str", type(old_str).__name__, "must be a string"
        )
    if not isinstance(new_str, str):
        raise ParameterInvalidError(
            "new_str", type(new_str).__name__, "must be a string"
        )
    return old_str, new_str


def _raise_observe(exc: FileEditorError) -> None:
    """Contract errors surface via ``ObserveError`` — aborts remaining
    commands in the same CTML channel scope. str_replace_editor semantics
    require the model to observe every edit's outcome before proceeding.
    """
    CommandUtil.raise_observe(f"[{type(exc).__name__}] {exc}")
