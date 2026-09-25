"""Question-set as a thinking frame — a session-scoped self-model that re-asserts itself | meta | beta

A **frame** is a question set the model resolves from its own context, to hold a stable
situational understanding across context loss (compaction, a new window, a different model
instance). The questions are the asset; the answers are working state. Every question
must be answerable from context; ``unknown`` is a first-class resolution — the unresolved
set is the frame's most valuable output. A frame is an *extraction* device, not a decision
device.

Why it is more than a document: a message read once sinks into the context and stops
reaching attention even while it stays in the window — subsequent rounds' attention
follows the recent tokens, not the cold prefix. A frame therefore has to re-assert
itself. Each loaded frame is a named notice fragment whose body is its unresolved
questions; the fragment carries a ``refreshed at`` timestamp that jumps once per refresh
threshold, forcing the warm layer to re-emit the whole fragment and pull the frame back
into the working set. The cadence is a reminder, not a sync: the content is already in
context; what is missing is the bit that says *this is still here*.

The channel exposes a small imperative surface (``load``, ``define``, ``resolve``,
``resolved``, ``reset``, ``export``); the disk format is a pydantic-dumped ``.frame.yml``
whose first line points back at :class:`Frame`. ``root`` is a **file-operation boundary**
(read + write must resolve inside it); frames are **not** discovered by scanning it — the
channel loads only what the model asks for, or the ``init_frame`` handed at build time.

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.frame_channel import Frame, new_frame_channel

    orientation = Frame(
        label="orientation",
        description="reconstruct the situation after context loss",
        questions=["What environment am I in?", "Who am I talking to?"],
    )
    main = new_shell_main_channel()
    main.import_channels(new_frame_channel(root=project_root, init_frame=orientation))
"""

from __future__ import annotations

import asyncio
import datetime
import json
import time
from pathlib import Path
from typing import Optional

import dateutil.tz
import yaml
from ghoshell_common.helpers import generate_import_path, yaml_pretty_dump
from pydantic import BaseModel, Field, ValidationError

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

__all__ = [
    "Frame",
    "new_frame_channel",
    "new_frame_channel_from_file",
    "FRAME_SUFFIX",
    "DEFAULT_REFRESH_SECONDS",
]

FRAME_SUFFIX = ".frame.yml"
"""Every frame file on disk ends with this suffix; ``load``/``export`` enforce it."""

DEFAULT_REFRESH_SECONDS = 120.0
"""Default re-assert cadence — how long a frame may hold attention before it fades."""


class Frame(BaseModel):
    """A question set the model uses as a session self-model.

    Serialisable both ways: dumped to ``<label>.frame.yml`` for cross-session use, or
    injected directly at channel construction (e.g. from a ghost's startup config).
    Answers travel with the frame — whether they are loaded back is a caller's choice.
    """

    label: str = Field(
        description="Stable identity of this frame — appears in the notice header and in "
                    "resolve()/resolved() as the target selector. Also the default "
                    "filename stem for export()."
    )
    description: str = Field(
        default="",
        description="One-line self-explanation: what situation this frame orients.",
    )
    questions: list[str] = Field(
        default_factory=list,
        description="Extraction questions, in order. Every question must be answerable "
                    "from the model's own context; do not include decision questions "
                    "(\"what should I do next\") — those belong in reasoning, not here.",
    )
    answers: dict[int, str] = Field(
        default_factory=dict,
        description="Resolved answers keyed by question index. Kept on the frame so the "
                    "disk format is a single artefact; whether a fresh load reads them "
                    "back is controlled by load(with_answers=...).",
    )


class SessionFrame:
    """A loaded frame with re-assert clock state layered on top."""

    def __init__(self, frame: Frame) -> None:
        self.frame = frame
        # Monotonic clock of the last re-assert; -inf so the first meta refresh always
        # signs, making the frame appear as soon as it is loaded.
        self.refreshed_monotonic: float = float("-inf")
        self.refreshed_at: str = _clock_text()

    @property
    def label(self) -> str:
        return self.frame.label

    @property
    def unresolved(self) -> list[int]:
        return [i for i in range(len(self.frame.questions)) if i not in self.frame.answers]

    @property
    def complete(self) -> bool:
        return not self.unresolved

    def render_notice(self) -> str:
        """The warm fragment body — unresolved questions, then the refresh timestamp.

        A resolved question leaves the fragment: the answer is already in the transcript,
        so re-sending it is cost without information. A complete frame yields no body at
        all (the caller turns that into removal).
        """
        lines = [f"[{i}] {self.frame.questions[i]}" for i in self.unresolved]
        if not lines:
            return ""
        lines.append(f"refreshed at: {self.refreshed_at}")
        return "\n".join(lines)


def _clock_text() -> str:
    return datetime.datetime.now(dateutil.tz.gettz()).strftime("%H:%M")


def new_frame_channel(
    root: str | Path,
    init_frame: Frame | None = None,
    *,
    refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
    name: str = "frame",
    description: str = "Question-set as a thinking frame — a session self-model that re-asserts itself",
) -> MutableChannel:
    """Build a frame channel.

    :param root: file-operation boundary — every ``load`` / ``export`` path resolves
        inside it or raises. Not a discovery root; the channel does not scan it.
    :param init_frame: a frame injected at construction — the default first frame for
        this session. Bypasses ``load``; the caller (e.g. a ghost's startup) is the source.
    :param refresh_seconds: how long a frame may hold attention before it re-asserts.
    :param name: channel name.
    :param description: channel description.
    """
    root_path = Path(root).expanduser().resolve()
    frames: dict[str, SessionFrame] = {}

    if init_frame is not None:
        frames[init_frame.label] = SessionFrame(init_frame.model_copy(deep=True))

    chan = new_channel(name=name, description=description)

    # ---- helpers (closure) ------------------------------------------------------------

    def _resolve_inside_root(path_like: str | Path) -> Path:
        """Resolve a user-supplied path and reject anything outside root."""
        candidate = Path(str(path_like)).expanduser()
        candidate = candidate if candidate.is_absolute() else root_path / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(root_path)
        except ValueError as exc:
            raise ValueError(f"path escapes frame root: {candidate}") from exc
        return candidate

    def _find(label: str) -> Optional[SessionFrame]:
        label = (label or "").strip()
        if not label:
            # No explicit label: the most recently loaded frame (dict preserves insertion).
            if not frames:
                return None
            return next(reversed(frames.values()))
        return frames.get(label)

    def _read_frame_file(path: Path) -> Frame:
        text = path.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid yaml at {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"frame file must be a yaml mapping: {path}")
        try:
            return Frame.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"frame file does not match Frame schema ({path}): {exc}") from exc

    def _write_frame_file(path: Path, frame: Frame) -> None:
        import_path = generate_import_path(Frame)
        body = yaml_pretty_dump(frame.model_dump())
        content = f"# dump from `{import_path}`\n{body}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    # ---- static surface (instruction + refresh + notices) -----------------------------

    def _render_instruction() -> str:
        blocks = [
            "## Frame\n"
            "A frame is a question set you resolve from your own context to hold a "
            "stable self-model across context loss. Every question is answerable from "
            "context; `unknown` is a valid resolution. Your unresolved questions are "
            "re-asserted below on a refresh cadence — treat each re-appearance as a "
            "prompt to re-check whether the context has caught up. Use `resolve` to "
            "lift a question out, `resolved` to pull answers, `define` to invent a new "
            "frame in place, and `export` to persist one for future sessions."
        ]
        for sf in frames.values():
            done = len(sf.frame.questions) - len(sf.unresolved)
            head = f"[{sf.label}] {done}/{len(sf.frame.questions)}"
            if sf.frame.description:
                head += f" — {sf.frame.description}"
            blocks.append(head)
        return "\n\n".join(blocks)

    chan.build.instruction(_render_instruction)

    @chan.build.refresh_meta
    async def _refresh_clocks() -> None:
        """Re-sign every frame whose threshold has passed. Bumping ``refreshed_at``
        changes the fragment text, which is what makes the warm layer re-emit it —
        the timestamp is the whole re-assertion mechanism."""
        now = time.monotonic()
        for sf in frames.values():
            if now - sf.refreshed_monotonic >= refresh_seconds:
                sf.refreshed_monotonic = now
                sf.refreshed_at = _clock_text()

    @chan.build.named_notices
    def _frame_notices() -> dict[str, Optional[str]]:
        notices: dict[str, Optional[str]] = {}
        for label, sf in frames.items():
            body = sf.render_notice()
            notices[label] = body if body else None
        return notices

    # ---- commands ---------------------------------------------------------------------

    @chan.build.command(name="load", always_observe=False)
    async def load(path: str, with_answers: bool = False) -> str:
        """Load a frame file from under the root, and make it the current frame.

        :param path: the ``.frame.yml`` file, absolute or relative to the frame root.
            Paths that resolve outside the root are rejected.
        :param with_answers: keep answers dumped in the file. Default drops them, giving
            you a fresh session self-model on a familiar question set.
        """
        resolved = await asyncio.to_thread(_resolve_inside_root, path)
        if not resolved.is_file():
            raise ValueError(f"no such frame file: {resolved}")
        frame = await asyncio.to_thread(_read_frame_file, resolved)
        if not with_answers:
            frame.answers = {}
        if frame.label in frames:
            return f"[{frame.label}] already loaded"
        frames[frame.label] = SessionFrame(frame)
        return f"loaded [{frame.label}] {len(frame.answers)}/{len(frame.questions)}"

    def _define_doc() -> str:
        schema = json.dumps(Frame.model_json_schema(), ensure_ascii=False, indent=2)
        return (
            "Define a frame in place from a JSON body and load it — no disk read.\n"
            "\n"
            "The body goes in the tag content as JSON matching this schema:\n"
            f"{schema}\n"
            "\n"
            "Wrap the body in `<![CDATA[ ... ]]>` — the JSON contains braces and quotes "
            "that would otherwise confuse the CTML parser.\n"
            "\n"
            ":param text__: the JSON body — a Frame object.\n"
            ":return: a short status line like ``defined [label] 0/N``."
        )

    @chan.build.command(name="define", always_observe=False, doc=_define_doc)
    async def define(text__: str) -> str:
        text = (text__ or "").strip()
        if not text:
            raise ValueError("define body is empty")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"define body is not valid JSON: {exc}") from exc
        try:
            frame = Frame.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"define body does not match Frame schema: {exc}") from exc
        if not frame.questions:
            raise ValueError("frame must have at least one question")
        if frame.label in frames:
            return f"[{frame.label}] already loaded"
        frames[frame.label] = SessionFrame(frame)
        return f"defined [{frame.label}] {len(frame.answers)}/{len(frame.questions)}"

    @chan.build.command(name="resolve", always_observe=False)
    async def resolve(question_index: int, answer: str, label: str = "") -> str:
        """Resolve one question with a short statement extracted from your context.
        ``unknown`` is a valid answer; resolving again overwrites.

        :param question_index: the question's 0-based index within the frame.
        :param answer: your answer, extracted from context.
        :param label: frame label. Empty = the most recently loaded frame.
        """
        sf = _find(label)
        if sf is None:
            raise ValueError(f"no such frame: {label!r}")
        if not 0 <= question_index < len(sf.frame.questions):
            raise ValueError(
                f"question index {question_index} out of range for [{sf.label}] "
                f"({len(sf.frame.questions)} questions)"
            )
        sf.frame.answers[question_index] = answer
        if sf.complete:
            return f"[{sf.label}] complete"
        done = len(sf.frame.questions) - len(sf.unresolved)
        return f"[{sf.label}] {done}/{len(sf.frame.questions)}"

    @chan.build.command(name="resolved", always_observe=True)
    async def resolved(label: str = "") -> str:
        """Pull the answers you have given so far for a frame.

        :param label: frame label. Empty = the most recently loaded frame.
        """
        sf = _find(label)
        if sf is None:
            return "no frame loaded" if not frames else f"no such frame: {label!r}"
        answers = sf.frame.answers
        if not answers:
            return f"[{sf.label}] nothing resolved"
        lines = [f"[{sf.label}] {len(answers)}/{len(sf.frame.questions)} resolved"]
        for i in sorted(answers):
            lines.append(f"  [{i}] {sf.frame.questions[i]}\n     → {answers[i]}")
        return "\n".join(lines)

    @chan.build.command(name="reset", always_observe=False)
    async def reset(label: str = "") -> str:
        """Drop all answers on a frame, keeping the questions. A cheap, reversible clear
        when the situation has moved and the current answers no longer describe it.

        :param label: frame label. Empty = the most recently loaded frame.
        """
        sf = _find(label)
        if sf is None:
            raise ValueError(f"no such frame: {label!r}")
        sf.frame.answers = {}
        return f"[{sf.label}] reset"

    @chan.build.command(name="export", always_observe=False)
    async def export(label: str = "", filename: str = "") -> str:
        """Persist a loaded frame to ``<root>/<filename>.frame.yml``.

        Exports the frame as-is, including any answers — the on-disk record is one
        artefact, not a template. Callers that want a template load it with
        ``with_answers=False``.

        :param label: frame label to export. Empty = the most recently loaded frame.
        :param filename: destination stem, relative to root, without the suffix.
            Empty = use the frame's label.
        """
        sf = _find(label)
        if sf is None:
            raise ValueError(f"no such frame: {label!r}")
        stem = (filename or sf.label).strip()
        if not stem:
            raise ValueError("empty destination filename")
        rel = stem if stem.endswith(FRAME_SUFFIX) else f"{stem}{FRAME_SUFFIX}"
        target = await asyncio.to_thread(_resolve_inside_root, rel)
        await asyncio.to_thread(_write_frame_file, target, sf.frame)
        try:
            display = target.relative_to(root_path).as_posix()
        except ValueError:
            display = str(target)
        return f"exported [{sf.label}] -> {display}"

    return chan


def new_frame_channel_from_file(
    root: str | Path,
    init_frame_file: str | Path,
    *,
    refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
    name: str = "frame",
    description: str = "Question-set as a thinking frame — a session self-model that re-asserts itself",
) -> MutableChannel:
    """Convenience: read a frame file into a :class:`Frame` and hand it to
    :func:`new_frame_channel`. The file must live inside ``root``.

    Answers in the file are dropped — an init frame is a fresh self-model.
    """
    root_path = Path(root).expanduser().resolve()
    file_path = Path(str(init_frame_file)).expanduser()
    file_path = file_path if file_path.is_absolute() else root_path / file_path
    file_path = file_path.resolve()
    try:
        file_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"init_frame_file escapes frame root: {file_path}") from exc
    if not file_path.is_file():
        raise ValueError(f"no such frame file: {file_path}")
    data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"frame file must be a yaml mapping: {file_path}")
    frame = Frame.model_validate(data)
    frame.answers = {}
    return new_frame_channel(
        root=root_path,
        init_frame=frame,
        refresh_seconds=refresh_seconds,
        name=name,
        description=description,
    )
