"""Question-set as thinking framework — extract a structured self-model from context | meta | beta

A **frame** is a question set the model resolves from its own context, to hold a stable
situational understanding across context loss (compaction, a new window, a different
model instance). Frames are plain ``*.frame.md`` files discovered by suffix under a
root path — no registry, no explicit link between them: the filesystem is the index.
This channel loads frames, tracks how far each is resolved, and re-presents them as
context. The questions are the asset; the answers are working state.

A frame is an extraction device, not a decision device: every question is answerable
from context, and an explicit ``unknown`` is a first-class resolution — the unresolved
set is the frame's most valuable output. The format is specified in
``frames/SPECIFICATION.md`` (returned by the ``spec`` command).

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.frame_channel import new_frame_channel

    main = new_shell_main_channel()
    main.import_channels(new_frame_channel(root="frames", entry="orientation.frame.md"))
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Optional

import yaml

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

__all__ = ["new_frame_channel", "FRAME_SUFFIX"]

FRAME_SUFFIX = ".frame.md"
"""A file is a frame iff its name ends with this suffix."""

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n")

_INSTRUCTION = """\
## Frame — 问题集即思维框架

Your situation, held as a set of questions you resolve from your own context. Asking
the right question makes an answer that already exists in your context explicit and
ready to reason on. A frame is an extraction device: every question is answerable from
context, and `unknown` is a valid, useful resolution — the unresolved questions are
your blind spots, and they are the point.

The loaded frames and their answers are shown below. Resolve questions as they become
answerable; overwrite them as your understanding sharpens. Frames are plain files:
`list` discovers them under the root, `template` emits a starter to author your own,
and `reload` re-reads an edited file from disk. The format spec is available on demand.
"""

_FRAME_TEMPLATE = """\
---
description: <one line — what this frame orients>
nexts:
  - <relative path to a follow-up frame, optional>
---

<question — answerable from context>

<question — answerable from context>
"""


class _Frame:
    """A loaded frame and its live resolution state."""

    def __init__(
        self,
        *,
        path: Path,
        label: str,
        description: str,
        nexts: list[str],
        questions: list[str],
    ) -> None:
        self.path = path
        self.label = label
        self.description = description
        self.nexts = nexts
        self.questions = questions
        self.answers: list[Optional[str]] = [None] * len(questions)

    @property
    def resolved_count(self) -> int:
        return sum(1 for answer in self.answers if answer is not None)

    @property
    def complete(self) -> bool:
        return self.resolved_count == len(self.questions)


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split an optional leading YAML frontmatter block from the body."""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ValueError(f"malformed frontmatter: {exc}") from exc
    if not isinstance(data, dict):
        data = {}
    return data, text[match.end():]


def _parse_questions(body: str) -> list[str]:
    """Body paragraphs, blank-line separated — each paragraph is one question."""
    return [p.strip() for p in _PARAGRAPH_RE.split(body.strip()) if p.strip()]


def _label_of(path: Path, root: Path) -> str:
    """Label = path relative to root, suffix stripped. Falls back to the stem."""
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.stem
    return rel[: -len(FRAME_SUFFIX)] if rel.endswith(FRAME_SUFFIX) else rel


def _root_relative(target: Path, root: Path) -> str:
    try:
        return target.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(target)


def _read_spec() -> str:
    path = Path(__file__).parent / "frames" / "SPECIFICATION.md"
    if not path.is_file():
        return "[frame] SPECIFICATION.md not found"
    return path.read_text(encoding="utf-8")


def _read_frame(path: Path, root: Path) -> _Frame:
    frontmatter, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    description = str(frontmatter.get("description") or "")
    raw_nexts = frontmatter.get("nexts") or []
    if isinstance(raw_nexts, str):
        raw_nexts = [raw_nexts]
    # nexts are relative to this file's directory; store them root-relative so the
    # hint the model gets back is directly usable by load().
    nexts = [_root_relative(path.parent / str(n), root) for n in raw_nexts]
    return _Frame(
        path=path,
        label=_label_of(path, root),
        description=description,
        nexts=nexts,
        questions=_parse_questions(body),
    )


def new_frame_channel(
    root: str | Path,
    entry: str | Path | None = None,
    *,
    name: str = "frame",
    description: str = "Question-set as thinking framework — extract a structured self-model from context",
) -> MutableChannel:
    """Build a frame channel over a root path.

    :param root: the directory frames are discovered under.
    :param entry: a frame file to load at build time (absolute, or relative to root).
        Everything else the model opens itself via ``load``.
    :param name: channel name.
    :param description: channel description.
    """
    root_path = Path(root).expanduser()
    frames: dict[str, _Frame] = {}
    state: dict[str, str] = {"current": "", "error": ""}

    def _resolve_path(path_like: str | Path) -> Path:
        p = Path(str(path_like)).expanduser()
        return p if p.is_absolute() else root_path / p

    def _find(handle: str) -> Optional[_Frame]:
        handle = (handle or "").strip()
        if not handle:
            current = state["current"]
            return frames.get(current) if current else None
        if handle.endswith(FRAME_SUFFIX):
            handle = handle[: -len(FRAME_SUFFIX)]
        return frames.get(handle)

    def _load(path_like: str | Path) -> tuple[_Frame, bool]:
        path = _resolve_path(path_like)
        if not path.is_file():
            raise ValueError(f"no such frame file: {path}")
        label = _label_of(path, root_path)
        existing = frames.get(label)
        if existing is not None:
            state["current"] = label
            return existing, False
        frame = _read_frame(path, root_path)
        frames[frame.label] = frame
        state["current"] = frame.label
        return frame, True

    def _reload(path_like: str | Path) -> _Frame:
        path = _resolve_path(path_like)
        if not path.is_file():
            raise ValueError(f"no such frame file: {path}")
        frame = _read_frame(path, root_path)
        frames[frame.label] = frame
        state["current"] = frame.label
        return frame

    def _discover() -> list[str]:
        if not root_path.is_dir():
            return []
        return sorted(
            _label_of(path, root_path)
            for path in root_path.rglob(f"*{FRAME_SUFFIX}")
            if path.is_file()
        )

    if entry is not None:
        try:
            _load(entry)
        except Exception as exc:  # surface at notice; don't break the channel tree
            state["error"] = str(exc)

    chan = new_channel(name=name, description=description)

    def _render_instruction() -> str:
        blocks = [_INSTRUCTION]
        if state["error"]:
            blocks.append(f"[frame] {state['error']}")
        for label, frame in frames.items():
            head = f"[{label}] {frame.resolved_count}/{len(frame.questions)}"
            if frame.description:
                head += f" — {frame.description}"
            blocks.append(head)
            for index, question in enumerate(frame.questions):
                answer = frame.answers[index]
                if answer is None:
                    blocks.append(f"  {index}. {question}")
                else:
                    blocks.append(f"  {index}. {question}\n     → {answer}")
            if frame.complete and frame.nexts:
                blocks.append(f"  nexts: {', '.join(frame.nexts)}")
        return "\n\n".join(blocks)

    chan.build.instruction(_render_instruction)

    @chan.build.command(name="list", always_observe=True)
    async def list_frames() -> str:
        """List frame files under the root — loaded vs available.

        Loaded frames show their resolution progress; available-but-unloaded frames are
        shown bare. Load what you are about to work on.

        :return: every ``*.frame.md`` under the root.
        """
        discovered = await asyncio.to_thread(_discover)
        if not discovered:
            return f"[frame] no frames under {root_path}"
        loaded = set(frames)
        lines = [f"[frame] {len(discovered)} frame(s) under root; {len(loaded)} loaded"]
        for label in discovered:
            if label in loaded:
                frame = frames[label]
                lines.append(f"  loaded   {label}  ({frame.resolved_count}/{len(frame.questions)})")
            else:
                lines.append(f"  {label}")
        return "\n".join(lines)

    @chan.build.command(name="load", always_observe=False)
    async def load(path: str) -> str:
        """Load a frame file and make it the current frame.

        A frame is a ``*.frame.md`` file. The path may be absolute or relative to the
        frame root. Find frames by listing the root; load only what you are about to
        work on.

        :param path: the frame file to load.
        """
        frame, is_new = await asyncio.to_thread(_load, path)
        if not is_new:
            return f"[{frame.label}] already loaded"
        return f"loaded [{frame.label}] 0/{len(frame.questions)}"

    @chan.build.command(name="reload", always_observe=False)
    async def reload(path: str) -> str:
        """Re-read a frame file from disk, discarding its resolution state.

        Use after editing a frame's questions: the file is the index, so an edited file
        is picked up by re-reading. Answers reset — they are working state; the questions
        are the asset.

        :param path: the frame file to re-read (absolute, or relative to the root).
        """
        frame = await asyncio.to_thread(_reload, path)
        return f"reloaded [{frame.label}] 0/{len(frame.questions)}"

    @chan.build.command(name="template", always_observe=True)
    async def template() -> str:
        """Return a starter ``*.frame.md`` template to author a new frame.

        Copy it into a new file under the root, fill the description and questions, then
        ``load`` it. Questions are blank-line-separated paragraphs — see ``spec`` for the
        full format.
        """
        return _FRAME_TEMPLATE

    @chan.build.command(name="resolve", always_observe=False)
    async def resolve(label: str, question_index: int, answer: str) -> str:
        """Resolve one question of a loaded frame.

        Write the answer as a short statement extracted from your context. When the
        context does not answer the question, write ``unknown`` — an explicit unknown
        is a valid, useful resolution. Resolving again overwrites.

        :param label: the frame's label — its path under the root, suffix stripped
            (e.g. ``orientation`` or ``debug/concurrency``). Empty = the current frame.
        :param question_index: the question's 0-based index within the frame.
        :param answer: your answer, extracted from context.
        """
        frame = _find(label)
        if frame is None:
            raise ValueError(f"no such frame: {label!r}")
        if not 0 <= question_index < len(frame.questions):
            raise ValueError(
                f"question index {question_index} out of range for [{frame.label}] "
                f"({len(frame.questions)} questions)"
            )
        was_complete = frame.complete
        frame.answers[question_index] = answer
        if frame.complete and not was_complete:
            hint = ", ".join(frame.nexts) if frame.nexts else "none"
            return f"[{frame.label}] complete. nexts hint: {hint}"
        return "resolved"

    @chan.build.command(name="status", always_observe=True)
    async def status(label: str = "") -> str:
        """Show a frame's resolution status.

        Unresolved questions are listed first — they are the frame's blind spots, the
        most useful part of the reading.

        :param label: frame label — its path under the root, suffix stripped. Empty = the current frame.
        """
        frame = _find(label)
        if frame is None:
            return "no frame loaded" if not frames else f"no such frame: {label!r}"
        lines = [f"[{frame.label}] {frame.resolved_count}/{len(frame.questions)} resolved"]
        if frame.description:
            lines.append(f"  {frame.description}")
        unresolved = [i for i, answer in enumerate(frame.answers) if answer is None]
        if unresolved:
            lines.append("  unresolved (blind spots):")
            lines.extend(f"    {i}. {frame.questions[i]}" for i in unresolved)
        resolved = [i for i, answer in enumerate(frame.answers) if answer is not None]
        if resolved:
            lines.append("  resolved:")
            lines.extend(f"    {i}. {frame.questions[i]} → {frame.answers[i]}" for i in resolved)
        if frame.nexts:
            lines.append(f"  nexts: {', '.join(frame.nexts)}")
        return "\n".join(lines)

    @chan.build.command(name="spec", always_observe=True)
    async def spec() -> str:
        """Return the ``.frame.md`` format specification.

        Read this before authoring a frame of your own. Frames are plain files: create
        one with your file tools, then load it.
        """
        return await asyncio.to_thread(_read_spec)

    return chan
