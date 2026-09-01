"""Prompt reading util — resolve @-file references into a list of Messages.

The @ file protocol: a prompt may carry ``@path`` references inline — like a
mention. A reference is ``@`` followed by either a quoted path (``@"my
file.txt"`` / ``@'my file.txt'``) or a bare run of non-whitespace characters
(``@app.py``). Each reference resolves to a ``Message`` via
``message_from_file``: its ``contents`` are the file's anthropic-compatible
content (Text / Base64Image), and when ``expose_file_meta`` is set (the
caller's external flag — file exposable) the Message carries the meta layer
(``tag="file"`` + path/type/size attributes), rendered by
``as_contents(with_meta=True)`` as an XML file block.

Unresolvable references stay inline as the plain string the caller typed —
the fallback is "if @ doesn't hit a file, keep it as-is". A bare reference
strips trailing sentence punctuation before resolution, so ``@app.py.``
resolves ``app.py``; a quoted path is taken verbatim. Thinking does NOT go
through @ — it is a separate call parameter, injected as history.
"""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from ghoshell_moss.message import Base64Image, Content, Message, Text

__all__ = ["message_from_prompt", "message_from_file"]

# @ followed by a quoted path or a bare non-whitespace run. The `@` must not be
# preceded by a non-whitespace char, so ``hello@world`` stays plain text.
_AT_REF_RE = re.compile(r"(?<!\S)@(?:\"([^\"\n]*)\"|'([^'\n]*)'|([^\s\"']+))")
# trailing sentence punctuation / whitespace that should not belong to a path
_TRAILING_PUNCT_RE = re.compile(r"[\s.,;:!?'\"()\[\]{}<>]+$")


def message_from_prompt(
        text: str,
        *,
        base_dir: str | Path | None = None,
        expose_file_meta: bool = False,
) -> list[Message]:
    """Parse a prompt string, resolving inline @-file references into Messages.

    Each ``@path`` (bare token or quoted) is resolved via ``message_from_file``.
    Resolved files become Messages; unresolvable references stay inline as the
    plain string. Text around references is preserved. ``base_dir`` — relative
    refs resolve against this (default cwd).
    """
    messages: list[Message] = []
    last = 0
    for m in _AT_REF_RE.finditer(text):
        if m.start() > last:
            messages.append(Message.new().with_content(text[last:m.start()]))
        if m.group(1) is not None:
            raw = m.group(1)
        elif m.group(2) is not None:
            raw = m.group(2)
        else:
            raw = _TRAILING_PUNCT_RE.sub("", m.group(3))
        if not raw:
            messages.append(Message.new().with_content(m.group(0)))
            last = m.end()
            continue
        msg = message_from_file(
            raw, base_dir=base_dir, expose_file_meta=expose_file_meta,
        )
        if msg is None:
            messages.append(Message.new().with_content(m.group(0)))
        else:
            messages.append(msg)
        last = m.end()
    if last < len(text):
        messages.append(Message.new().with_content(text[last:]))
    return messages


def message_from_file(
        path: str | Path,
        *,
        base_dir: str | Path | None = None,
        expose_file_meta: bool = False,
) -> Message | None:
    """Convert a file into a moss Message, or None if not resolvable.

    ``contents`` are anthropic-compatible: text files → Text, images →
    Base64Image, other types → no content. With ``expose_file_meta`` the
    Message carries the meta layer (``tag="file"`` + path/type/size) and
    unsupported types are still exposed as a bare file block; without it,
    text and image stay bare content and unsupported types are dropped.
    """
    base = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
    display = str(path)
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = base / p
    p = p.resolve()
    if not p.is_file():
        return None

    media_type, _ = mimetypes.guess_type(str(p))
    content = _file_content(p, media_type)
    if not expose_file_meta:
        if content is None:
            return None
        return Message.new().with_content(content)

    msg = Message.new(tag="file", attributes=_file_meta(display, media_type, p))
    if content is not None:
        msg.with_content(content)
    return msg


def _file_content(path: Path, media_type: str | None) -> Content | None:
    """File → anthropic-compatible Content; None for unsupported types."""
    if media_type and media_type.startswith("image/"):
        return Base64Image.from_file(path).to_content()
    if media_type and media_type.startswith("text/"):
        return Text.new(path.read_text(encoding="utf-8", errors="replace")).to_content()
    if media_type:
        # known but not text/image (application / audio / video) → unsupported
        return None
    # unknown type — try to read as utf-8 text
    try:
        path.read_bytes().decode("utf-8")
        return Text.new(path.read_text(encoding="utf-8", errors="replace")).to_content()
    except UnicodeDecodeError:
        return None


def _file_meta(display_path: str, media_type: str | None, resolved: Path) -> dict[str, str]:
    """File basic info (file foo.py style) — path / type / size."""
    attrs: dict[str, str] = {"path": display_path}
    if media_type:
        attrs["type"] = media_type
    attrs["size"] = str(resolved.stat().st_size)
    return attrs
