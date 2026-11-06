"""The transcript view — the conversation as a *reading*, not as a request.

A session knows two different things about its own history, and the difference matters:

- **what a model would see next**, assembled from the surface: every tool call with its result
  attached, in the shape the provider accepts. That is `Session.deriveMessages()`, and it is
  built for the next request.
- **what happened**, told short: what was asked, what was answered, which tools were reached
  for. Tool *results* are dropped — they are usually longer than the conversation they were
  fetched for, and they answer "what came back", not "what we did". Reasoning is dropped too: it
  is the model's private scratch, not the record.

The second one is this view. Keeping the two apart is the point: reaching for `deriveMessages()`
to show somebody what a conversation contained is how a reading turns into a prompt, and the
first symptom is a transcript nobody can skim.

### What it leaves out, and why each exclusion is deliberate

| event | shown? | why |
|---|---|---|
| `user/message` (source.kind == "user") | `>` as written | the actual input |
| `user/message` (anything else) | no | synthetic context — file notices, skill text, cron pings |
| `assistant/message` | `~`, text blocks only | reasoning and usage frames are not part of the record |
| `tool/call` | `@` name + arguments (truncated) | what was reached for |
| `tool/result` | no | what came back — usually the bulk of the bytes, and not the story |
| anything else | no | turn/step boundaries, chunks, headers, todo — mechanism |

### Who reads it

Not the session: it belongs to whoever wants a reading of that session. `TranscriptView.read(session)`
pages the events in and renders them; the session is an argument, not an owner. That keeps the
view usable for a ref, a fork, a live session, or anything else that can page its history.
"""

from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .models.session_events import SessionEvent

if TYPE_CHECKING:
    from .session import DshAgentSession

__all__ = ["TranscriptView"]

_TURN_START = "turn/start"


def _content_text(content: Any) -> str:
    """Join the text blocks of a content list; everything else is dropped on purpose."""
    blocks = content if isinstance(content, list) else []
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "".join(parts)


def _clip(text: str, limit: int) -> str:
    """Cap an argument list — a call is a hint, not the payload.

    Newlines are kept: a JSON argument reads better as the model wrote it, and the renderer
    already knows how to fold continuation lines under the marker.
    """
    return text if limit <= 0 or len(text) <= limit else text[: limit - 1] + "…"


class TranscriptView(BaseModel):
    """One reading of a conversation: the options that shaped it, then the result.

    The same object is both — it is constructed with options, `read()` fills in what came out.
    That is deliberate: a reading whose options are not recorded with it cannot be reproduced,
    and a caller comparing two transcripts needs to know whether they differ in the content or
    in the shape.
    """

    model_config = ConfigDict(extra="allow")

    # ---- how to read ---- #
    limit_turns: int | None = Field(
        default=None, description="keep only the newest N turns; None = all of them."
    )
    include_tool_calls: bool = Field(default=True, description="render `@` lines at all.")
    tool_arguments_limit: int = Field(
        default=160, description="characters of one call's arguments to keep; <= 0 = keep all."
    )
    indent: str = Field(default="  ", description="left margin of one rendered line.")
    page_events: int = Field(
        default=200, description="how many events to ask for per history() page."
    )

    # ---- what came out ---- #
    text: str = Field(default="", description="the rendered reading.")
    turns: int = Field(default=0, description="how many turns the reading covers.")
    messages: int = Field(default=0, description="how many user/assistant lines it contains.")
    tool_calls: int = Field(default=0, description="how many tool calls it mentions.")

    def render(self, events: list[SessionEvent]) -> TranscriptView:
        """Render already-loaded events into this view. Pure: no session, no I/O."""
        turns: list[list[tuple[str, str]]] = []
        current: list[tuple[str, str]] = []

        def flush() -> None:
            if current:
                turns.append(current)

        for event in events:
            if event.meta.type == _TURN_START:
                flush()
                current = []
                continue
            line = self._line_for(event)
            if line is not None:
                current.append(line)
        flush()

        if self.limit_turns is not None:
            turns = turns[-self.limit_turns:] if self.limit_turns > 0 else []

        # Counts describe the reading that was produced, not the events that happened to be
        # loaded: paging may overshoot `limit_turns`, and a transcript whose counters disagree
        # with its own text is worse than having no counters.
        kept = [line for group in turns for line in group]
        self.text = "\n".join(starmap(self._fold, kept))
        self.turns = len(turns)
        self.messages = sum(1 for marker, _ in kept if marker != "@")
        self.tool_calls = sum(1 for marker, _ in kept if marker == "@")
        return self

    @classmethod
    async def read(cls, session: DshAgentSession, **options: Any) -> TranscriptView:
        """Page a session's history in, then render it.

        Pages go backwards (that is how `history()` pages) and are reassembled in order. When
        `limit_turns` is set, paging stops as soon as enough turns are in hand — a reading of the
        last three turns should not have to pull the whole conversation to answer.
        """
        view = cls(**options)
        pages: list[list[SessionEvent]] = []
        before_seq: int | None = None
        seen_turns = 0

        while True:
            page = await session.history(before_seq=before_seq, max_messages=view.page_events)
            if not page:
                break
            pages.append(page)
            before_seq = page[0].meta.seq
            if before_seq <= 0:
                break
            if view.limit_turns is not None:
                seen_turns += sum(1 for e in page if e.meta.type == _TURN_START)
                if seen_turns > view.limit_turns:
                    break

        events = [event for page in reversed(pages) for event in page]
        return view.render(events)

    # ---- rendering primitives ---- #

    def _line_for(self, event: SessionEvent) -> tuple[str, str] | None:
        """One event → (marker, body), or None when it is not part of the reading."""
        data = event.data or {}
        kind = event.meta.type

        if kind == "user/message":
            source = data.get("source") or {}
            if source.get("kind") != "user":
                return None
            text = _content_text(data.get("content")).rstrip()
            return (">", text) if text else None

        if kind == "assistant/message":
            message = data.get("message") or {}
            text = _content_text(message.get("content")).rstrip()
            return ("~", text) if text else None

        if kind == "tool/call" and self.include_tool_calls:
            name = data.get("name") or ""
            arguments = _clip(data.get("arguments") or "", self.tool_arguments_limit)
            return ("@", f"{name}({arguments})")

        return None

    def _fold(self, marker: str, body: str) -> str:
        head = f"{self.indent}{marker} "
        return head + body.replace("\n", "\n" + " " * len(head))
