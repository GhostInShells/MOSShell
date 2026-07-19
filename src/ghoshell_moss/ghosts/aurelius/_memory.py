"""Ghost-side Memento adapter: persistent frames in, model history out."""

import re
import threading
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import TextContent, UserContent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from ghoshell_moss.core.blueprint.mindflow import Moment
from ghoshell_moss.core.memento import (
    CommitView,
    Memento,
    MementoBranch,
    MomentRecord,
    join_trailers,
    new_filesystem_memento,
)
from ghoshell_moss.core.memento.porcelain import (
    record_to_moment,
    update_moment,
    window_messages,
)
from ghoshell_moss.message import Message

from ._adapter import messages_to_parts

__all__ = ["AureliusMemory", "SearchHit"]


class SearchHit(BaseModel):
    """One plaintext match in the owner's trajectory — an address, not a summary."""

    moment_id: str
    commit_id: str | None = None
    commit_seq: int | None = None
    frozen: bool
    role: str
    snippet: str


_REFLECTION_BY = "memento-reflection"
_DEFAULT_USER_SOURCES = ("input_signal_nucleus", "input", "user")
_MECHANICAL_SUMMARY_MAX_CHARS = 600
_MECHANICAL_INPUT_MAX_CHARS = 140
_MECHANICAL_REPLY_MAX_CHARS = 140
_PAIRED_COMMAND_RE = re.compile(
    r"<(?P<name>[A-Za-z_][\w.-]*:[\w.-]+)\b[^>]*>.*?</(?P=name)\s*>",
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)


def _compact_visible_text(text: str, limit: int) -> str:
    visible = text
    previous = None
    while visible != previous:
        previous = visible
        visible = _PAIRED_COMMAND_RE.sub("", visible)
    visible = _TAG_RE.sub("", visible)
    return " ".join(visible.split())[:limit]


def _escape_memento_body(text: str) -> str:
    """A summary is reflected/user-derived text; never let it forge a <memento> boundary."""
    return text.replace("<", "‹").replace(">", "›")


class AureliusMemory:
    """Single-owner persistent conversation memory for one Aurelius Ghost."""

    def __init__(
        self,
        root: str | Path,
        owner: str,
        *,
        detail_n: int = 12,
        summary_m: int = -1,
        auto_commit_every: int = 4,
        index_user_sources: tuple[str, ...] = _DEFAULT_USER_SOURCES,
    ) -> None:
        if detail_n <= 0:
            raise ValueError("detail_n must be greater than zero")
        if summary_m < -1:
            raise ValueError("summary_m must be -1 or greater")
        if auto_commit_every < 0:
            raise ValueError("auto_commit_every must be zero or greater")
        self._root = Path(root)
        self._owner = owner
        self._detail_n = detail_n
        self._summary_m = summary_m
        self._auto_commit_every = auto_commit_every
        self._index_user_sources = tuple(index_user_sources)
        self._memento: Memento = new_filesystem_memento(self._root, owner)
        self._branch: MementoBranch = self._memento.current()
        self._closed = False
        # The single-writer contract binds one (root, owner) to one instance, but this
        # one instance is written from two thread domains: the event loop (remember /
        # reflection) and the to_thread workers that run CTML control commands. Serialize
        # every mutation and every branch-pointer read through a reentrant lock so
        # duplex articulate/action can't interleave a staging append with a commit
        # truncate, or swap self._branch mid-render.
        self._lock = threading.RLock()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def branch(self) -> MementoBranch:
        return self._branch

    @property
    def detail_n(self) -> int:
        return self._detail_n

    @property
    def summary_m(self) -> int:
        return self._summary_m

    def messages(self) -> list[Message]:
        with self._lock:
            window = self._branch.window(
                detail_n=self._detail_n,
                summary_m=self._summary_m,
            )
            return list(window_messages(self._branch, window))

    def model_history(
        self,
        *,
        detail_n: int | None = None,
        summary_m: int | None = None,
    ) -> list[ModelMessage]:
        """Render the current branch as valid pydantic-ai history.

        ``detail_n``/``summary_m`` override the configured window for this render
        only — the context-budget loop uses them to shrink what enters the model
        context without touching persisted policy or any frozen record.
        """
        with self._lock:
            return self._render_history(
                detail_n=self._detail_n if detail_n is None else detail_n,
                summary_m=self._summary_m if summary_m is None else summary_m,
            )

    def _render_history(self, *, detail_n: int, summary_m: int) -> list[ModelMessage]:
        window = self._branch.window(
            detail_n=detail_n,
            summary_m=summary_m,
        )
        history: list[ModelMessage] = []

        # Folded commits render as a stamped preamble. We do NOT fabricate a model
        # response to acknowledge them — a turn the model never uttered has no place
        # in its own history. Instead the preamble rides on the first real user turn
        # (or stands alone as a leading request when the window has no detail turns).
        preamble: list[UserContent] = []
        if window.summaries:
            lines = [
                f'<memento commit="{view.id}" seq="{view.seq}" '
                f'note_seq="{view.note_seq}" kind="{view.note.kind()}">'
                f"{_escape_memento_body(view.summary() or '[mechanical checkpoint]')}"
                "</memento>"
                for view in window.summaries
            ]
            preamble.append(TextContent(content="\n".join(lines)))

        moments = [record_to_moment(record) for record in window.details]
        for messages, logos in Moment.to_history_turns(moments):
            parts = messages_to_parts(messages)
            if preamble:
                parts = preamble + parts
                preamble = []
            if parts:
                history.append(ModelRequest(parts=[UserPromptPart(content=parts)]))
            if logos:
                history.append(ModelResponse(parts=[TextPart(content=logos)]))

        if preamble:  # summaries with no detail turns — a lone leading request is valid.
            history.append(ModelRequest(parts=[UserPromptPart(content=preamble)]))
        return history

    def remember(self, moment: Moment, *, threads: tuple[str, ...] = ()) -> CommitView | None:
        """Persist a completed frame and optionally create a mechanical anchor.

        ``threads`` tags the staged record (e.g. ``("failed",)`` for a frame whose
        articulation raised). Tagging keeps a failed frame witnessed — "saw X, tried
        to answer, errored" is a trajectory event, not something to erase — while
        letting readers tell it apart from a completed turn.
        """
        with self._lock:
            update_moment(self._branch, moment, threads=threads, by=self._owner)
            if self._auto_commit_every > 0 and len(self._branch.staging()) >= self._auto_commit_every:
                return self._branch.commit(self._mechanical_summary(), kind="mechanical", by=self._owner)
            return None

    def reflection_candidates(self) -> list[CommitView]:
        """Own unreflected mechanical commits, plus legacy commits with an empty note."""
        with self._lock:
            candidates: list[CommitView] = []
            for view in self._branch.own_commits():
                is_legacy_empty = not view.summary().strip()
                if view.note.kind() != "mechanical" and not is_legacy_empty:
                    continue
                if any(note.by == _REFLECTION_BY for note in self._branch.notes(view.id)):
                    continue
                candidates.append(view)
            return candidates

    def apply_reflection(self, commit_id: str, summary: str) -> CommitView:
        """Append an LLM-derived interpretation without touching frozen Moment records."""
        with self._lock:
            view = self.find_commit(commit_id)
            if view is None:
                raise ValueError(f"commit not found or ambiguous: {commit_id!r}")
            body = join_trailers(summary.strip(), [("Kind", view.note.kind()), ("Reflection", "llm")])
            return self._branch.reinterpret(commit_id, body, by=_REFLECTION_BY)

    def commit_transcript(self, commit_id: str, *, max_chars: int) -> str:
        """Bounded observable source for a reflector; never exposes hidden model reasoning."""
        lines: list[str] = []
        with self._lock:
            for record in self._branch.commit_records(commit_id):
                moment = record_to_moment(record)
                for messages in moment.percepts.values():
                    for message in messages:
                        text = message.to_content_string().strip()
                        if text:
                            lines.append(f"input: {text}")
                if moment.logos.strip():
                    lines.append(f"logos: {moment.logos.strip()}")
        return "\n".join(lines)[:max_chars]

    def find_commit(self, token: str) -> CommitView | None:
        """Resolve a stable sequence number or an unambiguous commit-id prefix."""
        matched = [
            view for view in self._branch.all_commits()
            if token == str(view.seq) or view.id.startswith(token)
        ]
        return matched[0] if len(matched) == 1 else None

    def semantic_commit(self, summary: str) -> CommitView:
        if not summary.strip():
            raise ValueError("semantic commit summary cannot be empty")
        with self._lock:
            if not self._branch.staging():
                raise ValueError("semantic commit requires at least one staged Moment")
            return self._branch.commit(summary.strip(), kind="semantic", by=self._owner)

    def reinterpret(self, token: str, summary: str) -> CommitView:
        if not summary.strip():
            raise ValueError("reinterpret summary cannot be empty")
        with self._lock:
            view = self.find_commit(token)
            if view is None:
                raise ValueError(f"commit not found or ambiguous: {token!r}")
            body = join_trailers(summary.strip(), [("Kind", view.note.kind())])
            return self._branch.reinterpret(view.id, body, by=self._owner)

    def fork(self, token: str, name: str = "") -> MementoBranch:
        with self._lock:
            view = self.find_commit(token)
            if view is None:
                raise ValueError(f"commit not found or ambiguous: {token!r}")
            child = self._memento.checkout(
                base_fork=self._branch.meta.fork,
                base_branch_id=self._branch.meta.branch_id,
                base_commit_id=view.id,
                name=name or f"fork-of-{view.seq}",
            )
            self._memento.switch(child.meta.branch_id)
            self._branch = child
            return child

    def switch(self, prefix: str) -> MementoBranch:
        with self._lock:
            matched = [meta for meta in self._memento.list_branches() if meta.branch_id.startswith(prefix)]
            if len(matched) != 1:
                raise ValueError(f"branch not found or ambiguous: {prefix!r}")
            self._memento.switch(matched[0].branch_id)
            self._branch = self._memento.get_branch(matched[0].branch_id)
            return self._branch

    def branches(self) -> list[dict[str, str]]:
        with self._lock:
            current = self._branch.meta.branch_id
            return [
                {"id": meta.branch_id, "name": meta.name, "current": str(meta.branch_id == current)}
                for meta in self._memento.list_branches()
            ]

    def describe_commit(self, token: str) -> str:
        with self._lock:
            view = self.find_commit(token)
            if view is None:
                raise ValueError(f"commit not found or ambiguous: {token!r}")
            lines = [f"commit seq={view.seq} id={view.id} kind={view.note.kind()}", f"summary: {view.summary()}"]
            for record in self._branch.commit_records(view.id):
                moment = record_to_moment(record)
                inputs = " | ".join(moment.percepts_texts())
                lines.append(f"moment={moment.id} input={inputs} logos={moment.logos}")
            return "\n".join(lines)

    def search(self, keyword: str, *, limit: int = 20, window: int = 80) -> list[SearchHit]:
        """Plaintext grep over this owner's trajectory — ``cat``/``grep`` as the query language.

        No semantic parsing, no canonical keys: a case-insensitive substring scan over the
        raw Moment text of every frozen commit plus current staging. Returns stable addresses
        (commit_id/moment_id) the model can expand with ``memory_show``.
        """
        needle = keyword.strip()
        if not needle:
            raise ValueError("search keyword cannot be empty")
        folded = needle.casefold()
        hits: list[SearchHit] = []
        for record, view in self._records_newest_first():
            if len(hits) >= limit:
                break
            moment = record_to_moment(record)
            for role, text in (
                ("input", " ".join(moment.percepts_texts())),
                ("logos", moment.logos),
            ):
                index = text.casefold().find(folded)
                if index < 0:
                    continue
                start = max(0, index - window)
                snippet = " ".join(text[start : index + len(needle) + window].split())
                hits.append(
                    SearchHit(
                        moment_id=moment.id,
                        commit_id=view.id if view is not None else None,
                        commit_seq=view.seq if view is not None else None,
                        frozen=view is not None,
                        role=role,
                        snippet=snippet,
                    )
                )
                if len(hits) >= limit:
                    break
        return hits

    def _records_newest_first(self) -> list[tuple[MomentRecord, CommitView | None]]:
        """Staging first (most recent), then frozen commits newest-first; each Moment once."""
        pairs: list[tuple[MomentRecord, CommitView | None]] = []
        seen: set[str] = set()
        with self._lock:
            for record in reversed(self._branch.staging()):
                if record.id not in seen:
                    pairs.append((record, None))
                    seen.add(record.id)
            for view in reversed(self._branch.all_commits()):
                for record in reversed(self._branch.commit_records(view.id)):
                    if record.id not in seen:
                        pairs.append((record, view))
                        seen.add(record.id)
        return pairs

    def _mechanical_summary(self) -> str:
        """Build a globally bounded user-facing index without internal control turns."""
        records = self._branch.staging()
        lines = [f"[extractive mechanical index] moments={len(records)}"]
        for record in records:
            moment = record_to_moment(record)
            inputs = " ".join(
                message.to_content_string()
                for source in self._index_user_sources
                for message in moment.percepts.get(source, [])
            )
            user_text = _compact_visible_text(inputs, _MECHANICAL_INPUT_MAX_CHARS)
            if not user_text:
                continue
            reply_text = _compact_visible_text(moment.logos, _MECHANICAL_REPLY_MAX_CHARS)
            line = f"- user={user_text}"
            if reply_text:
                line += f" reply={reply_text}"
            lines.append(line)

        body = "\n".join(lines)
        if len(body) <= _MECHANICAL_SUMMARY_MAX_CHARS:
            return body
        suffix = "\n[truncated]"
        return body[: _MECHANICAL_SUMMARY_MAX_CHARS - len(suffix)].rstrip() + suffix

    def inspect(self) -> dict:
        head = self._branch.head()
        return {
            "root": str(self._root),
            "owner": self._owner,
            "branch_id": self._branch.meta.branch_id,
            "staging_count": len(self._branch.staging()),
            "commit_count": len(self._branch.own_commits()),
            "head_commit_id": head.id if head else None,
            "detail_n": self._detail_n,
            "summary_m": self._summary_m,
            "auto_commit_every": self._auto_commit_every,
            "reflection_pending": len(self.reflection_candidates()),
        }

    def close(self) -> None:
        if not self._closed:
            self._memento.close()
            self._closed = True
