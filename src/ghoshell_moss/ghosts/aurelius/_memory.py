"""Ghost-side Memento adapter: persistent frames in, model history out."""

from pathlib import Path

from pydantic_ai import TextContent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart

from ghoshell_moss.core.blueprint.mindflow import Moment
from ghoshell_moss.core.memento import (
    CommitView,
    Memento,
    MementoBranch,
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

__all__ = ["AureliusMemory"]

_REFLECTION_BY = "memento-reflection"


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
        self._memento: Memento = new_filesystem_memento(self._root, owner)
        self._branch: MementoBranch = self._memento.current()
        self._closed = False

    @property
    def root(self) -> Path:
        return self._root

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def branch(self) -> MementoBranch:
        return self._branch

    def messages(self) -> list[Message]:
        window = self._branch.window(
            detail_n=self._detail_n,
            summary_m=self._summary_m,
        )
        return list(window_messages(self._branch, window))

    def model_history(self) -> list[ModelMessage]:
        """Render the current branch as valid pydantic-ai history."""
        window = self._branch.window(
            detail_n=self._detail_n,
            summary_m=self._summary_m,
        )
        history: list[ModelMessage] = []

        if window.summaries:
            lines = [
                f'<memento commit="{view.id}" kind="{view.note.kind()}">'
                f"{view.summary() or '[mechanical checkpoint]'}"
                "</memento>"
                for view in window.summaries
            ]
            history.append(ModelRequest(parts=[TextContent(content="\n".join(lines))]))
            history.append(ModelResponse(parts=[TextPart(content="[memento summaries loaded]")]))

        moments = [record_to_moment(record) for record in window.details]
        for messages, logos in Moment.to_history_turns(moments):
            parts = messages_to_parts(messages)
            if parts:
                history.append(ModelRequest(parts=parts))
            if logos:
                history.append(ModelResponse(parts=[TextPart(content=logos)]))
        return history

    def remember(self, moment: Moment) -> CommitView | None:
        """Persist a completed frame and optionally create a mechanical anchor."""
        update_moment(self._branch, moment, by=self._owner)
        if self._auto_commit_every > 0 and len(self._branch.staging()) >= self._auto_commit_every:
            return self._branch.commit(self._mechanical_summary(), kind="mechanical", by=self._owner)
        return None

    def reflection_candidates(self) -> list[CommitView]:
        """Own unreflected mechanical commits, plus legacy commits with an empty note."""
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
        view = self.find_commit(commit_id)
        if view is None:
            raise ValueError(f"commit not found or ambiguous: {commit_id!r}")
        body = join_trailers(summary.strip(), [("Kind", view.note.kind()), ("Reflection", "llm")])
        return self._branch.reinterpret(commit_id, body, by=_REFLECTION_BY)

    def commit_transcript(self, commit_id: str, *, max_chars: int) -> str:
        """Bounded observable source for a reflector; never exposes hidden model reasoning."""
        lines: list[str] = []
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
        return self._branch.commit(summary.strip(), kind="semantic", by=self._owner)

    def reinterpret(self, token: str, summary: str) -> CommitView:
        view = self.find_commit(token)
        if view is None:
            raise ValueError(f"commit not found or ambiguous: {token!r}")
        body = join_trailers(summary.strip(), [("Kind", view.note.kind())])
        return self._branch.reinterpret(view.id, body, by=self._owner)

    def fork(self, token: str, name: str = "") -> MementoBranch:
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
        matched = [meta for meta in self._memento.list_branches() if meta.branch_id.startswith(prefix)]
        if len(matched) != 1:
            raise ValueError(f"branch not found or ambiguous: {prefix!r}")
        self._memento.switch(matched[0].branch_id)
        self._branch = self._memento.get_branch(matched[0].branch_id)
        return self._branch

    def branches(self) -> list[dict[str, str]]:
        return [
            {"id": meta.branch_id, "name": meta.name, "current": str(meta.branch_id == self._branch.meta.branch_id)}
            for meta in self._memento.list_branches()
        ]

    def describe_commit(self, token: str) -> str:
        view = self.find_commit(token)
        if view is None:
            raise ValueError(f"commit not found or ambiguous: {token!r}")
        lines = [f"commit seq={view.seq} id={view.id} kind={view.note.kind()}", f"summary: {view.summary()}"]
        for record in self._branch.commit_records(view.id):
            moment = record_to_moment(record)
            inputs = " | ".join(moment.percepts_texts())
            lines.append(f"moment={moment.id} input={inputs} logos={moment.logos}")
        return "\n".join(lines)

    def _mechanical_summary(self) -> str:
        """Build a bounded extractive index without inventing an interpretation."""
        lines = ["[extractive mechanical index]"]
        for record in self._branch.staging():
            moment = record_to_moment(record)
            inputs = " ".join(" ".join(moment.percepts_texts()).split())[:240]
            logos = " ".join(moment.logos.split())[:240]
            lines.append(f"- moment={moment.id} input={inputs or '[none]'} logos={logos or '[silent]'}")
        return "\n".join(lines)

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
