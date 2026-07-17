"""Ghost-side Memento adapter: persistent frames in, model history out."""

from pathlib import Path

from pydantic_ai import TextContent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart

from ghoshell_moss.core.blueprint.mindflow import Moment
from ghoshell_moss.core.memento import (
    CommitView,
    Memento,
    MementoBranch,
    new_filesystem_memento,
)
from ghoshell_moss.core.memento.porcelain import (
    record_to_moment,
    update_moment,
    window_messages,
)
from ghoshell_moss.message import Message

from ._adapter import messages_to_parts

__all__ = ["DataMemory"]


class DataMemory:
    """Single-owner persistent conversation memory for one Data Ghost."""

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
        }

    def close(self) -> None:
        if not self._closed:
            self._memento.close()
            self._closed = True
