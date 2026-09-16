"""ThreadStore — the single source of truth for file-editor threads.

Wraps the pure structures (:mod:`ghoshell_file_editor.structure`) with an
append-only JSONL log so a process crash does not lose the last version: every
durable fact (open / action / confirm / reject / reply) appends one record, and
:meth:`ThreadStore.replay` rebuilds the store from the log. A streaming action
is open in memory (``open_action`` + ``append_delta``) but only becomes durable
when ``tail_action`` writes its record.

The log records facts only — payloads and verdicts. Effects, the head, and the
version list are all recomputed from them, so replay is deterministic and the
log stays as small as the dialogue.

No websocket, no channel, no MOSS deps — this layer is exercised directly by
tests. Durability is the only job here; the real side effect of ``export``
(writing to disk) is the channel's concern, not the store's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .structure import (
    Action,
    Anchor,
    Author,
    Kind,
    Reply,
    Seq,
    Thread,
    Verdict,
    cascade_seqs,
    compute_effect,
)

__all__ = ["ThreadStore"]


class _Log:
    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, record: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def records(self) -> list[dict]:
        if not self._path.exists():
            return []
        out: list[dict] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out


@dataclass
class ThreadStore:
    """Holds threads + their action lines, appending each op to an optional log."""

    def __init__(self, log_path: str | Path | None = None) -> None:
        self._threads: dict[str, Thread] = {}
        self._log = _Log(Path(log_path)) if log_path is not None else None

    # -- log --

    def _append_log(self, record: dict) -> None:
        if self._log is not None:
            self._log.append(record)

    # -- reads --

    def get_thread(self, thread_id: str) -> Thread | None:
        return self._threads.get(thread_id)

    def threads(self) -> list[Thread]:
        """Every thread, in the order they were opened."""
        return list(self._threads.values())

    def get_action(self, thread_id: str, n: int) -> Action | None:
        thread = self._threads.get(thread_id)
        if thread is None:
            return None
        return thread.actions.get(n)

    def content(self, thread_id: str) -> str:
        """The thread's current text — a derivation, not a stored field."""
        return self._threads[thread_id].content

    # -- mutations --

    def open_thread(
        self,
        thread_id: str,
        label: str,
        path: str | None = None,
        motivation: str = "",
        base_content: str = "",
    ) -> Thread:
        """Start a dialogue line. ``base_content`` is the loaded text, v0.

        A path carries at most one open line — two heads over one file would
        each believe they own the next state.
        """
        if thread_id in self._threads:
            raise KeyError(f"thread {thread_id!r} already open")
        if path is not None:
            for other in self._threads.values():
                if other.state == "open" and other.path == path:
                    raise ValueError(
                        f"thread {other.id!r} is already open on {path!r}; "
                        f"one dialogue line per path"
                    )
        thread = Thread(
            id=thread_id,
            label=label,
            path=path,
            motivation=motivation,
            base_content=base_content,
        )
        self._threads[thread_id] = thread
        self._append_log({
            "op": "open",
            "thread_id": thread_id,
            "label": label,
            "path": path,
            "motivation": motivation,
            "base_content": base_content,
        })
        return thread

    def _new_action(
        self,
        thread_id: str,
        author: Author,
        kind: Kind,
        description: str,
    ) -> Action:
        thread = self._threads[thread_id]
        n = max(thread.actions) + 1 if thread.actions else 1
        action = Action(
            seq=Seq(thread_id=thread_id, n=n),
            author=author,
            kind=kind,
            description=description,
            payload="",
        )
        thread.actions[n] = action
        return action

    def append_action(
        self,
        thread_id: str,
        author: Author,
        kind: Kind,
        description: str,
        payload: str,
    ) -> Seq:
        """Append one completed action, its effect computed at append time.

        Atomic for kinds whose payload is already known (``reference`` /
        ``rewind`` / ``export``). Streaming mutations go through
        :meth:`open_action` / :meth:`append_delta` / :meth:`tail_action`.
        """
        action = self._new_action(thread_id, author, kind, description)
        action.payload = payload
        action.effect = compute_effect(self._threads[thread_id], kind, payload)
        self._append_log({
            "op": "action",
            "thread_id": thread_id,
            "n": action.seq.n,
            "author": author,
            "kind": kind,
            "description": description,
            "payload": payload,
        })
        return action.seq

    def open_action(
        self,
        thread_id: str,
        author: Author,
        kind: Kind,
        description: str,
    ) -> Seq:
        """Start a streaming action — a mutation whose payload the model is
        still producing. It sits in the line with state ``streaming``, an empty
        payload and no effect; it is not durable until :meth:`tail_action`."""
        action = self._new_action(thread_id, author, kind, description)
        action.state = "streaming"
        return action.seq

    def append_delta(self, thread_id: str, n: int, text: str) -> None:
        """Accumulate one streaming chunk onto an open action."""
        action = self._threads[thread_id].actions[n]
        if action.state != "streaming":
            raise ValueError(f"action {n} on thread {thread_id!r} is not streaming")
        action.payload += text

    def tail_action(self, thread_id: str, n: int) -> Action:
        """Finalize a streaming action: compute its effect, make it durable."""
        thread = self._threads[thread_id]
        action = thread.actions[n]
        if action.state != "streaming":
            raise ValueError(f"action {n} on thread {thread_id!r} is not streaming")
        action.effect = compute_effect(thread, action.kind, action.payload)
        action.state = "tailed"
        self._append_log({
            "op": "action",
            "thread_id": thread_id,
            "n": n,
            "author": action.author,
            "kind": action.kind,
            "description": action.description,
            "payload": action.payload,
        })
        return action

    def confirm(self, thread_id: str, n: int, by: Author) -> list[Action]:
        """Confirm up to ``n``, returning the actions whose verdict flipped.

        Confirmation is a boundary, not a per-action switch: every pending
        action at or before ``n`` is confirmed together. Nothing is computed —
        the effects already exist, and the head is derived from the verdicts.
        """
        thread = self._threads[thread_id]
        flipped: list[Action] = []
        for action in thread.actions.values():
            if action.seq.n > n:
                break
            if action.verdict != "pending":
                continue
            action.verdict = "confirmed"
            action.verdict_by = by
            flipped.append(action)
        self._append_log({"op": "confirm", "thread_id": thread_id, "n": n, "by": by})
        return flipped

    def reject(self, thread_id: str, n: int, by: Author) -> list[Seq]:
        """Reject ``n`` and every later pending action on the thread.

        Confirmed actions are frozen — undoing one is an error, not a silent
        cascade. Undo is expressed by appending a ``rewind`` action, which keeps
        the line append-only.
        """
        thread = self._threads[thread_id]
        target = thread.actions.get(n)
        if target is None:
            raise KeyError(f"no action {n} on thread {thread_id!r}")
        if target.verdict == "confirmed":
            raise ValueError(
                f"action {n} on thread {thread_id!r} is already confirmed; "
                f"append a rewind action to undo it"
            )
        cascaded = cascade_seqs(thread, n)
        for seq in cascaded:
            action = thread.actions[seq.n]
            if action.verdict == "pending":
                action.verdict = "rejected"
                action.verdict_by = by
        self._append_log({"op": "reject", "thread_id": thread_id, "n": n, "by": by})
        return cascaded

    def reply(
        self,
        thread_id: str,
        n: int,
        author: Author,
        anchor: Anchor,
        diff: str | None = None,
        text: str = "",
    ) -> Reply:
        """Attach a dialogue entry under an action. Decides nothing."""
        action = self._threads[thread_id].actions[n]
        reply = Reply(
            n=len(action.replies),
            author=author,
            anchor=anchor,
            diff=diff,
            text=text,
        )
        action.replies.append(reply)
        self._append_log({
            "op": "reply",
            "thread_id": thread_id,
            "n": n,
            "reply_n": reply.n,
            "author": author,
            "anchor": anchor,
            "diff": diff,
            "text": text,
        })
        return reply

    # -- replay --

    @classmethod
    def replay(cls, log_path: str | Path) -> "ThreadStore":
        """Rebuild a store from its log, then keep appending to that log.

        Effects are recomputed, not stored; the replayed store is live — the
        log stays attached so new facts land on the same file.
        """
        store = cls()
        for rec in _Log(Path(log_path)).records():
            op = rec["op"]
            if op == "open":
                store.open_thread(
                    rec["thread_id"], rec["label"],
                    path=rec.get("path"), motivation=rec.get("motivation", ""),
                    base_content=rec.get("base_content", ""),
                )
            elif op == "action":
                store.append_action(
                    rec["thread_id"], rec["author"], rec["kind"],
                    rec["description"], rec["payload"],
                )
            elif op == "confirm":
                store.confirm(rec["thread_id"], rec["n"], rec["by"])
            elif op == "reject":
                store.reject(rec["thread_id"], rec["n"], rec["by"])
            elif op == "reply":
                store.reply(
                    rec["thread_id"], rec["n"], rec["author"], rec["anchor"],
                    diff=rec.get("diff"), text=rec.get("text", ""),
                )
            else:
                raise ValueError(f"unknown log op: {op!r}")
        store._log = _Log(Path(log_path))
        return store
