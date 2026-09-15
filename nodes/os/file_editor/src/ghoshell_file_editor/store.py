"""ThreadStore — the single source of truth for file-editor threads.

Wraps the pure structures (:mod:`ghoshell_file_editor.structure`) with an
append-only JSONL log so a process crash does not lose the last version: every
mutation (open / append / confirm / reject / reply) appends one record, and
:meth:`ThreadStore.replay` rebuilds the store from the log.

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
    Author,
    Effect,
    Kind,
    Reply,
    Seq,
    Thread,
    Version,
    Anchor,
    Verdict,
    cascade_seqs,
    effect_of,
    is_mutating,
    result_content,
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
    """Holds threads + actions, appending each op to an optional JSONL log."""

    def __init__(self, log_path: str | Path | None = None) -> None:
        self._threads: dict[str, Thread] = {}
        self._actions: dict[tuple[str, int], Action] = {}
        self._next_n: dict[str, int] = {}
        self._log = _Log(Path(log_path)) if log_path is not None else None

    # -- log --

    def _append_log(self, record: dict) -> None:
        if self._log is not None:
            self._log.append(record)

    # -- reads --

    def get_thread(self, thread_id: str) -> Thread | None:
        return self._threads.get(thread_id)

    def get_action(self, thread_id: str, n: int) -> Action | None:
        return self._actions.get((thread_id, n))

    def version_content(self, thread_id: str, version_id: str) -> str:
        thread = self._threads[thread_id]
        for v in thread.versions:
            if v.id == version_id:
                return v.content
        raise KeyError(f"no version {version_id!r} on thread {thread_id!r}")

    # -- mutations --

    def open_thread(
        self,
        thread_id: str,
        label: str,
        path: str | None = None,
        motivation: str = "",
        base_content: str = "",
    ) -> Thread:
        if thread_id in self._threads:
            raise KeyError(f"thread {thread_id!r} already open")
        thread = Thread(id=thread_id, label=label, path=path, motivation=motivation)
        if base_content:
            v0 = Version(
                id=f"{thread_id}:v0",
                thread_id=thread_id,
                parent=None,
                action_seq=None,
                content=base_content,
                effect=Effect(before="", after=base_content, diff=""),
            )
            thread.versions.append(v0)
            thread.head = v0.id
        self._threads[thread_id] = thread
        self._next_n[thread_id] = 1
        self._append_log({
            "op": "open",
            "thread_id": thread_id,
            "label": label,
            "path": path,
            "motivation": motivation,
            "base_content": base_content,
        })
        return thread

    def append_action(
        self,
        thread_id: str,
        author: Author,
        kind: Kind,
        description: str,
        payload: str,
        from_version: str | None = None,
        n: int | None = None,
    ) -> Seq:
        thread = self._threads[thread_id]
        if n is None:
            n = self._next_n.get(thread_id, 1)
        seq = Seq(thread_id=thread_id, n=n)
        self._actions[(thread_id, n)] = Action(
            seq=seq,
            author=author,
            kind=kind,
            description=description,
            payload=payload,
            from_version=from_version,
        )
        thread.order.append(seq)
        self._next_n[thread_id] = n + 1
        self._append_log({
            "op": "action",
            "thread_id": thread_id,
            "n": n,
            "author": author,
            "kind": kind,
            "description": description,
            "payload": payload,
            "from_version": from_version,
        })
        return seq

    def confirm(self, thread_id: str, n: int) -> Version | None:
        thread = self._threads[thread_id]
        action = self._actions[(thread_id, n)]
        action.verdict = "confirmed"
        if not is_mutating(action.kind):
            self._append_log({"op": "confirm", "thread_id": thread_id, "n": n})
            return None
        base = thread.head_version.content if thread.head_version is not None else ""
        after = result_content(action, base, lambda vid: self.version_content(thread_id, vid))
        effect = effect_of(base, after)
        version = Version(
            id=f"{thread_id}:v{len(thread.versions)}",
            thread_id=thread_id,
            parent=thread.head,
            action_seq=action.seq,
            content=after,
            effect=effect,
        )
        thread.versions.append(version)
        thread.head = version.id
        action.effect = effect
        self._append_log({"op": "confirm", "thread_id": thread_id, "n": n})
        return version

    def reject(self, thread_id: str, n: int) -> list[Seq]:
        thread = self._threads[thread_id]
        cascaded = cascade_seqs(thread, Seq(thread_id, n))
        for seq in cascaded:
            self._actions[(seq.thread_id, seq.n)].verdict = "rejected"
        self._append_log({"op": "reject", "thread_id": thread_id, "n": n})
        return cascaded

    def reply(
        self,
        thread_id: str,
        n: int,
        author: Author,
        anchor: Anchor,
        diff: str | None = None,
        text: str = "",
        verdict: Verdict | None = None,
    ) -> Reply:
        action = self._actions[(thread_id, n)]
        reply = Reply(
            n=len(action.replies),
            author=author,
            anchor=anchor,
            diff=diff,
            text=text,
            verdict=verdict,
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
            "verdict": verdict,
        })
        return reply

    # -- replay --

    @classmethod
    def replay(cls, log_path: str | Path) -> "ThreadStore":
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
                    from_version=rec.get("from_version"), n=rec["n"],
                )
            elif op == "confirm":
                store.confirm(rec["thread_id"], rec["n"])
            elif op == "reject":
                store.reject(rec["thread_id"], rec["n"])
            elif op == "reply":
                store.reply(
                    rec["thread_id"], rec["n"], rec["author"], rec["anchor"],
                    diff=rec.get("diff"), text=rec.get("text", ""),
                    verdict=rec.get("verdict"),
                )
            else:
                raise ValueError(f"unknown log op: {op!r}")
        return store
