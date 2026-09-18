"""DocStore — the single source of truth for editable threads.

The channel drives it from the model side, the web surface from the human side,
and neither talks to the other directly.

Three places a thread's text can live, each with a different job:

- **memory** (this store) — the line of actions, their effects, the dialogue.
  This is what the human surface renders and what ``rewind`` targets. Lost on a
  crash, by design: nothing here is a durable record.
- **the draft file** — a working copy under the drafts directory, rewritten
  after every action that moves the text. This is the crash net: if the node
  dies, the text survives even though the line does not.
- **the real file** — touched only by ``export``, and only once a human accepts.
  The one real side effect.

There is deliberately no append-only log: the durable artifact is the draft, not
the history, so the filesystem stays bounded (one small file per live thread,
unlinked when the thread ends).

A thread ends one of two ways — ``mark_exported`` after the text lands on disk
(the final chapter) and ``close`` when it is abandoned. Ended threads keep their
line for the human's traceability; reclamation is a separate, silent step that
drops the oldest ended thread once they outgrow a bounded window.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import tempfile
import time
from pathlib import Path

from .structure import (
    Action,
    Author,
    Dialogue,
    Kind,
    Thread,
    content_at,
    effect_of,
    replace_once,
)

__all__ = ["DocStore"]

MAX_LIVE_THREADS = 16
"""How many un-exported threads may exist at once. Reaching the cap is a normal
state, not an error: the model is told to export or close."""

MAX_ENDED_THREADS = 8
"""How many ended (exported/closed) threads stay in memory for reference. Their
durable record is the file on disk, not the line."""

DRAFT_TTL_SECONDS = 7 * 24 * 3600
"""How long an unclaimed draft survives a crash before the sweep removes it."""

def _safe_name(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in text)


class DocStore:
    """Threads, their action lines, their drafts, and export verdicts.

    :param drafts_dir: where working copies are mirrored. Created on demand.
    :param max_live_threads: ceiling on live threads.
    :param draft_ttl: age past which an unclaimed draft is swept.
    """

    def __init__(
        self,
        *,
        drafts_dir: str | Path,
        root: str | Path,
        max_live_threads: int = MAX_LIVE_THREADS,
        draft_ttl: float = DRAFT_TTL_SECONDS,
    ) -> None:
        self._drafts_dir = Path(drafts_dir)
        self._root = Path(root).resolve()
        self._tempdir = Path(tempfile.gettempdir()).resolve()
        self._max_live = max_live_threads
        self._draft_ttl = draft_ttl
        self._threads: dict[str, Thread] = {}
        self._waiters: dict[str, asyncio.Future] = {}
        self._verdicts: dict[str, str] = {}
        self._recoverable: list[str] = []

    @property
    def drafts_dir(self) -> Path:
        return self._drafts_dir

    @property
    def root(self) -> Path:
        return self._root

    def resolve_target(self, path: str) -> str:
        """Resolve a file path and require it to live inside an allowed root.

        Two roots are trusted — the project home and the system temp dir — the
        former because it is the workspace, the latter because a throwaway file
        has a low blast radius. Anything else is outside the default
        authorization and is refused, not asked about.
        """
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self._root / p
        r = p.resolve()
        if self._inside(r, self._root) or self._inside(r, self._tempdir):
            return str(r)
        raise ValueError(
            f"{path!r} escapes the allowed roots "
            f"({self._root}, {self._tempdir})"
        )

    @staticmethod
    def _inside(p: Path, root: Path) -> bool:
        return p == root or root in p.parents

    # -- threads ------------------------------------------------------------

    def get(self, thread_id: str) -> Thread | None:
        return self._threads.get(thread_id)

    def threads(self) -> list[Thread]:
        """Every thread, in the order it was opened."""
        return list(self._threads.values())

    def live(self) -> list[Thread]:
        return [t for t in self._threads.values() if t.state == "live"]

    def open(
        self,
        thread_id: str,
        label: str = "",
        *,
        path: str | None = None,
        draft: str | None = None,
    ) -> Thread:
        """Start an editable object.

        ``draft`` adopts a working copy left behind by a crash (its text is the
        baseline); ``path`` loads a readable text file as the baseline; neither
        starts blank — an editable object does not have to come from a document.
        """
        if not thread_id:
            raise ValueError("thread id is required")
        if thread_id in self._threads:
            raise ValueError(f"thread {thread_id!r} already exists")
        if len(self.live()) >= self._max_live:
            open_ones = ", ".join(t.id for t in self.live())
            raise ValueError(
                f"{self._max_live} live threads already open ({open_ones}) — "
                f"export or close one first"
            )

        base = ""
        draft_name = ""
        if draft:
            source = self._drafts_dir / draft
            if not source.is_file():
                raise ValueError(f"no draft named {draft!r}")
            base = self._read_text(source, draft)
            draft_name = draft
        elif path:
            resolved = self.resolve_target(path)
            source = Path(resolved)
            if not source.is_file():
                raise ValueError(f"no readable file at {path!r}")
            base = self._read_text(source, path)
            for other in self.live():
                if other.path == resolved:
                    raise ValueError(
                        f"thread {other.id!r} is already editing {path!r}"
                    )
            path = resolved

        thread = Thread(
            id=thread_id,
            label=label or thread_id,
            path=path,
            base=base,
            draft=draft_name or self._new_draft_name(thread_id),
        )
        self._threads[thread_id] = thread
        self._mirror(thread)
        if draft_name in self._recoverable:
            self._recoverable.remove(draft_name)
        return thread

    def close(self, thread_id: str) -> Thread:
        """Abandon a thread: drop the working copy, touch no file.

        The line stays in memory for the human's traceability — it is reclaimed
        only when ended threads outgrow the reclamation window (see
        :meth:`_trim_ended`), never in a way the human has to notice.
        """
        thread = self._require(thread_id)
        self._unlink_draft(thread)
        thread.state = "closed"
        self._forget_verdicts(thread_id)
        self._trim_ended()
        return thread

    def mark_exported(self, thread_id: str, path: str) -> Thread:
        """The final chapter: the text is on disk now.

        The line stays intact — the human can still trace every action that led
        to the export. Reclamation is a separate, silent step: once ended
        threads exceed the window, the oldest is dropped whole, not emptied in
        place.
        """
        thread = self._require(thread_id)
        thread.exported_to = path
        thread.state = "exported"
        self._unlink_draft(thread)
        self._forget_verdicts(thread_id)
        self._trim_ended()
        return thread

    def _trim_ended(self) -> None:
        """Silently drop the oldest ended threads past the reclamation window.

        This is the analogue of the subprocesses' reclamation zone: a bounded
        buffer of recently-ended threads stays fully traceable, and beyond it the
        oldest is dropped whole. It only fires on capacity — a human watching a
        normal session never sees their history vanish.
        """
        ended = [t for t in self._threads.values() if t.state != "live"]
        while len(ended) > MAX_ENDED_THREADS:
            oldest = ended.pop(0)
            self._threads.pop(oldest.id, None)

    def set_thread_auto(self, thread_id: str, auto: bool) -> Thread:
        """Flip a thread's trust. Auto needs a final target, so a pathless
        thread cannot be trusted to auto-export."""
        thread = self._require_live(thread_id)
        if auto and not thread.path:
            raise ValueError(
                f"thread {thread_id!r} has no path — there is no target to "
                f"auto-export to"
            )
        thread.auto = auto
        return thread

    def _forget_verdicts(self, thread_id: str) -> None:
        for key in [k for k in self._verdicts if k.startswith(f"{thread_id}:")]:
            self._verdicts.pop(key, None)
            self._waiters.pop(key, None)

    def _require(self, thread_id: str) -> Thread:
        thread = self._threads.get(thread_id)
        if thread is None:
            raise KeyError(f"no thread {thread_id!r}")
        return thread

    def _require_live(self, thread_id: str) -> Thread:
        thread = self._require(thread_id)
        if thread.state != "live":
            raise ValueError(
                f"thread {thread_id!r} is {thread.state} — it takes no more "
                f"actions; open a new one"
            )
        return thread

    def _read_text(self, path: Path, label: str) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{label!r} is not readable text")

    # -- actions ------------------------------------------------------------

    def _add(
        self,
        thread: Thread,
        kind: Kind,
        label: str,
        author: Author,
        *,
        text: str = "",
        payload: str = "",
        state: str = "applied",
    ) -> Action:
        n = thread.actions[-1].n + 1 if thread.actions else 1
        action = Action(
            n=n,
            kind=kind,
            author=author,
            label=label,
            state=state,
            text=text,
            payload=payload,
        )
        thread.actions.append(action)
        return action

    def action(self, thread_id: str, n: int) -> Action:
        thread = self._require(thread_id)
        action = thread.get(n)
        if action is None:
            raise KeyError(f"no action {n} on thread {thread_id!r}")
        return action

    def record(
        self,
        thread_id: str,
        kind: Kind,
        label: str,
        *,
        text: str = "",
        payload: str = "",
        author: Author = "g",
        state: str = "applied",
    ) -> Action:
        """Append an action that carries no content change (open / read / export).

        These never move the text, so they never get an effect — they exist on
        the line for the human's benefit: a read that leaves a card behind is how
        the human sees exactly what the model saw.
        """
        thread = self._require_live(thread_id)
        return self._add(
            thread, kind, label, author, text=text, payload=payload, state=state
        )

    # -- streaming mutations (write / append) -------------------------------

    def begin(
        self, thread_id: str, kind: Kind, label: str = "", author: Author = "g"
    ) -> Action:
        if kind not in ("write", "append"):
            raise ValueError(f"{kind!r} does not stream")
        thread = self._require_live(thread_id)
        return self._add(
            thread, kind, label, author, state="streaming"
        )

    def feed(self, thread_id: str, n: int, chunk: str) -> Action:
        action = self.action(thread_id, n)
        if action.state != "streaming":
            raise ValueError(f"action {n} is not streaming")
        action.text += chunk
        return action

    def tail(self, thread_id: str, n: int) -> Action:
        """Land a streamed action in memory and mirror it to the draft."""
        thread = self._require_live(thread_id)
        action = self.action(thread_id, n)
        if action.state != "streaming":
            raise ValueError(f"action {n} is not streaming")
        before = thread.content
        after = action.text if action.kind == "write" else before + action.text
        action.effect = effect_of(before, after)
        action.state = "applied"
        self._mirror(thread)
        return action

    def cancel(self, thread_id: str, n: int) -> Action:
        action = self.action(thread_id, n)
        if action.state == "streaming":
            action.state = "cancelled"
        return action

    # -- atomic mutations ---------------------------------------------------

    def replace(
        self,
        thread_id: str,
        ops: list[tuple[str, str]],
        label: str = "",
        author: Author = "g",
    ) -> list[Action]:
        """Apply str_replace ops in order — one card per op.

        Each op must match exactly once in the text as it stands when the op runs,
        so a batch is applied sequentially and a later op sees an earlier op's
        result.
        """
        thread = self._require_live(thread_id)
        out: list[Action] = []
        for i, (old, new) in enumerate(ops):
            before = thread.content
            after = replace_once(before, old, new)
            name = label if len(ops) == 1 else f"{label} [{i + 1}/{len(ops)}]"
            action = self._add(
                thread,
                "str_replace",
                name,
                author,
                text=new,
                payload=json.dumps({"old_str": old, "new_str": new}, ensure_ascii=False),
            )
            action.effect = effect_of(before, after)
            out.append(action)
        self._mirror(thread)
        return out

    def rewind(self, thread_id: str, target: int, author: Author = "g") -> Action:
        """Append an ordinary action that puts the text back where ``target`` left it.

        ``0`` means the baseline. The target must be an action that actually moved
        the text — there is nothing to go back to otherwise.
        """
        thread = self._require_live(thread_id)
        after = self._target_content(thread, target)
        before = thread.content
        action = self._add(
            thread,
            "rewind",
            f"rewind to v{target}",
            author,
            payload=str(target),
        )
        action.effect = effect_of(before, after)
        self._mirror(thread)
        return action

    def _target_content(self, thread: Thread, target: int) -> str:
        if target == 0:
            return thread.base
        found = thread.get(target)
        if found is None:
            raise ValueError(f"no action {target} on thread {thread.id!r}")
        if found.effect is None:
            raise ValueError(
                f"action {target} on thread {thread.id!r} moved no text — "
                f"rewind needs a version, not a read"
            )
        return content_at(thread, target)

    # -- export verdicts ----------------------------------------------------

    def pending_exports(self) -> list[tuple[str, Action]]:
        return [
            (t.id, a)
            for t in self._threads.values()
            for a in t.actions
            if a.state == "awaiting"
        ]

    def waiter(self, thread_id: str, n: int) -> asyncio.Future:
        """The future the channel parks on while an export awaits its verdict.

        The verdict is recorded separately from the future so a verdict landing
        before the channel parks is not lost — the waiter is created resolved.
        """
        key = f"{thread_id}:{n}"
        future = self._waiters.get(key)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            self._waiters[key] = future
            verdict = self._verdicts.get(key)
            if verdict is not None:
                future.set_result(verdict)
        return future

    def settle(self, thread_id: str, n: int, verdict: str) -> bool:
        """Hand down a verdict on an awaiting export. False = already decided."""
        key = f"{thread_id}:{n}"
        action = self._threads.get(thread_id)
        found = action.get(n) if action is not None else None
        if found is None or found.state != "awaiting":
            return False
        self._verdicts[key] = verdict
        future = self._waiters.get(key)
        if future is not None and not future.done():
            future.set_result(verdict)
        return True

    def finish_export(self, thread_id: str, n: int, state: str) -> Action:
        thread = self._require(thread_id)
        action = self.action(thread_id, n)
        action.state = state
        return action

    # -- dialogue -----------------------------------------------------------

    def say(self, thread_id: str, n: int, author: Author, text: str) -> Action:
        action = self.action(thread_id, n)
        action.dialogue.append(Dialogue(author=author, text=text))
        return action

    # -- drafts -------------------------------------------------------------

    def _new_draft_name(self, thread_id: str) -> str:
        return f"{_safe_name(thread_id)}-{secrets.token_hex(4)}.txt"

    def draft_path(self, thread: Thread) -> Path:
        return self._drafts_dir / thread.draft

    def _mirror(self, thread: Thread) -> None:
        """Rewrite the working copy. Small files, sync write, crash net."""
        if not thread.draft:
            return
        path = self.draft_path(thread)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(thread.content, encoding="utf-8")

    def _unlink_draft(self, thread: Thread) -> None:
        if not thread.draft:
            return
        try:
            self.draft_path(thread).unlink(missing_ok=True)
        except OSError:
            pass

    def orphans(self) -> list[str]:
        """Draft files no live thread owns — work a crash left behind."""
        if not self._drafts_dir.is_dir():
            return []
        owned = {t.draft for t in self._threads.values() if t.draft}
        return sorted(
            p.name
            for p in self._drafts_dir.iterdir()
            if p.is_file() and p.name != ".gitignore" and p.name not in owned
        )

    def recoverable(self) -> list[str]:
        """The drafts the last :meth:`sweep` kept — work a crash left behind.

        Cached rather than rescanned: the notice renders on every meta refresh,
        and a directory listing has no business on that path.
        """
        return list(self._recoverable)

    def sweep(self) -> list[str]:
        """Remove expired orphans; keep the ones still young enough to adopt.

        Called once at startup. A draft nobody claims within the TTL is garbage;
        inside it, it is someone's unsaved work.
        """
        now = time.time()
        kept: list[str] = []
        for name in self.orphans():
            path = self._drafts_dir / name
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            if age > self._draft_ttl:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            else:
                kept.append(name)
        self._recoverable = kept
        return kept
