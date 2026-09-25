"""moss nodes answer-node — headless QA answering terminal.

A minimal, GUI-free way to answer questions broadcast on a QA namespace.
This is the CLI example the GUI will mirror: rich renders the question,
prompt_toolkit drives the interaction.

Interaction model (single persistent PromptSession under patch_stdout):
  input / password — free text (Tab to complete suggestions), empty = reject
  confirm / choose — pick an option key from the completer dropdown, optional
                     trailing note becomes Answer.content; "reject" skips.
  apply            — "approve" / "reject" (real answers); 0 / Ctrl+C cancels.
  select           — numbered Rich table, comma-separated numbers, 0 = reject.

Concurrency contract (the QA protocol is live while we interact):
  * a question may become `done()` while still being displayed — we re-check
    before replying and skip gracefully ("resolved elsewhere").
  * replying is first-wins and does NOT resolve immediately — after reply we
    report "submitted — awaiting requester verdict"; resolution arrives via
    on_answer / on_cancel and drives the pending count.

Runs in-process via Matrix.new(), matching the audio CLI examples
(no subprocess, no node directory — 开箱自带极简).
"""

from __future__ import annotations

import asyncio
import re

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, DynamicCompleter
from prompt_toolkit.patch_stdout import patch_stdout

from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.concepts.qa import QA, Question, Answer, YES
from ghoshell_moss.cli.utils import console, print_info

_KIND_ICONS: dict[str, str] = {
    "confirm": "?",
    "input": "...",
    "password": "***",
    "choose": "o",
    "select": "+",
    "apply": "!",
}

_REJECT_WORDS = {"r", "q", "quit", "cancel", "x", "skip"}


def _short_qid(q: Question) -> str:
    meta = q.meta
    return meta.id[:8] if meta and meta.id else "?"


def _notice(message: str) -> None:
    """Coordinate a transient notice with the active prompt via patched stdout."""
    console.print(Text(message, style="dim"))


def run_answer_node(namespace: str = "") -> None:
    """Sync CLI entry — create an in-process Matrix and run the QA loop."""
    matrix = Matrix.new("answer-node", category="cli")
    matrix.run(lambda m: _answer_loop(m, namespace))


async def _answer_loop(matrix: Matrix, namespace: str) -> None:
    qa_mgr = matrix.session.qa
    if qa_mgr is None:
        print_info(
            "QAManager not available (session.qa is None). "
            "Is the QA provider wired into the matrix?"
        )
        return

    async with qa_mgr:
        watcher = qa_mgr.watch(namespace)
        queue: asyncio.Queue[QA] = asyncio.Queue()
        pending: dict[str, QA] = {}
        session = PromptSession(complete_while_typing=True, complete_in_thread=True)

        def _invalidate() -> None:
            app = getattr(session, "app", None)
            if app is not None and app.is_running:
                app.invalidate()

        def _on_resolved(qid: str) -> None:
            pending.pop(qid, None)
            _invalidate()

        def _on_question(qa: QA) -> None:
            qid = qa.question.meta.id if qa.question.meta else ""
            pending[qid] = qa
            qa.on_answer(lambda _a: _on_resolved(qid))
            qa.on_cancel(lambda _q: _on_resolved(qid))
            queue.put_nowait(qa)

        watcher.on_question(_on_question)
        # late-join: pick up questions broadcast before we started watching
        for qa in watcher.questions(answered=False):
            _on_question(qa)

        print_info(
            f"[answer-node] watching QA namespace {namespace!r} — Ctrl+C to exit."
        )
        await _interactive_loop(queue, pending, session)


async def _interactive_loop(
    queue: asyncio.Queue[QA],
    pending: dict[str, QA],
    session: PromptSession,
) -> None:
    current: dict[str, QA | None] = {"qa": None}

    def _toolbar() -> str:
        qa = current["qa"]
        if qa is None:
            return f"{len(pending)} pending · waiting for questions…"
        q = qa.question
        return f"{len(pending)} pending · {q.kind.upper()} #{_short_qid(q)} · {_strategy_hint(q)}"

    def _completer() -> Completer | None:
        qa = current["qa"]
        if qa is None:
            return None
        return _completer_for(qa.question)

    dynamic_completer = DynamicCompleter(_completer)

    with patch_stdout(raw=True):
        while True:
            qa = await queue.get()
            if qa.done() or qa.replied() is not None:
                continue
            current["qa"] = qa
            console.print(_build_question_panel(qa.question))
            while True:
                try:
                    raw = await session.prompt_async(
                        "> ",
                        completer=dynamic_completer,
                        bottom_toolbar=_toolbar,
                    )
                except KeyboardInterrupt:
                    _reject_current(qa)
                    raise
                if qa.done() or qa.replied() is not None:
                    _notice("question resolved elsewhere — skipped")
                    break
                try:
                    answer = _parse_answer(qa.question, raw)
                except ValueError as e:
                    _notice(f"invalid input: {e}")
                    continue
                try:
                    qa.reply(answer)
                except ValueError:
                    _notice("already replied — another watcher answered first")
                else:
                    _notice("submitted — awaiting requester verdict")
                break


def _reject_current(qa: QA) -> None:
    try:
        qa.reply(qa.question.reject("canceled by user"))
    except ValueError:
        pass


# -- question display (Rich) -----------------------------------------------


def _build_question_panel(q: Question) -> Panel:
    icon = _KIND_ICONS.get(q.kind, "?")
    issuer = q.meta.issuer if q.meta else ""
    qid = _short_qid(q)

    header = Text()
    header.append(f"[{icon}] ", style="bold yellow")
    header.append(q.kind.upper(), style="bold")
    header.append(f"  #{qid}", style="dim")
    if issuer:
        header.append("  from ", style="dim")
        header.append(issuer, style="cyan")

    body = [Text(q.content, style="bold")]
    if q.markdown:
        body.append(Text(""))
        body.append(Markdown(q.markdown, code_theme="ansi_dark"))

    if q.kind == "apply":
        body.append(Text(""))
        body.append(_options_table([("approve", "Approve"), ("reject", "Reject")]))
    elif q.options and q.kind not in ("input", "password"):
        body.append(Text(""))
        body.append(_options_table(list(q.options.items())))
    elif q.options:
        body.append(Text(""))
        body.append(Text("suggestions: " + ", ".join(q.options.values()), style="dim"))

    body.append(Text(""))
    body.append(Text(_strategy_hint(q), style="dim"))

    return Panel(
        Group(*body),
        title=header,
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
    )


def _options_table(opts: list[tuple[str, str]]) -> Table:
    table = Table(box=None, expand=True, show_header=False, padding=(0, 1))
    table.add_column("#", style="dim", width=2)
    table.add_column("Key", style="cyan")
    table.add_column("Label")
    for i, (key, label) in enumerate(opts):
        table.add_row(str(i + 1), key, label)
    return table


# -- completer --------------------------------------------------------------


def _completer_for(q: Question) -> Completer | None:
    if q.kind in ("input", "password") and q.options:
        return _SuggestionCompleter(list(q.options.values()))
    if q.kind in ("confirm", "choose", "apply"):
        return _OptionsCompleter(_single_completions(q))
    return None


def _single_completions(q: Question) -> list[tuple[str, str]]:
    """(word, display_meta) for the completer dropdown of single-select kinds."""
    if q.kind == "confirm":
        return [
            ("yes", q.options.get("yes", "")),
            ("no", q.options.get("no", "")),
            ("reject", "skip this question"),
        ]
    if q.kind == "choose":
        return list(q.options.items()) + [("reject", "skip this question")]
    if q.kind == "apply":
        return [("approve", "Approve"), ("reject", "Reject")]
    return []


class _SuggestionCompleter(Completer):
    def __init__(self, words: list[str]) -> None:
        self._words = words

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        for w in self._words:
            if w.startswith(text):
                yield Completion(w, start_position=-len(text))


class _OptionsCompleter(Completer):
    def __init__(self, items: list[tuple[str, str]]) -> None:
        self._items = items

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        for word, meta in self._items:
            if word.startswith(text):
                yield Completion(word, start_position=-len(text), display_meta=meta)


# -- strategy hint + answer parsing (pure, shared with GUI) -----------------


def _strategy_hint(q: Question) -> str:
    kind = q.kind
    if kind == "input":
        return "type answer (Tab to complete) · empty=reject"
    if kind == "password":
        return "type answer (masked) · empty=reject"
    if kind == "confirm":
        return "Tab to pick · yes / no / reject"
    if kind == "choose":
        return "Tab to pick · reject to skip"
    if kind == "apply":
        return "approve / reject · 0 to cancel"
    if kind == "select":
        n = len(q.options)
        return f"1..{n}=select, comma for multi (optional note) · 0=reject"
    return ""


def _parse_answer(q: Question, raw: str) -> Answer:
    text = raw.strip()
    kind = q.kind
    if kind in ("input", "password"):
        return q.answer(text) if text else q.reject("canceled by user")
    if kind == "select":
        if _is_reject(text, len(q.options)):
            return q.reject("canceled by user")
        return _parse_select(q, text)
    if text in _REJECT_WORDS or text == "0":
        return q.reject("canceled by user")
    return _parse_single(q, text)


def _is_reject(text: str, n: int) -> bool:
    """True if the input means 'skip this question' (reject)."""
    if text in _REJECT_WORDS or text == "0":
        return True
    try:
        return int(text) - 1 == n   # (N+1) = reject, ghost-TUI style
    except ValueError:
        return False


def _parse_single(q: Question, text: str) -> Answer:
    """confirm / choose / apply — an option key, optional trailing note → content."""
    m = re.match(r"^(\S+)(?:\s+(.*))?$", text)
    if not m:
        raise ValueError("expect an option (Tab to complete)")
    key = m.group(1)
    note = (m.group(2) or "").strip()

    keys = [k for k, _ in _single_completions(q)]
    matches = [k for k in keys if k == key or k.startswith(key)]
    if len(matches) != 1:
        raise ValueError(f"unknown option: {key}")
    choice = matches[0]

    if choice == "reject" and q.kind in ("confirm", "choose"):
        return q.reject(note or "canceled by user")
    return _build_answer(q, [choice], note)


def _parse_select(q: Question, text: str) -> Answer:
    """select — comma-separated numbers, optional trailing note → content."""
    opts = list(q.options.keys())
    n = len(opts)
    if n == 0:
        return q.reject("no options")

    num_text: list[str] = []
    note_parts: list[str] = []
    for token in text.split():
        if re.fullmatch(r"[\d,]+", token):
            num_text.append(token)
        else:
            note_parts.append(token)
    note = " ".join(note_parts)

    nums: list[int] = []
    for token in num_text:
        for seg in token.split(","):
            seg = seg.strip()
            if seg != "":
                nums.append(int(seg))
    if not nums:
        raise ValueError("expect at least one number (comma for multi)")

    seen: list[str] = []
    for i in nums:
        idx = i - 1
        if not (0 <= idx < n):
            raise ValueError(f"expect 1..{n}")
        choice = opts[idx]
        if choice not in seen:
            seen.append(choice)
    if not seen:
        raise ValueError("select at least one option")
    return _build_answer(q, seen, note)


def _build_answer(
    q: Question,
    keys: list[str],
    note: str = "",
    *,
    rejected: bool = False,
) -> Answer:
    """Build an Answer from selected option keys + an appended note (content)."""
    if rejected:
        return q.reject(note or "canceled by user")
    kind = q.kind
    if kind in ("input", "password"):
        return q.answer(note)
    if kind == "confirm":
        return q.confirm(keys[0] == YES if keys else True, note)
    if kind == "choose":
        if not keys:
            return q.reject(note or "canceled by user")
        return q.choose(keys[0], note)
    if kind == "apply":
        if keys and keys[0] == "approve":
            return q.approve(note)
        return q.reject(note or "rejected")
    if kind == "select":
        return q.select(*keys, content=note)
    raise ValueError(f"unsupported kind: {kind}")
