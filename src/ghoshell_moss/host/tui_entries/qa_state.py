"""QA State — non-blocking question notification center in the TUI.

C-q toggles in/out.  Two internal modes:

  list   — all pending questions, number keys or arrows to pick
  detail — full question view with kind-aware answer form

Interaction model (Claude Code-inspired):
  - 1..N       = select option (choose/confirm: immediate submit; select: toggle)
  - N+1        = reject / cancel (always an exit)
  - Enter      = confirm current selection
  - Space      = toggle (select kind only)
  - Escape     = detail → list (not reject; question stays pending)
  - Tab        = complete from suggestions (input/password kinds)
"""

from typing import Literal

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from typing_extensions import Self

from ghoshell_moss.core.concepts.qa import QA, Question, YES
from ghoshell_moss.host.tui import TUIState

from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.table import Table
from rich.console import Group


_KIND_ICONS: dict[str, str] = {
    "confirm":  "?",
    "input":    "...",
    "password": "***",
    "choose":   "o",
    "select":   "+",
    "apply":    "!",
}


def _short_qid(qid: str) -> str:
    return qid[:8]


class _SuggestionCompleter(Completer):
    """Tab-complete from question options for input/password kinds."""

    def __init__(self, words: list[str]) -> None:
        self._words = words

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        for w in self._words:
            if w.startswith(text):
                yield Completion(w, start_position=-len(text))


class QAState(TUIState):
    """TUI state for browsing and answering pending questions.

    Not in C-t rotation — accessed via C-q toggle managed by MossHostTUI.
    """

    def __init__(
        self,
        qa_registry: dict[str, QA],
        on_exit: callable = None,
        name: str = "qa",
    ) -> None:
        self._name = name
        self._qa_registry = qa_registry
        self._on_exit = on_exit
        self._mode: Literal["list", "detail"] = "list"
        self._selected_idx: int = 0
        self._detail_qa: QA | None = None
        self._selected_options: set[int] = set()

    def name(self) -> str:
        return self._name

    # -- lifecycle ----------------------------------------------------------

    def on_switch(self, alive: bool) -> None:
        if alive:
            self._mode = "list"
            self._selected_idx = 0
            self._detail_qa = None
            self._selected_options.clear()
            self._render_list()

    def refresh(self) -> None:
        """Called when QA registry changes externally. Re-renders current view."""
        if self._mode == "list":
            self._render_list()
        elif self._detail_qa is not None:
            self._render_detail()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def on_interrupt(self, event: KeyPressEvent) -> None:
        """Escape: detail → list.  List mode: no-op (use C-q)."""
        if self._mode == "detail":
            self._mode = "list"
            self._detail_qa = None
            self._selected_options.clear()
            self._render_list()

    # -- completer -----------------------------------------------------------

    def completer(self) -> Completer | None:
        if self._mode == "detail" and self._detail_qa is not None:
            q = self._detail_qa.question
            if q.kind in ("input", "password") and q.options:
                return _SuggestionCompleter(list(q.options.values()))
        return None

    # -- text input (input / password kinds) ---------------------------------

    def handle_input(self, console_input: str) -> None:
        if self._mode != "detail" or self._detail_qa is None:
            return
        qa = self._detail_qa
        kind = qa.question.kind

        if kind in ("input", "password"):
            if console_input == "":
                # empty input = reject (user just pressed Enter)
                self._reject_detail()
                return
            answer = qa.question.answer(console_input)
            try:
                qa.reply(answer)
            except ValueError:
                self.console.hint("Already replied — another watcher may have answered first")
            self._mode = "list"
            self._detail_qa = None
            self._render_list()

    # -- key bindings -------------------------------------------------------

    def key_bindings(self) -> KeyBindings | None:
        kb = KeyBindings()

        @kb.add("up")
        def _up(event: KeyPressEvent) -> None:
            self._move_selection(-1)

        @kb.add("down")
        def _down(event: KeyPressEvent) -> None:
            self._move_selection(1)

        @kb.add("enter")
        def _enter(event: KeyPressEvent) -> None:
            if self._mode == "list":
                items = self._pending_items()
                if not items and self._on_exit:
                    self._on_exit()
                    return
                self._open_detail()
            elif self._mode == "detail" and self._detail_qa is not None:
                kind = self._detail_qa.question.kind
                if kind in ("input", "password"):
                    text = event.current_buffer.text
                    event.current_buffer.reset()
                    self.handle_input(text)
                else:
                    self._confirm_detail()

        @kb.add("space")
        def _space(event: KeyPressEvent) -> None:
            if self._mode == "detail" and self._detail_qa is not None:
                if self._detail_qa.question.kind == "select":
                    self._toggle_current()
                    self._render_detail()

        # Number keys 1-9: select in list, select/toggle in detail
        for n in range(1, 10):

            @kb.add(str(n))
            def _number(event: KeyPressEvent, num: int = n) -> None:
                if self._mode == "list":
                    self._list_number_key(num)
                elif self._mode == "detail":
                    self._detail_number_key(num)

        return kb

    # -- navigation ----------------------------------------------------------

    def _move_selection(self, delta: int) -> None:
        if self._mode == "list":
            items = self._pending_items()
            if not items:
                return
            self._selected_idx = max(0, min(self._selected_idx + delta, len(items) - 1))
            self._render_list()
        elif self._mode == "detail" and self._detail_qa is not None:
            kind = self._detail_qa.question.kind
            if kind in ("confirm", "choose", "apply", "select"):
                opts = list(self._detail_qa.question.options.keys())
                if not opts:
                    return
                self._selected_idx = max(0, min(self._selected_idx + delta, len(opts) - 1))
                self._render_detail()

    def _pending_items(self) -> list[QA]:
        """Undone questions that this watcher hasn't replied to yet."""
        return [
            qa for qa in self._qa_registry.values()
            if not qa.done() and qa.replied() is None
        ]

    # -- list mode: number keys ----------------------------------------------

    def _list_number_key(self, num: int) -> None:
        items = self._pending_items()
        idx = num - 1
        if 0 <= idx < len(items):
            self._selected_idx = idx
            self._open_detail()

    # -- detail mode: number keys --------------------------------------------

    def _detail_number_key(self, num: int) -> None:
        qa = self._detail_qa
        if qa is None:
            return
        q = qa.question
        kind = q.kind
        opts = list(q.options.keys())
        n_opts = len(opts)

        if kind in ("confirm", "choose", "apply"):
            if num <= n_opts:
                # 1..N = select + immediate submit
                self._selected_idx = num - 1
                self._confirm_detail()
            elif num == n_opts + 1:
                # N+1 = reject
                self._reject_detail()
        elif kind == "select":
            if num <= n_opts:
                # 1..N = toggle
                self._selected_idx = num - 1
                self._toggle_current()
                self._render_detail()
            elif num == n_opts + 1:
                # N+1 = reject
                self._reject_detail()

    def _toggle_current(self) -> None:
        if self._selected_idx in self._selected_options:
            self._selected_options.discard(self._selected_idx)
        else:
            self._selected_options.add(self._selected_idx)

    # -- list mode -----------------------------------------------------------

    def _open_detail(self) -> None:
        items = self._pending_items()
        if not items or self._selected_idx >= len(items):
            return
        self._detail_qa = items[self._selected_idx]
        self._selected_idx = 0
        self._selected_options.clear()
        self._mode = "detail"
        self._render_detail()

    def _render_list(self) -> None:
        items = self._pending_items()

        if not items:
            self.console.rprint(
                Panel("No pending questions.", border_style="dim", title="QA"))
            self.console.hint("Enter or C-q back to previous view")
            return

        table = Table(box=None, expand=True, show_header=False, padding=(0, 0))
        table.add_column("#", style="dim", width=2)
        table.add_column("", width=4)
        table.add_column("Question")
        table.add_column("ID", style="dim", width=10)

        for i, qa in enumerate(items):
            q = qa.question
            icon = _KIND_ICONS.get(q.kind, "?")
            marker = ">" if i == self._selected_idx else " "
            qid = _short_qid(q.meta.id if q.meta else "?")
            content = q.content.replace("\n", " ")[:68]
            table.add_row(
                str(i + 1) if i < 9 else " ",
                f"{marker}[{icon}]",
                content,
                f"#{qid}",
            )

        header = Text(f" {len(items)} Questions ", style="bold yellow")
        self.console.rprint(
            Panel(table, title=header, title_align="left", border_style="yellow", padding=(0, 1)))
        self.console.hint(
            "1-9 pick  up/down navigate  Enter open  C-q back")

    # -- detail mode ---------------------------------------------------------

    def _render_detail(self) -> None:
        qa = self._detail_qa
        if qa is None:
            return
        q = qa.question

        # header
        qid = _short_qid(q.meta.id if q.meta else "?")
        issuer = q.meta.issuer if q.meta else ""
        header = Text()
        header.append(f"{q.kind.upper()} ", style="bold yellow")
        header.append(f"#{qid}", style="dim")
        if issuer:
            header.append(f"  from ", style="dim")
            header.append(issuer, style="cyan")

        # body: content + markdown (parallel)
        body_parts = [Text(q.content, style="bold")]
        if q.markdown:
            body_parts.append(Text(""))
            body_parts.append(Markdown(q.markdown, code_theme="ansi_dark"))
        body = Panel(Group(*body_parts), border_style="cyan", padding=(0, 1))

        # answer form
        form = self._render_answer_form(q)

        self.console.rprint(header)
        self.console.rprint(body)
        if form is not None:
            self.console.rprint(form)

    def _render_answer_form(self, q: Question) -> Panel | None:
        kind = q.kind
        opts = list(q.options.items())  # [(key, label), ...]
        n = len(opts)

        if kind == "confirm":
            return self._render_option_panel(
                opts,
                title="Confirm",
                hint_fn=lambda i: f"1-{n} choose  Enter confirm" if n else "",
                reject_key=n + 1 if n else None,
            )
        elif kind == "choose":
            return self._render_option_panel(
                opts,
                title="Choose one",
                hint_fn=lambda i: f"1-{n} pick  Enter confirm  {n+1}=cancel",
                reject_key=n + 1,
            )
        elif kind == "select":
            return self._render_multi_panel(q, opts)
        elif kind == "apply":
            apply_opts = [("approve", "Approve"), ("reject", "Reject")]
            return self._render_option_panel(
                apply_opts,
                title="Decision",
                hint_fn=lambda i: "1=approve  2=reject  Enter confirm",
                reject_key=None,  # reject IS option 2
            )
        elif kind in ("input", "password"):
            mask_hint = " (input masked)" if kind == "password" else ""
            sug_hint = ""
            if q.options:
                words = list(q.options.values())
                sug_hint = f"  Suggestions: {', '.join(words[:6])}"
            self.console.hint(
                f"Type your answer{mask_hint} — Enter submit  empty=reject  C-q back{sug_hint}")
            return None

        return None

    def _render_option_panel(
        self,
        opts: list[tuple[str, str]],
        *,
        title: str,
        hint_fn,
        reject_key: int | None,
    ) -> Panel:
        table = Table(box=None, expand=True, show_header=False, padding=(0, 0))
        table.add_column("#", style="dim", width=2)
        table.add_column("", width=2)
        table.add_column("Key", style="cyan", width=12)
        table.add_column("Label")

        for i, (key, label) in enumerate(opts):
            num = str(i + 1)
            marker = ">" if i == self._selected_idx else " "
            table.add_row(num, marker, key, label)

        if reject_key is not None:
            table.add_row(str(reject_key), " ", "cancel", Text("reject", style="dim"))

        hint = hint_fn(len(opts))
        return Panel(
            Group(table, Text(hint, style="dim")),
            title=title,
            title_align="left",
            border_style="green",
            padding=(0, 1),
        )

    def _render_multi_panel(self, q: Question, opts: list[tuple[str, str]]) -> Panel:
        table = Table(box=None, expand=True, show_header=False, padding=(0, 0))
        table.add_column("#", style="dim", width=2)
        table.add_column("", width=2)
        table.add_column("Key", style="cyan", width=12)
        table.add_column("Label")

        for i, (key, label) in enumerate(opts):
            num = str(i + 1)
            checked = "[x]" if i in self._selected_options else "[ ]"
            marker = ">" if i == self._selected_idx else " "
            table.add_row(num, marker, f"{checked} {key}", label)

        n_sel = len(self._selected_options)
        n = len(opts)
        hint = f"1-{n} toggle  Space toggle  Enter submit ({n_sel})  {n + 1}=cancel"

        return Panel(
            Group(
                table,
                Text(f"min {q.min_selection}  max {q.max_selection}  selected {n_sel}", style="dim"),
                Text(hint, style="dim"),
            ),
            title="Select options",
            title_align="left",
            border_style="green",
            padding=(0, 1),
        )

    # -- detail actions ------------------------------------------------------

    def _confirm_detail(self) -> None:
        """Enter pressed in detail mode — confirm current selection."""
        qa = self._detail_qa
        if qa is None:
            return
        q = qa.question
        kind = q.kind
        opts = list(q.options.keys())

        try:
            if kind == "confirm":
                if opts:
                    choice = opts[self._selected_idx]
                    answer = q.confirm(choice == YES)
                else:
                    answer = q.confirm(True)
            elif kind == "choose":
                if not opts:
                    return
                choice = opts[self._selected_idx]
                answer = q.choose(choice)
            elif kind == "select":
                choices = [opts[i] for i in sorted(self._selected_options) if i < len(opts)]
                answer = q.select(*choices)
            elif kind == "apply":
                if self._selected_idx == 0:
                    answer = q.approve()
                else:
                    answer = q.reject("rejected")
            else:
                return  # input/password handled via handle_input
            qa.reply(answer)
        except ValueError:
            self.console.hint("Already replied — another watcher may have answered first")
            self._mode = "list"
            self._detail_qa = None
            self._selected_options.clear()
            self._render_list()
            return

        self._mode = "list"
        self._detail_qa = None
        self._selected_options.clear()
        self._render_list()

    def _reject_detail(self) -> None:
        """Reject/cancel the current question."""
        qa = self._detail_qa
        if qa is None:
            return
        answer = qa.question.reject("canceled by user")
        try:
            qa.reply(answer)
        except ValueError:
            self.console.hint("Already resolved")
        self._mode = "list"
        self._detail_qa = None
        self._selected_options.clear()
        self._render_list()
