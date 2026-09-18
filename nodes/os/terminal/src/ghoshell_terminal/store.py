"""CardStore — the single source of truth both faces read and write.

The channel drives it from the model side, the web surface from the human side,
and neither talks to the other directly; every frame the browser sees is derived
from this store. Verdicts are the one place where the two sides must rendezvous:
the channel's background task parks on a future the store hands out, and the web
surface settles it. That future lives here rather than in either face because
"has this card been decided" is card state, not channel or surface state.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from .card import Card, CardState, CardType, Thread

__all__ = ["CardStore", "Mode"]


class Mode:
    """The three global postures. Plain constants — the wire format is the string."""

    APPROVAL = "approval"
    """Every command waits for a human verdict. The default."""

    AUTO = "auto"
    """A command matching an accepted rule runs without asking."""

    DISABLED = "disabled"
    """The channel's commands drop out of the model's interface entirely."""

    ALL = (APPROVAL, AUTO, DISABLED)


_TAIL_LINES = 400
"""How many output lines stay in memory per card. The complete record is the
output file; this is only what ``read()`` and a late-connecting client need."""


class CardStore:
    """Cards, threads, rules and the global mode.

    :param root: the only subtree commands may run in. A thread's cwd must
        resolve inside it.
    :param outputs_dir: where per-card output files are written. Created on
        demand.
    """

    def __init__(self, *, root: str | Path, outputs_dir: str | Path) -> None:
        self._root = Path(root).resolve()
        self._outputs_dir = Path(outputs_dir)
        self._threads: dict[str, Thread] = {}
        self._cards: dict[int, Card] = {}
        self._order: list[int] = []
        self._rules: list[Card] = []
        self._mode = Mode.APPROVAL
        self._waiters: dict[int, asyncio.Future] = {}
        self._verdicts: dict[int, str] = {}
        self._counter = 0

    # -- mode ---------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> str:
        if mode not in Mode.ALL:
            raise ValueError(f"unknown mode {mode!r} (one of {list(Mode.ALL)})")
        self._mode = mode
        return self._mode

    @property
    def root(self) -> Path:
        return self._root

    # -- threads ------------------------------------------------------------

    def open_thread(self, name: str, cwd: str, description: str = "") -> Thread:
        """Register a named working context. cwd must resolve inside the root."""
        if not name:
            raise ValueError("thread name is required")
        if name in self._threads:
            raise ValueError(f"thread {name!r} already exists")
        resolved = self._resolve_cwd(cwd or str(self._root))
        thread = Thread(name=name, cwd=resolved, description=description)
        self._threads[name] = thread
        return thread

    def get_thread(self, name: str) -> Thread | None:
        return self._threads.get(name)

    def threads(self) -> list[Thread]:
        return list(self._threads.values())

    def _resolve_cwd(self, cwd: str) -> str:
        path = Path(cwd)
        if not path.is_absolute():
            path = self._root / path
        resolved = path.resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise ValueError(
                f"cwd {cwd!r} escapes the root {self._root} — threads must stay inside it"
            )
        return str(resolved)

    # -- cards --------------------------------------------------------------

    def new_card(
        self,
        card_type: CardType,
        *,
        title: str = "",
        description: str = "",
        thread: str = "",
        cwd: str = "",
        level: str = "info",
    ) -> Card:
        self._counter += 1
        card = Card(
            id=self._counter,
            type=card_type,
            title=title,
            description=description,
            thread=thread,
            cwd=cwd,
            level=level,
        )
        self._cards[card.id] = card
        self._order.append(card.id)
        return card

    def get(self, card_id: int) -> Card | None:
        return self._cards.get(card_id)

    def cards(self) -> list[Card]:
        return [self._cards[i] for i in self._order if i in self._cards]

    def append_content(self, card_id: int, text: str) -> Card:
        card = self._require(card_id)
        card.content += text
        card.updated = time.time()
        return card

    def add_dialogue(self, card_id: int, author: str, text: str) -> Card:
        from .card import Dialogue

        card = self._require(card_id)
        card.dialogue.append(Dialogue(author=author, text=text))
        card.updated = time.time()
        return card

    def set_state(
        self, card_id: int, state: CardState, *, exit_code: int | None = None
    ) -> Card:
        card = self._require(card_id)
        card.state = state
        card.updated = time.time()
        if exit_code is not None:
            card.exit_code = exit_code
        if card.settled:
            card.ended = card.updated
        return card

    def append_output(self, card_id: int, lines: list[str]) -> Card:
        """Record output lines: bounded memory tail + complete file record."""
        card = self._require(card_id)
        if not lines:
            return card
        card.output_tail.extend(lines)
        if len(card.output_tail) > _TAIL_LINES:
            del card.output_tail[: len(card.output_tail) - _TAIL_LINES]
        card.output_chars += sum(len(line) for line in lines)
        card.updated = time.time()
        self._outputs_dir.mkdir(parents=True, exist_ok=True)
        path = self._outputs_dir / f"card_{card.id}.log"
        with path.open("a", encoding="utf-8") as f:
            f.writelines(lines)
        card.output_file = str(path)
        return card

    def output_text(self, card_id: int) -> str:
        return "".join(self._require(card_id).output_tail)

    # -- verdicts -----------------------------------------------------------

    def waiter(self, card_id: int) -> asyncio.Future:
        """The future the channel parks on while a card awaits its verdict.

        The verdict is recorded separately from the future, so a verdict that
        lands before the channel has even parked is not lost: the waiter is
        created already-resolved.
        """
        future = self._waiters.get(card_id)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            self._waiters[card_id] = future
            verdict = self._verdicts.get(card_id)
            if verdict is not None:
                future.set_result(verdict)
        return future

    def settle(self, card_id: int, verdict: str) -> bool:
        """Hand down a verdict on a pending card. False = already decided.

        Guarding here (not only in the UI) is what makes the actions debounce:
        once a card leaves ``awaiting`` every later accept/deny is a no-op.
        """
        card = self._cards.get(card_id)
        if card is None or card.state is not CardState.AWAITING:
            return False
        self._verdicts[card_id] = verdict
        future = self._waiters.get(card_id)
        if future is not None and not future.done():
            future.set_result(verdict)
        return True

    def _require(self, card_id: int) -> Card:
        card = self._cards.get(card_id)
        if card is None:
            raise KeyError(f"no card {card_id}")
        return card

    # -- rules --------------------------------------------------------------

    def activate_rule(self, card_id: int) -> Card:
        """An accepted rule card joins the live rule set (used in auto mode)."""
        card = self._require(card_id)
        if card.type is not CardType.RULE:
            raise ValueError(f"card {card_id} is not a rule")
        if not card.content.strip():
            raise ValueError(f"rule card {card_id} has an empty pattern")
        re.compile(card.content)
        if card.id not in [c.id for c in self._rules]:
            self._rules.append(card)
        return card

    def rules(self) -> list[Card]:
        return list(self._rules)

    def match_rule(self, command: str) -> Card | None:
        """The first active rule whose pattern matches, or None.

        Rules are untrusted-by-construction — the human accepted each one — so a
        broken pattern is skipped rather than raised.
        """
        for rule in self._rules:
            try:
                if re.search(rule.content, command):
                    return rule
            except re.error:
                continue
        return None

    # -- counting (for notices) --------------------------------------------

    def awaiting(self) -> list[Card]:
        return [c for c in self.cards() if c.state is CardState.AWAITING]

    def running(self) -> list[Card]:
        return [c for c in self.cards() if c.state is CardState.RUNNING]
