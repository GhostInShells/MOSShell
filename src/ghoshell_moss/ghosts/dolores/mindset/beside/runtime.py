"""BesideRuntime — issue a thinking group now, start it at the next step edge, deliver it later.

The timing model: ``now`` = think yourself; ``beside`` = issue now, start at the next step, return
as a signal. A ticket is the receipt's other half: the ego gets ``<beside-think seq="k"/>`` at
once, and the group starts when the current step ends — the one edge where this step's tool
results are in the log and the next step has not begun.

Two rules make it safe on the dsh seam:

- the step-end listener never awaits a model call (the session event dispatcher is serial), it
  only spawns tasks;
- every failure is content: a direction that fails becomes ``<error i="n">`` in the same delivery,
  so the ego never waits on a result that will not come.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from datetime import datetime
from xml.sax.saxutils import escape

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.mindflow import Signal
from ghoshell_moss.core.mindflow.mind_delivery_nucleus import new_mind_delivery_signal
from ghoshell_moss.deepseek_harness.v0_2_0 import ContentBlock, DshSession, DshSessionAnchor
from ghoshell_moss.message import Message

from ghoshell_moss.ghosts.dolores.mindset.ego_lifecycle import Discard, EgoLifecycle, StepEndEvent
from ghoshell_moss.ghosts.dolores.mindset.emanation import emanation_position
from ghoshell_moss.ghosts.dolores.mindset.prompter import DefaultPrompterKeys, PromptManager
from ghoshell_moss.ghosts.dolores.mindset.seeds import BESIDE_SESSION_LINE, BESIDE_THINK_SEED
from .config import EgoBesideMindConfig

__all__ = ["BesideRuntime", "BesideTicket", "BesideNoAnswer"]


class BesideNoAnswer(RuntimeError):
    """A round that came back without prose.

    A tool call and an empty reply are the same fact to a caller: nothing was said. Left alone
    they would ride the delivery as an empty ``<direction/>``, indistinguishable from "it thought
    and had nothing to add" — so both are turned into a result the ego can read.
    """


@dataclasses.dataclass(frozen=True, slots=True)
class BesideTicket:
    """One issued thinking group, waiting for its step edge."""

    seq: int
    directions: tuple[str, ...]
    issued_at: datetime


class BesideRuntime:
    """Ticket queue + per-step fan-out + one delivery per group.

    Dependencies are injected as narrow callables so the runtime never reaches back into the ego
    or the ghost: ``session_source`` returns the *current* ego session (a checkout replaces it),
    ``send_signal`` hands one signal to the mindflow, and ``config_source`` re-reads the config
    store so a runtime edit takes effect on the next group.
    """

    def __init__(
            self,
            *,
            lifecycle: EgoLifecycle,
            session_source: Callable[[], DshSession | None],
            send_signal: Callable[[Signal], None],
            config_source: Callable[[], EgoBesideMindConfig],
            prompt_manager: PromptManager | None = None,
            logger: LoggerItf | None = None,
    ) -> None:
        self._lifecycle = lifecycle
        self._session_source = session_source
        self._send_signal = send_signal
        self._config_source = config_source
        self._prompt_manager = prompt_manager
        self._logger = logger or get_moss_logger()
        self._tickets: list[BesideTicket] = []
        self._tasks: set[asyncio.Task[None]] = set()
        self._seq = 0
        self._discard: Discard | None = None

    # ── lifecycle ──────────────────────────────────────────────────

    async def __aenter__(self) -> "BesideRuntime":
        self._discard = self._lifecycle.on_step_end(self._on_step_end)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._discard is not None:
            self._discard()
            self._discard = None
        self._tickets.clear()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def wait_all_done(self, timeout: float | None = None) -> None:
        """Drain the in-flight groups (tests, and a graceful shutdown before the bus closes)."""
        if not self._tasks:
            return
        pending = set(self._tasks)
        if timeout is None:
            await asyncio.gather(*pending, return_exceptions=True)
            return
        _, pending = await asyncio.wait(pending, timeout=timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # ── the command side ───────────────────────────────────────────

    def directions_max(self) -> int:
        """The cap the next group is truncated at — read live, so a config edit reaches the
        instruction on its next refresh without a restart."""
        return self._config_source().directions_max

    def think(self, text: str, sep: str = "||") -> str:
        """Queue one group and return its receipt; the group starts at the next step end.

        The receipt is plain text, not a container: it is read once, to check what the call
        actually parsed — how many directions went in, how many the cap dropped. That check is
        the only feedback a wrong separator ever gets, so it must say the count.

        Empty segments are dropped; more than the configured cap is truncated (a cap is a cost
        bound, not a parse error). No direction at all is a caller error — a group with nothing
        to think about would produce an empty delivery.
        """
        if not sep:
            raise ValueError("beside.think needs a non-empty separator")
        parts = tuple(part.strip() for part in text.split(sep) if part.strip())
        if not parts:
            raise ValueError(f"beside.think needs at least one non-empty direction, separated by `{sep}`")
        directions = parts[: self._config_source().directions_max]
        self._seq += 1
        ticket = BesideTicket(seq=self._seq, directions=directions, issued_at=datetime.now().astimezone())
        self._tickets.append(ticket)
        receipt = f"beside-think #{ticket.seq}: {len(directions)} direction(s)"
        dropped = len(parts) - len(directions)
        if dropped:
            receipt += f", {dropped} dropped by the cap"
        return receipt

    # ── the step edge ──────────────────────────────────────────────

    async def _on_step_end(self, event: StepEndEvent) -> None:
        """Spawn every pending group; never await a model call here.

        This listener is itself one of the lifecycle bus's tasks, so creating tasks is safe. A
        ticket issued during step N is queued before step N's ``step/end`` arrives, which is
        exactly "start at the next step".
        """
        if not self._tickets:
            return
        tickets, self._tickets = self._tickets, []
        session = self._session_source()
        for ticket in tickets:
            task = asyncio.create_task(
                self._run_group(ticket, event, session),
                name=f"beside-think:{ticket.seq}",
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run_group(self, ticket: BesideTicket, edge: StepEndEvent, session: DshSession | None) -> None:
        config = self._config_source()
        if session is None:
            outcomes: list[object] = [RuntimeError("no live ego session to think beside") for _ in ticket.directions]
        else:
            anchor = DshSessionAnchor(session_id=session.session_id, seq=edge.seq, turn=edge.turn)
            outcomes = list(await asyncio.gather(*(
                self._ask(session, anchor, ticket, index, direction, config)
                for index, direction in enumerate(ticket.directions, start=1)
            )))
        message = self._result_message(ticket, outcomes)
        self._send_signal(new_mind_delivery_signal(
            message,
            kind="beside.think",
            description=f"beside think #{ticket.seq}",
        ))
        failures = sum(1 for outcome in outcomes if isinstance(outcome, BaseException))
        self._logger.info(
            "beside think #%d delivered: %d direction(s), %d error(s)",
            ticket.seq, len(outcomes), failures,
        )

    async def _ask(
            self,
            session: DshSession,
            anchor: DshSessionAnchor,
            ticket: BesideTicket,
            index: int,
            direction: str,
            config: EgoBesideMindConfig,
    ) -> str | Exception:
        """One off-record round for one direction; failures are returned, not raised."""
        container = (
            f'<emanation-reminder task="{DefaultPrompterKeys.BESIDE_THINK}"'
            f'{emanation_position(at=ticket.issued_at)}'
            f' seq="{ticket.seq}" i="{index}">\n'
            f"{BESIDE_SESSION_LINE}"
            f"<instruction>{escape(self._preamble())}</instruction>\n"
            f"The beside task: {escape(direction)}\n"
            "</emanation-reminder>"
        )
        try:
            reply = await session.chat_off_record(
                [[ContentBlock(type="text", text=container)]],
                anchor=anchor,
                llm={"reasoningEffort": config.effort},
                max_tokens=config.max_tokens or None,
                timeout=config.timeout,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a direction's failure is a result, not a crash
            return exc
        if reply.tool_calls:
            asked = ", ".join(call.name or "?" for call in reply.tool_calls)
            return BesideNoAnswer(f"asked to run a tool instead of answering: {asked}")
        text = (reply.text or "").strip()
        if not text:
            return BesideNoAnswer(
                f"came back empty (finish_reason={reply.finish_reason or 'unknown'})"
            )
        return text

    def _preamble(self) -> str:
        preamble = BESIDE_THINK_SEED
        if self._prompt_manager is not None:
            try:
                preamble = self._prompt_manager.render(DefaultPrompterKeys.BESIDE_THINK)
            except Exception:  # noqa: BLE001 — a ghost home without the asset falls back to the seed
                preamble = BESIDE_THINK_SEED
        return preamble

    @staticmethod
    def _result_message(ticket: BesideTicket, outcomes: list[object]) -> Message:
        """``<beside-think seq="k"><direction i=…>…</direction><error i=…>…</error></beside-think>``."""
        children: list[Message] = []
        for index, outcome in enumerate(outcomes, start=1):
            if isinstance(outcome, BaseException):
                children.append(Message.new(tag="error", attributes={"i": str(index)}).with_content(str(outcome)))
            else:
                children.append(Message.new(tag="direction", attributes={"i": str(index)}).with_content(str(outcome)))
        return (
            Message.new(tag="beside-think", attributes={"seq": str(ticket.seq)}, timestamp=True)
            .with_messages(*children)
        )
