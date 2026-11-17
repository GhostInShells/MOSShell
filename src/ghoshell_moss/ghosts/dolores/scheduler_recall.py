"""DshSchedulerRecaller — a scheduler anchor that points at a dsh session turn.

The scheduler stores an opaque anchor with every plan. This is the dolores binding that
knows what that anchor means: a ``(session_id, turn)`` pair — the position the plan was
written in. The turn comes from the ego's live cursor (``turn/start`` → ``turn/end``),
never from a fabricated offset.

It depends on ``EmanationTasks`` for the off-record round: the same transport, container,
prompt asset, failure mapping and usage line as memento chat. The recaller owns only the
anchor semantics and the plan-specific position.
"""

from __future__ import annotations

from datetime import datetime

from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.mindflow.scheduler import (
    AnchorRef,
    SchedulerRecaller,
    decode_anchor,
    encode_anchor,
)
from ghoshell_moss.deepseek_harness.v0_2_0 import DshSessionAnchor
from ghoshell_moss.ghosts.dolores._ego import EgoState
from ghoshell_moss.ghosts.dolores.mindset.emanation import emanation_position
from ghoshell_moss.ghosts.dolores.mindset.emanation_task import EmanationTasks
from ghoshell_moss.ghosts.dolores.mindset.prompter import DefaultPrompterKeys
from ghoshell_moss.ghosts.dolores.mindset.seeds import SCHEDULER_CHAT_SEED

__all__ = ["METATYPE", "DshSchedulerRecaller", "SchedulerRecallerProvider"]

METATYPE = "dsh-session"
"""Anchor convention: ``{session: <id>, turn: <n>}`` — one point in one ego session."""


class DshSchedulerRecaller(SchedulerRecaller):
    """Stamp the live ego turn; reopen it through the shared emanation chat."""

    def __init__(
            self,
            tasks: EmanationTasks | None,
            state: EgoState,
            *,
            logger: LoggerItf | None = None,
    ) -> None:
        self._tasks = tasks
        self._state = state
        self._logger = logger or get_moss_logger()

    def metatype(self) -> str:
        return METATYPE

    def stamp(self) -> AnchorRef | None:
        session_id = self._state.session_id
        if not session_id:
            # Before the ego session exists there is no "self" to point at; the plan is
            # still valid, it just cannot be asked why it exists.
            return None
        turn = self._state.turn_in_flight
        if turn is None:
            turn = self._state.turn
        if not turn:
            # No turn has started (or ended) yet: an anchor at turn 0 would point at a
            # position that does not exist. Honest absence beats a fabricated coordinate.
            return None
        return encode_anchor(METATYPE, session=session_id, turn=int(turn))

    def describe(self, anchor: AnchorRef) -> str:
        data = decode_anchor(anchor)
        session_id = data.get("session")
        turn = data.get("turn")
        if not session_id or turn is None:
            return ""
        short = str(session_id)[:8]
        return f"planned by the self of turn {turn} (session {short}…)"

    async def chat(self, anchor: AnchorRef, prompt: str) -> str:
        data = decode_anchor(anchor)
        session_id = data.get("session")
        turn = data.get("turn")
        if not session_id or turn is None:
            raise ValueError(
                "this plan carries no session anchor — it was written before the ego "
                "session existed, so there is no past self to talk to"
            )
        if self._tasks is None:
            raise RuntimeError("no emanation caller is bound; the scheduler cannot chat")
        target = DshSessionAnchor(session_id=str(session_id), turn=int(turn))
        return await self._tasks.chat(
            prompt,
            anchor=target,
            task_key=DefaultPrompterKeys.SCHEDULER_CHAT,
            position=emanation_position(at=datetime.now().astimezone())
                     + f' planned_turn="{int(turn)}"',
            seed=SCHEDULER_CHAT_SEED,
            kind="scheduler.chat",
            subject=f"plan turn={int(turn)}",
            usage_subject=f"turn={int(turn)}",
        )


class SchedulerRecallerProvider(Provider[SchedulerRecaller]):
    """IoC provider — the runtime depends on the contract, not on this concrete class."""

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> SchedulerRecaller:
        return DshSchedulerRecaller(
            con.force_fetch(EmanationTasks),
            con.force_fetch(EgoState),
            logger=con.get(LoggerItf),
        )
