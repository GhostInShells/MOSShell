"""DshSchedulerRecaller — a scheduler anchor that points at a dsh session turn.

The scheduler stores an opaque anchor with every plan. This is the dolores binding that
knows what that anchor means: a ``(session_id, turn)`` pair — the same position an ego
commit covers, so a plan written in turn N can be reopened exactly the way memento reopens
a commit's context.

It depends on ``DshConnection`` only. Memento is not involved: coordinates are memento's
vocabulary, and a plan has no coordinate until someone looks one up — the recaller talks
to the session directly, which is what "chat with the self who wrote this" actually needs.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from xml.sax.saxutils import escape

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.mindflow.scheduler import (
    AnchorRef,
    SchedulerRecaller,
    decode_anchor,
    encode_anchor,
)
from ghoshell_moss.deepseek_harness.v0_2_0 import (
    ContentBlock,
    DshConnection,
    DshSessionAnchor,
)
from ghoshell_moss.ghosts.dolores._ego import EgoState
from ghoshell_moss.ghosts.dolores._ego_memento import log_emanation_usage
from ghoshell_moss.ghosts.dolores.constants import NIRMANAKAYA_REMINDER
from ghoshell_moss.ghosts.dolores.mindset.emanation import emanation_position

__all__ = ["METATYPE", "DshSchedulerRecaller"]

METATYPE = "dsh-session"
"""Anchor convention: ``{session: <id>, turn: <n>}`` — one point in one ego session."""

_ANCHOR_INSTRUCTION = (
    "这是一条从未来发回的问题：你在过去写下了一条定时计划，现在它到期了。"
    "你当时只留下了计划本身，没有留下别的说明。"
    "回答来者的提问，只依据你到这个时点为止知道的东西；不知道就说不知道。"
)

#: Off-record calls use the low reasoning tier: the note/chat bypass paths share one prefix
#: cache family, and switching tiers would break it (see ``_ego_memento._BYPASS_EFFORT``).
_BYPASS_EFFORT = "low"


class DshSchedulerRecaller(SchedulerRecaller):
    """Stamp the live ego position; reopen it through ``chat_off_record``."""

    def __init__(
            self,
            connection: DshConnection,
            state: EgoState,
            *,
            logger: LoggerItf | None = None,
    ) -> None:
        self._connection = connection
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
        return encode_anchor(METATYPE, session=session_id, turn=int(self._state.turn))

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
        target = DshSessionAnchor(session_id=str(session_id), turn=int(turn))
        # follow=False: this only borrows the context; the live ego may still be following it.
        session = self._connection.attach_session(str(session_id), self._logger, follow=False)
        try:
            reply = await session.chat_off_record(
                [[ContentBlock(type="text", text=self._message(prompt, int(turn)))]],
                anchor=target,
                llm={"reasoningEffort": _BYPASS_EFFORT},
            )
        finally:
            with contextlib.suppress(Exception):
                await session.close()
        log_emanation_usage(
            self._logger, kind="scheduler.chat", subject=f"turn={turn}", usage=reply.usage,
        )
        if not reply.text:
            raise RuntimeError(
                f"the self of turn {turn} answered nothing"
                + (f" (tool calls: {[call.name for call in reply.tool_calls]})"
                   if reply.tool_calls else "")
            )
        return reply.text

    @staticmethod
    def _message(prompt: str, turn: int) -> str:
        position = emanation_position(at=datetime.now().astimezone())
        return (
            f"<{NIRMANAKAYA_REMINDER} task=\"scheduler.plan\"{position} planned_turn=\"{turn}\">\n"
            f"<instruction>{escape(_ANCHOR_INSTRUCTION)}</instruction>\n"
            f"<question>{escape(prompt)}</question>\n"
            f"</{NIRMANAKAYA_REMINDER}>"
        )
