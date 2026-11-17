"""Emanation task — one off-record round with a session, two answer shapes.

``request`` asks a structured question: the container carries the response schema, and the
answer is a **tool call**, ``moss_emanation_task_response(id=…, resp={…})``. In an off-record
round the harness never executes a tool — it hands the call back — so the call *is* the answer,
structured and separable from prose. A schema-carrying tool call can be validated, retried, and
billed per attempt.

``chat`` asks a free-text question: the container is an ``<emanation-reminder>`` (the same shape
memento chat uses), the instruction is a prompt asset, and the answer is prose. It is one round,
no retry — a tool call instead of prose is a failed answer, delivered as such.

The tool is declared in the ego preset (``dsh_plugin/index.ts``) and does nothing on the main
path — it answers "no pending request". Its job is to keep the tool menu identical between a
session's own turns and every off-record call, which is what lets those calls inherit the
session's prefix instead of starting a new cache family.

The module owns the transport (attach, one call, close, usage) and the container assembly. It
does not own anchor semantics, prompt content, or what a caller does with the answer.
"""

from __future__ import annotations

import contextlib
import itertools
import json
from typing import TYPE_CHECKING, TypeVar
from xml.sax.saxutils import escape

from pydantic import BaseModel, ValidationError

from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.contracts.configs import ConfigType
from ghoshell_moss.core.concepts.tools import model_json_schema_simplified
from ghoshell_moss.deepseek_harness.v0_2_0 import ContentBlock, DshSessionAnchor, TokenUsage
from ghoshell_moss.ghosts.dolores.constants import NIRMANAKAYA_REMINDER

if TYPE_CHECKING:
    from ghoshell_moss.deepseek_harness.v0_2_0 import DshConnection
    from ghoshell_moss.ghosts.dolores.mindset.prompter import PromptManager

__all__ = [
    "EmanationChatError",
    "EmanationTaskConfig",
    "EmanationTaskError",
    "EmanationTasks",
    "EmanationTasksProvider",
    "TOOL_NAME",
    "log_emanation_usage",
]

TOOL_NAME = "moss_emanation_task_response"

T = TypeVar("T", bound=BaseModel)

_INSTRUCTION = (
    "Answer this request with exactly one tool call: "
    '{tool}(id="{request_id}", resp={{...}}). `resp` must be a JSON object that validates '
    "against the response schema below — never a string, never prose. Do not call any other "
    "tool; do not answer in prose."
)

_RETRY = (
    '<emanation-task-retry id="{request_id}">\n'
    "Your previous attempt was rejected: {reason}\n"
    "Answer again with exactly one {tool} call carrying this id.\n"
    "</emanation-task-retry>"
)


def log_emanation_usage(logger: LoggerItf, *, kind: str, subject: str, usage: TokenUsage) -> None:
    """One INFO line per off-record model call — the ghost's token accounting.

    An off-record call never enters the session log, so without this line the ghost's
    parallel thinking is invisible to cost statistics. The format stays flat key=value
    pairs: a future aggregator greps one prefix instead of parsing prose.
    """
    logger.info(
        "emanation usage kind=%s %s input=%s cache_read=%s cache_write=%s output=%s reasoning=%s",
        kind, subject,
        usage.inputTokens, usage.cacheReadTokens, usage.cacheWriteTokens,
        usage.outputTokens, usage.reasoningTokens,
    )


class EmanationTaskError(RuntimeError):
    """No valid answer arrived within the attempts; the message carries the last reason."""


class EmanationChatError(RuntimeError):
    """A chat round got no usable prose back — the message says which shape it got instead."""


class EmanationTaskConfig(ConfigType):
    """Defaults for every request — per-call arguments override them."""

    timeout: float = 60.0
    retries: int = 1
    effort: str = "low"

    @classmethod
    def conf_name(cls) -> str:
        return 'emanation_task_config'


class EmanationTasks:
    """Issue one request to a session at an anchor and return the validated answer."""

    def __init__(
            self,
            connection: "DshConnection",
            *,
            timeout: float = 60.0,
            retries: int = 1,
            effort: str = "low",
            logger: LoggerItf | None = None,
            prompts: PromptManager | None = None,
    ) -> None:
        self._connection = connection
        self._timeout = timeout
        self._retries = max(0, retries)
        self._effort = effort
        self._logger = logger or get_moss_logger()
        self._prompts = prompts
        self._ids = itertools.count(1)

    async def request(
            self,
            prompt: str,
            model: type[T],
            *,
            anchor: DshSessionAnchor,
            timeout: float | None = None,
            retries: int | None = None,
            effort: str | None = None,
            max_tokens: int | None = None,
    ) -> T:
        """Ask the session at ``anchor`` one question and return the validated model.

        Each attempt is one off-record call at the same anchor: the request container, plus a
        correction line when the previous attempt was rejected. Nothing enters the record.

        :raise EmanationTaskError: no valid answer within ``retries + 1`` attempts.
        """
        request_id = f"et-{next(self._ids)}"
        container = self._container(request_id, prompt, model_json_schema_simplified(model))
        attempts = (self._retries if retries is None else max(0, retries)) + 1
        correction = ""
        reason = ""

        async with self._attach(anchor.session_id) as session:
            for attempt in range(attempts):
                bodies = [container]
                if correction:
                    bodies.append(correction)
                reply = await session.chat_off_record(
                    [[ContentBlock(type="text", text=body)] for body in bodies],
                    anchor=anchor,
                    llm={"reasoningEffort": effort or self._effort},
                    max_tokens=max_tokens,
                    timeout=timeout if timeout is not None else self._timeout,
                )
                self._log_usage(request_id, attempt, reply)

                value, reason = self._parse(reply, request_id, model)
                if value is not None:
                    return value
                correction = _RETRY.format(request_id=request_id, reason=reason, tool=TOOL_NAME)

        raise EmanationTaskError(
            f"{request_id}: no valid answer after {attempts} attempt(s) — {reason}"
        )

    async def chat(
            self,
            prompt: str,
            *,
            anchor: DshSessionAnchor,
            task_key: str,
            position: str,
            seed: str,
            kind: str = "chat",
            subject: str = "",
            usage_subject: str = "",
    ) -> str:
        """Ask the self at ``anchor`` one free-text question — one off-record round.

        The prompt asset named by ``task_key`` becomes the ``<instruction>``; ``seed`` is the
        fallback when no prompt registry is bound. ``position`` is the caller's coordinate
        attributes (``emanation_position`` output plus any caller-specific attribute). A reply
        with no prose is a failed answer, never an empty string.

        :raise EmanationChatError: the reply carried a tool call or no text.
        """
        instruction = self._instruction(task_key, seed)
        task_attr = escape(task_key, {'"': "&quot;"})
        body = (
            f'<{NIRMANAKAYA_REMINDER} task="{task_attr}"{position}>\n'
            f"<instruction>{escape(instruction)}</instruction>\n"
            f"<question>{escape(prompt)}</question>\n"
            f"</{NIRMANAKAYA_REMINDER}>"
        )
        async with self._attach(anchor.session_id) as session:
            # No explicit timeout: the off-record transport keeps its own default (120s), and
            # chat is not the request path's short, retried call.
            reply = await session.chat_off_record(
                [[ContentBlock(type="text", text=body)]],
                anchor=anchor,
                llm={"reasoningEffort": self._effort},
            )
        if getattr(reply, "usage", None) is not None:
            log_emanation_usage(
                self._logger, kind=kind, subject=usage_subject or subject or task_key,
                usage=reply.usage,
            )
        text = reply.text
        if text:
            return text
        where = subject or task_key
        if reply.tool_calls:
            asked = "; ".join(f"{call.name}({call.arguments[:120]})" for call in reply.tool_calls)
            raise EmanationChatError(
                f"chat with {where} answered with a tool call instead of prose: {asked}"
            )
        raise EmanationChatError(f"chat with {where} returned no text")

    # ── the four things this module owns ──

    @contextlib.asynccontextmanager
    async def _attach(self, session_id: str):
        """One detached session per round: borrow the context, never join its follow stream."""
        session = self._connection.attach_session(session_id, self._logger, follow=False)
        try:
            yield session
        finally:
            with contextlib.suppress(Exception):
                await session.close()

    def _instruction(self, task_key: str, seed: str) -> str:
        """The prompt asset when one is bound; the seed is the honest fallback."""
        if self._prompts is None:
            return seed
        return self._prompts.render(task_key)

    @staticmethod
    def _container(request_id: str, prompt: str, schema: dict) -> str:
        return (
            f'<emanation-task-request id="{request_id}">\n'
            f"{_INSTRUCTION.format(tool=TOOL_NAME, request_id=request_id)}\n"
            f"{prompt}\n"
            f"<response-schema>\n"
            f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
            f"</response-schema>\n"
            f"</emanation-task-request>"
        )

    @staticmethod
    def _parse(reply, request_id: str, model: type[T]) -> tuple[T | None, str]:
        """Return (value, "") on success, (None, reason) on a rejected attempt."""
        call = next((c for c in reply.tool_calls if c.name == TOOL_NAME), None)
        if call is None:
            if reply.text:
                return None, f"no tool call; the answer was prose: {reply.text[:160]!r}"
            asked = ", ".join(c.name or "?" for c in reply.tool_calls) or "nothing"
            return None, f"no tool call; the model asked for {asked}"
        try:
            payload = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as error:
            return None, f"{TOOL_NAME} arguments were not JSON: {error}"
        if not isinstance(payload, dict):
            return None, f"{TOOL_NAME} arguments must be an object, got {type(payload).__name__}"
        if payload.get("id") != request_id:
            return None, f"wrong id {payload.get('id')!r}; this request is {request_id!r}"
        resp = payload.get("resp")
        if not isinstance(resp, dict):
            return None, f"resp must be a JSON object, got {type(resp).__name__}"
        try:
            return model.model_validate(resp), ""
        except ValidationError as error:
            return None, f"resp did not validate: {error.errors()[:3]}"

    def _log_usage(self, request_id: str, attempt: int, reply) -> None:
        usage = getattr(reply, "usage", None)
        self._logger.info(
            "emanation-task id=%s attempt=%d in=%s cache_read=%s out=%s",
            request_id,
            attempt,
            getattr(usage, "inputTokens", 0),
            getattr(usage, "cacheReadTokens", 0),
            getattr(usage, "outputTokens", 0),
        )


class EmanationTasksProvider(Provider[EmanationTasks]):
    """IoC provider — the module is a plain singleton built from what the runtime published."""

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> EmanationTasks:
        from ghoshell_moss.deepseek_harness.v0_2_0 import DshConnection

        from ghoshell_moss.ghosts.dolores.configs import DoloresConfigStore
        from ghoshell_moss.ghosts.dolores.mindset.prompter import PromptManager

        config = con.force_fetch(DoloresConfigStore).get_or_create(EmanationTaskConfig())
        return EmanationTasks(
            con.force_fetch(DshConnection),
            timeout=config.timeout,
            retries=config.retries,
            effort=config.effort,
            prompts=con.force_fetch(PromptManager),
        )
