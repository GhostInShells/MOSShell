"""Emanation task — one structured question to a session, one validated answer.

A request is a container the session sees inside an off-record round::

    <emanation-task-request id="et-3">
    …the question, the instruction…
    <response-schema>{json}</response-schema>
    </emanation-task-request>

The answer is a **tool call**, ``moss_emanation_task_response(id=…, resp={…})``. In an
off-record round the harness never executes a tool — it hands the call back — so the call *is*
the answer, structured and separable from prose. That is the whole reason this mechanism exists:
a free-text answer has to be parsed and hoped over; a schema-carrying tool call can be validated,
retried, and billed per attempt.

The tool is declared in the ego preset (``dsh_plugin/index.ts``) and does nothing on the main
path — it answers "no pending request". Its job is to keep the tool menu identical between a
session's own turns and every off-record call, which is what lets those calls inherit the
session's prefix instead of starting a new cache family.

This module is a plain implementation, not an abstraction: it owns four things — the container,
the id, the parse, and the retry — and its contract is the returned value.
"""

from __future__ import annotations

import itertools
import json
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.contracts.configs import ConfigType
from ghoshell_moss.core.concepts.tools import model_json_schema_simplified
from ghoshell_moss.deepseek_harness.v0_2_0 import ContentBlock, DshSessionAnchor

if TYPE_CHECKING:
    from ghoshell_moss.deepseek_harness.v0_2_0 import DshConnection

__all__ = [
    "EmanationTasks",
    "EmanationTaskError",
    "EmanationTaskConfig",
    "EmanationTasksProvider",
    "TOOL_NAME",
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


class EmanationTaskError(RuntimeError):
    """No valid answer arrived within the attempts; the message carries the last reason."""


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
    ) -> None:
        self._connection = connection
        self._timeout = timeout
        self._retries = max(0, retries)
        self._effort = effort
        self._logger = logger or get_moss_logger()
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
        session = self._connection.attach_session(anchor.session_id, self._logger, follow=False)
        correction = ""
        reason = ""

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

    # ── the four things this module owns ──

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

        config = con.force_fetch(DoloresConfigStore).get_or_create(EmanationTaskConfig())
        return EmanationTasks(
            con.force_fetch(DshConnection),
            timeout=config.timeout,
            retries=config.retries,
            effort=config.effort,
        )
