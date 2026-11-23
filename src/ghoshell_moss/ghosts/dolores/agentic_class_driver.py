"""DoloresAgentClassDriver — the dsh runtime behind the agentic-class blueprint.

This driver assumes it runs in a **bash-launched process** (for example `python -c` inside
dsh's bash sandbox), on exactly one event loop. It is per-loop: `for_loop()` creates the
instance bound to the running loop; the loop's owner calls `close()`.

`prepare` is locked and registers the process lifecycle on first use. `call` is blocking:
one method call is one op, and the class source is the op table — the current op travels in
the request envelope, never as a re-rendered method. `close` archives every session by
session id (never by workspace). `AgentCtx` itself has no lifecycle.

The generated `__run__(self, ...)`/`__debug__(self, ...)` are compiled into a facsimile of
the defining module (attrs copied, imports preserved); the first argument is the live
instance. No proxy: the generated function is a module-level local function.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import hashlib
import inspect
import linecache
import logging
import sys
import threading
import weakref
from pathlib import Path
from types import ModuleType
from typing import Any

from ghoshell_dsh.v0_2_0 import DSH_VERSION
from ghoshell_dsh.v0_2_0.types.sessions import SessionCreateParams
from typing_extensions import Self

from ghoshell_moss.agents.blueprint._agentic_class_helpers import reflect_blocks
from ghoshell_moss.agents.blueprint.agent import AgentPreset
from ghoshell_moss.agents.blueprint.agentic_class import (
    AgentClassDriver,
    AgentCtx,
    AgentMethod,
    AgentMethodProtocolError,
    AgentMethodReentryError,
    AgentMethodTimeoutError,
    AgentMethodUnavailableError,
    AgentSessionConflictError,
    CallRecord,
)
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.codex.compiler import Compiler
from ghoshell_moss.deepseek_harness.v0_2_0 import DshConnection, DshConnectionConfig, DshSession
from ghoshell_moss.deepseek_harness.v0_2_0.agent import DshAgentSession
from ghoshell_moss.ghosts.dolores.agent_func import (
    AgentFuncProtocolError,
    AgentRuntime,
    compile_generated,
    parse_answer,
)

__all__ = ["DoloresAgentClassDriver", "close_agent_class_runtime"]

logger = logging.getLogger(__name__)

_LOOPS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, DoloresAgentClassDriver] = (
    weakref.WeakKeyDictionary()
)
_LOOPS_LOCK = threading.Lock()

_GENERATION_ERRORS = (
    SyntaxError,
    IndentationError,
    NameError,
    UnboundLocalError,
    ImportError,
    ModuleNotFoundError,
    AttributeError,
    TypeError,
    AgentMethodProtocolError,
    AgentMethodReentryError,
)


class DoloresAgentClassDriver(AgentClassDriver[DshAgentSession]):
    """One per event loop; a bash subprocess is its usual owner."""

    def __init__(self) -> None:
        self._agent_runtime = AgentRuntime()
        self._project_path = Environment.discover(bootstrap=True).project_path
        self._prepare_lock = asyncio.Lock()
        self._sessions: dict[str, tuple[DshAgentSession, bool]] = {}
        self._config: DshConnectionConfig | None = None
        self._exit_registered = False
        self._closed = False

    # -- per-loop lifecycle --------------------------------------------- #

    @classmethod
    def for_loop(cls) -> Self:
        loop = asyncio.get_running_loop()
        with _LOOPS_LOCK:
            driver = _LOOPS.get(loop)
            if driver is None or driver._closed:
                driver = cls()
                _LOOPS[loop] = driver
            return driver

    # -- prepare -------------------------------------------------------- #

    async def prepare(
        self, ctx: AgentCtx[DshAgentSession], target: object, method: AgentMethod
    ) -> None:
        if ctx.active:
            raise AgentMethodReentryError(
                f"agentic class: {type(target).__name__}.{method.__name__} re-entered"
            )
        async with self._prepare_lock:
            module = _module_for(ctx, target)
            if not ctx.modulename:
                ctx.modulename = module.__name__
            digest = _module_digest(module)
            if ctx.sandbox is None or getattr(ctx, "_sandbox_digest", "") != digest:
                ctx.sandbox = Compiler(
                    source="",
                    origin=module,
                    modulename=module.__name__,
                    filename=f"<agentic_class:{module.__name__}>",
                ).compiled
                ctx._sandbox_digest = digest  # type: ignore[attr-defined]
            await self._ensure_session(ctx, target)
            self._register_exit()

    async def _ensure_session(
        self, ctx: AgentCtx[DshAgentSession], target: object
    ) -> None:
        if ctx.session is not None:
            return
        connection = await self._agent_runtime.connection()
        self._config = connection.config
        config = ctx.config
        workspace = _resolve_workspace(self._project_path, config.workspace)
        writable = _resolve_writable(workspace, config.writable)
        cwd = str(workspace)
        title = ctx.title
        if config.session_id and config.session_id in self._sessions:
            raise AgentSessionConflictError(
                f"agentic class: session id {config.session_id!r} is already bound"
            )
        raw: DshSession | None = None
        try:
            if config.session_id:
                raw = connection.attach_session(config.session_id, follow=True)
                await raw.__aenter__()
            else:
                value = await connection.client.session_create(
                    SessionCreateParams(cwd=cwd, agentPreset=config.preset or None)
                )
                raw = connection.attach_session(str(value.sessionId), follow=True)
                await raw.__aenter__()
            await raw.wait_follow_snapshot(timeout=10.0)
            with contextlib.suppress(Exception):
                await raw.set_policy(
                    sandbox="read-only" if config.readonly else "workspace-write",
                    approval="never",
                )
            with contextlib.suppress(Exception):
                await raw.rename(title=title)
            session = DshAgentSession(
                raw,
                hive="dolores",
                engine=f"dsh@{DSH_VERSION}",
                name=ctx.modulename,
                preset=AgentPreset(id=config.preset),
                title=title,
                cwd=cwd,
            )
            session.prime(title=title, preset=config.preset, cwd=cwd)
            ctx.session = session
            ctx._session_id = raw.session_id
            self._sessions[raw.session_id] = (session, config.archive_after_close)
            logger.info(
                "agentic class: opened session %s for %s (%s) workspace=%s writable=%s",
                raw.session_id,
                ctx.modulename,
                title,
                workspace,
                writable,
            )
        except asyncio.CancelledError:
            if raw is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await asyncio.shield(raw.archive_session(stop=True))
            raise
        except Exception as exc:
            if raw is not None:
                with contextlib.suppress(Exception):
                    await raw.archive_session(stop=True)
            raise AgentMethodUnavailableError(
                f"agentic class: cannot open session for {ctx.modulename}: {exc}"
            ) from exc

    def _register_exit(self) -> None:
        if not self._exit_registered:
            atexit.register(self._archive_at_exit)
            self._exit_registered = True

    def _archive_at_exit(self) -> None:
        """Last resort for a bash subprocess: archive by id over a fresh connection."""
        session_ids = [
            session_id
            for session_id, (_, archive) in self._sessions.items()
            if archive
        ]
        if not session_ids or self._config is None:
            return
        try:
            asyncio.run(self._archive_fresh(self._config, session_ids))
        except Exception:
            logger.exception("agentic class: atexit archive failed")

    @staticmethod
    async def _archive_fresh(config: DshConnectionConfig, session_ids: list[str]) -> None:
        async with DshConnection(config) as connection:
            for session_id in session_ids:
                with contextlib.suppress(Exception):
                    await connection.archive_session(session_id, stop=True)

    # -- call ----------------------------------------------------------- #

    async def call(
        self,
        ctx: AgentCtx[DshAgentSession],
        target: object,
        method: AgentMethod,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if ctx.active:
            raise AgentMethodReentryError(
                f"agentic class: {type(target).__name__}.{method.__name__} re-entered"
            )
        if ctx.session is None:
            await self.prepare(ctx, target, method)
        session = ctx.session
        if session is None:
            raise AgentMethodUnavailableError("agentic class: session was not created")
        raw = session.engine_session
        module = _module_for(ctx, target)
        first = not getattr(ctx, "_instruction_sent", False)
        message = _call_message(ctx, target, method, args, kwargs)
        if first:
            instruction = await _build_instruction(ctx, target, method, module)
            message = f"{instruction}\n\n{message}"
        record = CallRecord(method=method.__name__, args=list(args), kwargs=dict(kwargs))
        try:
            run = await raw.run(message, timeout=ctx.config.timeout)
            ctx._instruction_sent = True  # type: ignore[attr-defined]
            record.session_id = session.id
            result = await self._protocol_loop(
                ctx, target, method, raw, run.final_response or "", args, kwargs
            )
            record.returned = result
            ctx.calls.append(record)
            logger.info(
                "agentic class: %s.%s -> ok (session %s)",
                type(target).__name__,
                method.__name__,
                session.id,
            )
            return result
        except asyncio.TimeoutError as exc:
            with contextlib.suppress(Exception):
                await session.cancel("agentic class: timeout")
            record.error = "timeout"
            record.session_id = session.id
            ctx.calls.append(record)
            raise AgentMethodTimeoutError(
                f"agentic class: {method.__name__} timed out after {ctx.config.timeout}s"
            ) from exc
        except asyncio.CancelledError:
            record.error = "cancelled"
            record.session_id = session.id
            ctx.calls.append(record)
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(raw.cancel())
            raise
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            record.session_id = session.id
            ctx.calls.append(record)
            raise

    async def debug_prompt(
        self,
        ctx: AgentCtx[DshAgentSession],
        target: object,
        method: AgentMethod,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        module = _module_for(ctx, target)
        instruction = await _build_instruction(ctx, target, method, module)
        call = _call_message(ctx, target, method, args, kwargs)
        return f"{instruction}\n\n{call}"

    async def _protocol_loop(
        self,
        ctx: AgentCtx[DshAgentSession],
        target: object,
        method: AgentMethod,
        raw: DshSession,
        answer: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        attempt = 0
        debug_rounds = 0
        while True:
            parsed = parse_answer(answer)
            if parsed.run is not None:
                try:
                    implementation = _compile(
                        ctx, parsed.run, "__run__", inspect.signature(method)
                    )
                except AgentMethodProtocolError as exc:
                    attempt += 1
                    if attempt >= ctx.config.max_attempts:
                        raise
                    answer = await self._run(
                        raw, _protocol_retry(exc, answer), ctx.config.timeout
                    )
                    continue
                ctx.active = True
                try:
                    return await implementation(target, *args, **kwargs)
                except asyncio.CancelledError:
                    with contextlib.suppress(Exception):
                        await raw.cancel()
                    raise
                except Exception as exc:
                    if not isinstance(exc, _GENERATION_ERRORS):
                        raise
                    attempt += 1
                    if attempt >= ctx.config.max_attempts:
                        raise
                    answer = await self._run(
                        raw, _execution_retry(exc), ctx.config.timeout
                    )
                    continue
                finally:
                    ctx.active = False
            if parsed.debug is not None and debug_rounds < ctx.config.max_debug_rounds:
                debug_rounds += 1
                try:
                    debug = _compile(ctx, parsed.debug, "__debug__")
                except AgentMethodProtocolError as exc:
                    attempt += 1
                    if attempt >= ctx.config.max_attempts:
                        raise
                    answer = await self._run(
                        raw, _protocol_retry(exc, answer), ctx.config.timeout
                    )
                    continue
                ctx.active = True
                try:
                    value = await debug(target, *args, **kwargs)
                except Exception as exc:
                    value = f"debug raised {type(exc).__name__}: {exc}"
                finally:
                    ctx.active = False
                text = value if isinstance(value, str) else repr(value)
                text = await self._debug_result(raw.session_id, debug_rounds, text, ctx)
                answer = await self._run(
                    raw, f"<debug-result>\n{text}\n</debug-result>", ctx.config.timeout
                )
                continue
            attempt += 1
            if attempt >= ctx.config.max_attempts:
                raise AgentMethodProtocolError(
                    f"agentic class: {method.__name__} produced no __run__ block; "
                    f"last answer: {answer[:800]!r}"
                )
            answer = await self._run(
                raw,
                _protocol_retry("no __run__ or __debug__ block found", answer),
                ctx.config.timeout,
            )

    async def _run(self, raw: DshSession, message: str, timeout: float) -> str:
        result = await raw.run(message, timeout=timeout)
        return result.final_response or ""

    async def _debug_result(
        self,
        session_id: str,
        turn: int,
        text: str,
        ctx: AgentCtx[DshAgentSession],
    ) -> str:
        cap = ctx.config.debug_result_cap
        if len(text) <= cap:
            return text
        path = await self._agent_runtime.debug_artifact(session_id, turn, text)
        head = text[: cap // 2]
        tail = text[-(cap // 2):]
        return (
            f"{head}\n... ({len(text)} chars; full result at {path}) ...\n{tail}\n"
            "Use your read-only file tools to read the full result."
        )

    # -- cancel / close ------------------------------------------------- #

    async def cancel(self, ctx: AgentCtx[DshAgentSession]) -> None:
        session = ctx.session
        if session is None:
            return
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(session.engine_session.cancel())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for session_id, (session, archive) in list(self._sessions.items()):
            if archive:
                with contextlib.suppress(Exception):
                    await session.engine_session.archive_session(stop=True)
                logger.info("agentic class: archived session %s", session_id)
            with contextlib.suppress(Exception):
                await session.close()
        self._sessions.clear()
        if self._exit_registered:
            atexit.unregister(self._archive_at_exit)
            self._exit_registered = False
        with contextlib.suppress(Exception):
            await self._agent_runtime.aclose()


async def close_agent_class_runtime() -> None:
    """Close the current loop's driver; called by the loop owner, never by AgentCtx."""
    loop = asyncio.get_running_loop()
    with _LOOPS_LOCK:
        driver = _LOOPS.pop(loop, None)
    if driver is not None:
        await driver.close()


# ── helpers ─────────────────────────────────────────────────────────────────


def _resolve_workspace(project_home: object, value: str | None) -> Path:
    home = Path(project_home).resolve()
    path = Path(value).expanduser() if value else Path.cwd()
    if not path.is_absolute():
        path = home / path
    path = path.resolve()
    if not path.is_relative_to(home):
        raise AgentMethodUnavailableError(
            f"agentic class: workspace {path} is outside project home {home}"
        )
    return path


def _resolve_writable(workspace: Path, value: str | None) -> Path:
    path = Path(value).expanduser() if value else workspace
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve()
    if not path.is_relative_to(workspace):
        raise AgentMethodUnavailableError(
            f"agentic class: writable {path} is outside workspace {workspace}"
        )
    return path


def _module_for(ctx: AgentCtx, target: object) -> ModuleType:
    module = sys.modules.get(ctx.modulename)
    if module is None:
        module = inspect.getmodule(target)
    if module is None:
        raise AgentMethodUnavailableError(
            f"agentic class: cannot find the module of {type(target).__name__}"
        )
    return module


def _module_digest(module: ModuleType) -> str:
    file = getattr(module, "__file__", None)
    if file:
        linecache.checkcache(file)
    try:
        source = inspect.getsource(module)
    except Exception:
        source = ""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12] if source else "nosource"


async def _build_instruction(
    ctx: AgentCtx,
    target: object,
    method: AgentMethod,
    module: ModuleType,
) -> str:
    try:
        source = inspect.getsource(module)
    except Exception:
        source = f"# source unavailable for {module.__name__}"
    signature = inspect.signature(method)
    doc = inspect.getdoc(method) or ""
    context = await ctx.context_call(target)
    context_block = (
        f"\n<instance-context>\n{context.strip()}\n</instance-context>\n" if context.strip() else ""
    )
    reflected = reflect_blocks(ctx)
    return f"""You implement one method of an agentic class in the MOSS project.

Class: `{module.__name__}.{type(target).__name__}`
Method: `{method.__name__}`
Signature: `{signature}`
Docstring: {doc}

The method body is never executed. Answer with exactly one module-level async function:

<run>
async def __run__(self, ...same parameters and return annotation as the method):
    ...
</run>

The first parameter is the live instance; the rest matches the method signature. The
wrapper compiles your code into a facsimile of the class's module — every module-level name
(imports included) and every `self` attribute is available. Return the method's value.

If you need to inspect runtime values first, answer instead with:

<debug>
async def __debug__(self, ...same parameters...) -> str:
    ...
</debug>

The wrapper calls it with the real instance and arguments, then sends back <debug-result>
with the string it returned.

This class is an op table: the module source below contains every `@agent_method`; the
current request envelope names the one to answer. Do not re-implement the others.

Answer protocol:
- The answer must contain exactly one <run> or <debug> block; otherwise the call fails
  with AgentMethodProtocolError.
- The signature must match exactly; parameter names, kinds and defaults are checked.
- The return value is passed through unchanged; wrap the op in a plain method if you
  need extra validation.
- Protocol/execution errors retry up to {ctx.config.max_attempts}. If a failure is a
  known irreversible side effect, raise AgentMethodAbortError to stop retries.
- Do not call another @agent_method on the same instance; inline the logic in this <run>.
- One <run> answers one call. Do not build a general-purpose library.
{context_block}
<module-source>
```python
{source}
```
</module-source>
{reflected}"""


def _call_message(
    ctx: AgentCtx,
    target: object,
    method: AgentMethod,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    return f"""<call module="{ctx.modulename}" class="{type(target).__name__}" method="{method.__name__}">
signature: {inspect.signature(method)}
docstring: {inspect.getdoc(method) or ""}
args: {_safe_repr(args)}
kwargs: {_safe_repr(kwargs)}
</call>"""


def _safe_repr(value: Any) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > 4000:
        return text[:4000] + f"... ({len(text)} chars)"
    return text


def _compile(
    ctx: AgentCtx,
    code: str,
    name: str,
    expected: inspect.Signature | None = None,
) -> Any:
    if ctx.sandbox is None:
        raise AgentMethodProtocolError("agentic class: the module facsimile is not prepared")
    try:
        return compile_generated(ctx.sandbox, code, name, expected)
    except AgentFuncProtocolError as exc:
        raise AgentMethodProtocolError(str(exc)) from exc


def _protocol_retry(detail: object, previous: str) -> str:
    return (
        "<protocol-error>\n"
        f"{detail}\n"
        "</protocol-error>\n"
        "Your previous answer was:\n"
        f"{previous[:1200]}\n"
        "Answer with exactly one <run> block (or one <debug> block)."
    )


def _execution_retry(exc: BaseException) -> str:
    import traceback

    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if len(text) > 4000:
        text = text[-4000:]
    hint = ""
    if isinstance(exc, AgentMethodReentryError):
        hint = "\nDo not call @agent_method methods on the same instance; implement the method directly."
    return (
        "<run-error>\n"
        f"{text}\n"
        "</run-error>\n"
        "The implementation raised. Fix it and answer with a new <run> block."
        f"{hint}"
    )
