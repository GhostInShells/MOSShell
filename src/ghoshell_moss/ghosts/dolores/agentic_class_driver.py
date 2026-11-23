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
import json
import linecache
import logging
import os
import re
import sys
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import parse_qs, urlparse

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

__all__ = ["DoloresAgentClassDriver", "close_agent_class_runtime"]

logger = logging.getLogger(__name__)

_LOOPS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, DoloresAgentClassDriver] = (
    weakref.WeakKeyDictionary()
)
_LOOPS_LOCK = threading.Lock()
_ENV_RELATIVE = Path("runtime/cache/agentic_class/dolores.json")
_CACHE_RELATIVE = Path("runtime/cache/agentic_class")

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
        self._agent_runtime = _AgentRuntime()
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

    @classmethod
    def publish_environment(
        cls,
        *,
        dsh_url: str,
        workspace_path: Path | str,
        project_path: Path | str,
        ghost_home: Path | str | None = None,
        default_workspace: Path | str | None = None,
        pid: int | None = None,
    ) -> Path:
        """Publish Dolores' own runtime environment for the class driver to read."""
        path = Path(workspace_path) / _ENV_RELATIVE
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "dsh_url": dsh_url,
            "workspace_path": str(Path(workspace_path)),
            "project_path": str(Path(project_path)),
            "ghost_home": str(ghost_home) if ghost_home else None,
            "default_workspace": str(default_workspace or project_path),
            "pid": pid,
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.chmod(0o600)
        tmp.replace(path)
        return path

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
        env = self._agent_runtime.environment()
        workspace = _resolve_workspace(env.project_path, config.workspace)
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
            parsed = _parse_answer(answer)
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


@dataclass(frozen=True)
class _Environment:
    dsh_url: str
    workspace_path: Path
    project_path: Path
    ghost_home: Path | None
    default_workspace: Path
    pid: int | None


class _AgentRuntime:
    """Dolores-owned dsh connection, fed by its environment declaration."""

    def __init__(self) -> None:
        self._connection: DshConnection | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._environment: _Environment | None = None

    def environment(self) -> _Environment:
        if self._environment is not None:
            return self._environment
        env = Environment.discover(bootstrap=True)
        path = env.workspace_path / _ENV_RELATIVE
        if not path.is_file():
            raise AgentMethodUnavailableError(
                f"agentic class: no environment declaration at {path}; start Dolores first"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = payload.get("pid")
        if pid is not None:
            try:
                os.kill(int(pid), 0)
            except (OSError, ValueError) as exc:
                raise AgentMethodUnavailableError(
                    f"agentic class: dsh pid {pid} is not alive"
                ) from exc
        self._environment = _Environment(
            dsh_url=payload["dsh_url"],
            workspace_path=Path(payload.get("workspace_path", env.workspace_path)),
            project_path=Path(payload.get("project_path", env.project_path)),
            ghost_home=Path(payload["ghost_home"]) if payload.get("ghost_home") else None,
            default_workspace=Path(payload.get("default_workspace", env.project_path)),
            pid=pid,
        )
        return self._environment

    async def connection(self) -> DshConnection:
        if self._closed:
            raise AgentMethodUnavailableError("agentic class: runtime is closed")
        current = self._connection
        if current is not None and current.is_running():
            return current
        async with self._lock:
            current = self._connection
            if current is not None and current.is_running():
                return current
            env = self.environment()
            config = _dsh_config(env.dsh_url)
            connection = DshConnection(config)
            try:
                await connection.__aenter__()
            except BaseException as exc:
                with contextlib.suppress(Exception):
                    await connection.__aexit__(type(exc), exc, exc.__traceback__)
                raise AgentMethodUnavailableError(
                    f"agentic class: cannot connect to dsh at {config.base_url}: {exc}"
                ) from exc
            self._connection = connection
            logger.info("agentic class: connected to dsh %s", config.base_url)
            return connection

    async def debug_artifact(self, session_id: str, turn: int, text: str) -> Path:
        path = (
            self.environment().workspace_path
            / _CACHE_RELATIVE
            / session_id
            / f"debug-{turn}.txt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        connection = self._connection
        self._connection = None
        if connection is not None:
            with contextlib.suppress(Exception):
                await connection.__aexit__(None, None, None)


def _dsh_config(url: str) -> DshConnectionConfig:
    parsed = urlparse(url)
    if not parsed.hostname:
        raise AgentMethodUnavailableError(f"agentic class: invalid dsh url {url!r}")
    token = parse_qs(parsed.query).get("token", [None])[0]
    return DshConnectionConfig(host=parsed.hostname, port=parsed.port or 3083, token=token)


@dataclass(frozen=True)
class ParsedAnswer:
    run: str | None = None
    debug: str | None = None


_RUN_RE = re.compile(r"<\|?run\|?>\s*(.*?)\s*<\|?/run\|?>", re.DOTALL | re.IGNORECASE)
_DEBUG_RE = re.compile(r"<\|?debug\|?>\s*(.*?)\s*<\|?/debug\|?>", re.DOTALL | re.IGNORECASE)
_RUN_FENCE_RE = re.compile(r"```[^\n]*\brun\b[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_DEBUG_FENCE_RE = re.compile(r"```[^\n]*\bdebug\b[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _parse_answer(text: str | None) -> ParsedAnswer:
    if not text:
        return ParsedAnswer()
    run = _RUN_RE.search(text)
    debug = _DEBUG_RE.search(text)
    if run is None and debug is None:
        run = _RUN_FENCE_RE.search(text)
        debug = _DEBUG_FENCE_RE.search(text)
    return ParsedAnswer(
        run=_strip_fence(run.group(1)) if run is not None else None,
        debug=_strip_fence(debug.group(1)) if debug is not None else None,
    )


def _strip_fence(code: str) -> str:
    code = code.strip()
    if not code.startswith("```"):
        return code
    lines = code.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _compile_generated(
    namespace: ModuleType,
    code: str,
    name: str,
    expected_signature: inspect.Signature | None = None,
) -> Any:
    try:
        compiled = compile(code, f"<agentic_class:{namespace.__name__}>", "exec")
        exec(compiled, namespace.__dict__)  # noqa: S102 — executing the answer is the contract
    except (SyntaxError, IndentationError) as exc:
        raise AgentMethodProtocolError(f"generated code does not compile: {exc}") from exc
    fn = namespace.__dict__.get(name)
    if name == "debug" and fn is None:
        fn = namespace.__dict__.get("__debug__")
    if fn is None:
        raise AgentMethodProtocolError(f"generated code did not define {name}")
    if not inspect.iscoroutinefunction(fn):
        raise AgentMethodProtocolError(f"generated {name} is not an async def")
    if expected_signature is not None:
        actual = inspect.signature(fn)
        if _signature_shape(actual) != _signature_shape(expected_signature):
            raise AgentMethodProtocolError(
                f"signature mismatch: expected {expected_signature}, got {actual}"
            )
    return fn


def _signature_shape(signature: inspect.Signature) -> tuple[tuple[str, Any, str], ...]:
    return tuple(
        (parameter.name, parameter.kind, _default_repr(parameter.default))
        for parameter in signature.parameters.values()
    )


def _default_repr(value: Any) -> str:
    if value is inspect.Parameter.empty:
        return "<empty>"
    return _safe_repr(value)


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
    return _compile_generated(ctx.sandbox, code, name, expected)


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
