"""agent_func — a function whose runtime semantics are settled per call.

Use it when the contract is *not* complete at definition time: natural language input,
ambiguous data, or behavior that depends on the call's live context. The function's body is
never executed; the defining module gets one dsh agent session, and each call asks that agent
for an implementation:

    from ghoshell_moss.ghosts.dolores.agent_func import agent_func

    @agent_func
    async def summarize(text: str) -> str:
        \"\"\"Summarize the text in one sentence.\"\"\"
        raise NotImplementedError

    @agent_func(agent_preset="minimal", mode="per_call")
    async def classify(text: str) -> str:
        ...

If the code can be written at definition time, do that instead — a normal function is
cheaper, deterministic and cacheable. agent_func is the scaffold for the part that cannot
be written yet. A stable implementation can later be promoted into ordinary code.

Protocol
--------
The agent answers with one block:

    <debug>
    async def debug(*args, **kwargs) -> str:
        ...
    </debug>

or, for the implementation:

    <run>
    async def __run__(same signature as the called function):
        ...
    </run>

A fenced block around it (```` ```python ... ``` ````) is cosmetic; the parser reads the
tags. `<debug>` runs with the real arguments and its returned string is sent back as
`<debug-result>`; it may be used repeatedly, up to `max_debug_rounds`. `<run>` is compiled
inside a copy of the module namespace, its signature is checked, and it is awaited.

The module may define `async def __context__() -> str`; it runs once when the module
namespace is first built and its text is inserted into the session's instruction. That makes
the context testable without a model.

Inside generated code, `agent_ctx` (and the `emit` alias) is available:

    agent_ctx.emit("note for the next round")   # one-way; never captured from print()
    agent_ctx.note("key", "value")

`print()` is **not** captured — use `agent_ctx.emit`. Notes ride with the next retry, or the
next call to the same module. A debug result larger than `debug_result_cap` is written to
`runtime/cache/agent_func/<session>/debug-<n>.txt` and the model receives a head/tail plus
the path.

Sessions
--------
`__run__` is generated on every call; no implementation is cached. The session is memory,
not a code cache. `mode` selects its scope:

- `per_module` (default) — one session per module + source hash + preset.
- `per_call` — a fresh session per call, archived afterwards.
- `shared` — one session per explicit `session_key`.

`session_id` is an existing live dsh session and wins over `mode`. Sessions are archived when
the ghost runtime closes (except explicitly borrowed sessions), so the durable artifact is
the code, not the conversation. If the source changes, the source hash changes and the next
call gets a fresh session; a running process still holds the old module object, so a restart
is the honest reload.

`agent_preset` chooses an existing dsh preset; `"ego"` is refused because it cannot be
started as a module agent. `model` is `"provider/model"`; `effort` is the reasoning depth.
The stable instruction is the first message — no plugin, no custom preset.

`agent_context(**overrides)` is a per-task scope for model/preset/effort/session overrides
and the call record:

    async with agent_context(effort="high") as ctx:
        await summarize("...")
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import hashlib
import inspect
import linecache
import logging
import re
import threading
import time
import traceback
import uuid
import weakref
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

from ghoshell_dsh.v0_2_0.types.sessions import SessionCreateParams

from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.project import Project
from ghoshell_moss.core.codex.compiler import Compiler
from ghoshell_moss.deepseek_harness.v0_2_0 import (
    DshConnection,
    DshConnectionConfig,
    DshSession,
)

__all__ = [
    "AgentCallRecord",
    "AgentContext",
    "AgentFuncConfig",
    "AgentFuncError",
    "AgentFuncProtocolError",
    "AgentFuncReentryError",
    "AgentFuncTimeoutError",
    "AgentFuncUnavailableError",
    "AgentMode",
    "agent_context",
    "agent_ctx",
    "agent_func",
    "build_instruction",
    "close_agent_func_runtime",
    "parse_answer",
    "recent_calls",
    "resolve_session_key",
]

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Coroutine[Any, Any, Any]])

_CACHE_RELATIVE = Path("runtime") / "cache"
_URL_FILENAME = "dolores_dsh_url.txt"
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_MAX_DEBUG_ROUNDS = 4
_DEFAULT_DEBUG_CAP = 8000
_DEFAULT_TIMEOUT = 600.0
_ARG_REPR_CAP = 4000
_NOTE_CAP = 4000
_TITLE_CAP = 120

_RUN_RE = re.compile(r"<\|?run\|?>\s*(.*?)\s*<\|?/run\|?>", re.DOTALL | re.IGNORECASE)
_DEBUG_RE = re.compile(r"<\|?debug\|?>\s*(.*?)\s*<\|?/debug\|?>", re.DOTALL | re.IGNORECASE)
_RUN_FENCE_RE = re.compile(r"```[^\n]*\brun\b[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_DEBUG_FENCE_RE = re.compile(
    r"```[^\n]*\bdebug\b[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE
)


class AgentFuncError(RuntimeError):
    """Base class for agent_func failures."""


class AgentFuncUnavailableError(AgentFuncError):
    """The module agent cannot be reached: no dsh connection or URL, or a bad session."""


class AgentFuncProtocolError(AgentFuncError):
    """The agent's answer does not satisfy the output protocol."""


class AgentFuncReentryError(AgentFuncError):
    """An agent_func re-entered a module already running in this task."""


class AgentFuncTimeoutError(AgentFuncError):
    """The dsh turn did not finish inside the call's timeout."""


class AgentMode(str, Enum):
    """How much conversation a module agent keeps."""

    PER_MODULE = "per_module"
    PER_CALL = "per_call"
    SHARED = "shared"


@dataclass(frozen=True)
class AgentFuncConfig:
    """The decorator's declaration — static defaults, overridable by ``agent_context``."""

    agent_preset: str = ""
    model: str = ""
    effort: str = ""
    cwd: str | None = None
    mode: AgentMode = AgentMode.PER_MODULE
    session_id: str = ""
    session_key: str = ""
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS
    max_debug_rounds: int = _DEFAULT_MAX_DEBUG_ROUNDS
    timeout: float = _DEFAULT_TIMEOUT
    debug_result_cap: int = _DEFAULT_DEBUG_CAP
    archive_on_close: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("agent_func: max_attempts must be >= 1")
        if self.max_debug_rounds < 0:
            raise ValueError("agent_func: max_debug_rounds must be >= 0")
        if self.session_id and self.session_key:
            raise ValueError("agent_func: session_id and session_key are mutually exclusive")
        if self.mode is AgentMode.SHARED and not (self.session_id or self.session_key):
            raise ValueError("agent_func: shared mode needs session_key (or session_id)")
        if self.agent_preset == "ego":
            raise ValueError("agent_func: the ego preset cannot be started as a module agent")


@dataclass
class AgentCallRecord:
    """One call, kept in the module-level ring buffer for debugging."""

    module: str
    function: str
    session_id: str = ""
    mode: str = ""
    source_hash: str = ""
    attempts: int = 0
    debug_rounds: int = 0
    started_at: float = field(default_factory=time.time)
    duration: float = 0.0
    status: str = ""
    error: str = ""
    generated_code: str = ""
    notes: list[str] = field(default_factory=list)


_RECORDS: deque[AgentCallRecord] = deque(maxlen=100)


def recent_calls(limit: int = 20) -> list[AgentCallRecord]:
    """The last calls, newest last — the in-memory debug surface."""
    if limit <= 0:
        return []
    return list(_RECORDS)[-limit:]


# ── the context generated code can talk to ──────────────────────────────────


@dataclass
class AgentContext:
    """Per-call context — the explicit replacement for print()."""

    module: str = ""
    function: str = ""
    session_id: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    record: AgentCallRecord | None = None

    def emit(self, text: object) -> None:
        """Leave one line for the next round or the next call."""
        value = str(text)
        if len(value) > _NOTE_CAP:
            value = value[:_NOTE_CAP] + f"... ({len(value)} chars)"
        self.notes.append(value)

    def note(self, key: str, value: object) -> None:
        """A keyed note; the same buffer as ``emit``."""
        self.emit(f"{key}: {value}")

    def _drain_notes(self) -> list[str]:
        notes = list(self.notes)
        self.notes.clear()
        return notes


_CURRENT_CTX: contextvars.ContextVar[AgentContext | None] = contextvars.ContextVar(
    "agent_func_context", default=None
)
_ACTIVE_MODULES: contextvars.ContextVar[frozenset[int]] = contextvars.ContextVar(
    "agent_func_active_modules", default=frozenset()
)


class _AgentContextProxy:
    """The object injected into generated code as ``agent_ctx``."""

    def emit(self, text: object) -> None:
        _require_context().emit(text)

    def note(self, key: str, value: object) -> None:
        _require_context().note(key, value)

    @property
    def session_id(self) -> str:
        return _require_context().session_id

    @property
    def function(self) -> str:
        return _require_context().function


agent_ctx = _AgentContextProxy()


def _require_context() -> AgentContext:
    ctx = _CURRENT_CTX.get()
    if ctx is None:
        raise AgentFuncError("agent_ctx used outside an agent_func call")
    return ctx


@contextlib.asynccontextmanager
async def agent_context(**overrides: Any):
    """Per-task overrides + a place to read the call record.

    ``async with agent_context(effort="high") as ctx: ...`` — nested calls inherit the
    overrides; the innermost explicit value wins over the decorator's default.
    """
    ctx = AgentContext(overrides=dict(overrides))
    token = _CURRENT_CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT_CTX.reset(token)


# ── protocol parsing ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParsedAnswer:
    run: str | None = None
    debug: str | None = None


def parse_answer(text: str | None) -> ParsedAnswer:
    """Read one assistant message. Tags are the contract; fences are cosmetic."""
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


# ── instruction / context ───────────────────────────────────────────────────


def build_instruction(
    module: ModuleType,
    *,
    source: str | None = None,
    context: str = "",
) -> str:
    """The stable first message — protocol + module source + optional module context."""
    if source is None:
        source = _module_source(module)
    location = getattr(module, "__file__", "") or "<unknown>"
    context_block = ""
    if context.strip():
        context_block = f"\n<module-context>\n{context.strip()}\n</module-context>\n"
    return f"""You are the module agent for `{module.__name__}` in the MOSS project.

Your task, one call at a time: implement the async function named in the <call> block. The
wrapper compiles your code inside a copy of the module namespace and calls it with the real
arguments. The module source and the call signature are the whole contract.

Answer with exactly one block. The function MUST be named `__run__` — never the original
function name — and its parameters and return annotation must match the call:

<run>
async def __run__(same parameters and return annotation as the called function):
    ...
</run>

If you need to inspect runtime values first, answer instead with:

<debug>
async def debug(*args, **kwargs) -> str:
    ...
</debug>

The wrapper runs debug with the real arguments and sends back <debug-result> with the string
it returned; you may then answer with <run>. Debug may run at most once per round, and the
loop is bounded.

Inside generated code, `agent_ctx` (alias `emit`) is available:

    agent_ctx.emit("a note for the next round")   # one-way, never triggers a turn
    agent_ctx.note("key", "value")

`print()` is NOT captured — use `agent_ctx.emit`.

Rules:
- Do not write files, run shell commands, or use tools. Only the code blocks are executed.
- The signature must match exactly; names, kinds and defaults are checked.
- One <run> answers one call. Do not build a general-purpose library.
- Imports and module-level names from the source below are available.

Module file: `{location}`
{context_block}
<module-source>
```python
{source}
```
</module-source>"""


def _module_source(module: ModuleType) -> str:
    file = getattr(module, "__file__", None)
    if file:
        linecache.checkcache(file)
    try:
        source = inspect.getsource(module)
    except Exception:
        source = ""
    if not source and file and Path(file).is_file():
        with contextlib.suppress(Exception):
            source = Path(file).read_text(encoding="utf-8")
    return source or f"# source unavailable for {module.__name__}"


def _module_source_hash(module: ModuleType) -> str:
    source = _module_source(module)
    if not source:
        return "nosource"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def module_title(module: ModuleType) -> str:
    """A readable dsh session title — never the compiler's modulename."""
    meta = getattr(module, "__agent__", None)
    name = ""
    description = ""
    if isinstance(meta, dict):
        name = str(meta.get("name") or "")
        description = str(meta.get("description") or "")
    if not name:
        doc = (inspect.getdoc(module) or "").strip()
        name = doc.splitlines()[0] if doc else module.__name__
    title = f"{module.__name__} · {name}"
    if description:
        title = f"{title} — {description}"
    return title.replace("\n", " ")[:_TITLE_CAP]


async def render_module_context(namespace: ModuleType) -> str:
    """Run ``__context__`` inside the module copy; failures degrade to one line."""
    fn = namespace.__dict__.get("__context__")
    if fn is None or not callable(fn):
        return ""
    try:
        value = fn()
        if inspect.isawaitable(value):
            value = await value
        return value if isinstance(value, str) else str(value)
    except Exception as exc:
        logger.exception("agent_func: %s.__context__ failed", namespace.__name__)
        return f"__context__ failed: {type(exc).__name__}: {exc}"


# ── compilation ─────────────────────────────────────────────────────────────


def build_module_namespace(module: ModuleType) -> ModuleType:
    """A copy of the module namespace — generated code never touches the real module."""
    namespace = Compiler(
        source="",
        origin=module,
        modulename=module.__name__,
        filename=f"<agent_func:{module.__name__}>",
    ).compiled
    namespace.__dict__["agent_ctx"] = agent_ctx
    namespace.__dict__["emit"] = agent_ctx.emit
    return namespace


def compile_generated(
    namespace: ModuleType,
    code: str,
    name: str,
    expected_signature: inspect.Signature | None = None,
) -> Callable[..., Any]:
    """Compile one generated block and return the named async callable."""
    try:
        compiled = compile(code, f"<agent_func:{namespace.__name__}>", "exec")
        exec(compiled, namespace.__dict__)  # noqa: S102 — executing the answer is the contract
    except (SyntaxError, IndentationError) as exc:
        raise AgentFuncProtocolError(f"generated code does not compile: {exc}") from exc
    fn = namespace.__dict__.get(name)
    if name == "debug" and fn is None:
        fn = namespace.__dict__.get("__debug__")
    if fn is None:
        raise AgentFuncProtocolError(f"generated code did not define {name}")
    if not inspect.iscoroutinefunction(fn):
        raise AgentFuncProtocolError(f"generated {name} is not an async def")
    if expected_signature is not None:
        actual = inspect.signature(fn)
        if _signature_shape(actual) != _signature_shape(expected_signature):
            raise AgentFuncProtocolError(
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


def _safe_repr(value: Any) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > _ARG_REPR_CAP:
        return text[:_ARG_REPR_CAP] + f"... ({len(text)} chars)"
    return text


# ── session identity ────────────────────────────────────────────────────────


def resolve_session_key(
    config: AgentFuncConfig,
    module_name: str,
    source_hash: str,
) -> str:
    """The logical session key for a config; empty means an explicit session_id."""
    if config.session_id:
        return ""
    if config.mode is AgentMode.SHARED:
        return f"shared:{config.session_key}:{config.agent_preset}"
    if config.mode is AgentMode.PER_MODULE:
        return f"module:{module_name}:{source_hash}:{config.agent_preset}"
    return f"call:{uuid.uuid4().hex}"


@dataclass
class _SessionLease:
    session: DshSession
    key: str
    created: bool
    owned: bool


# ── runtime ─────────────────────────────────────────────────────────────────


@dataclass
class _ModuleState:
    module: ModuleType
    lock: asyncio.Lock
    source_hash: str = ""
    source: str = ""
    context_text: str = ""
    namespace: ModuleType | None = None
    sessions: dict[str, DshSession] = field(default_factory=dict)
    explicit_session: DshSession | None = None
    turns: int = 0
    pending_notes: list[str] = field(default_factory=list)

    def refresh(self) -> bool:
        """Re-read the source; True when the module changed and the namespace must be rebuilt."""
        source = _module_source(self.module)
        digest = (
            hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
            if source
            else "nosource"
        )
        if digest == self.source_hash:
            return False
        self.source = source
        self.source_hash = digest
        self.namespace = None
        self.context_text = ""
        return True


_RUNTIMES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AgentRuntime] = (
    weakref.WeakKeyDictionary()
)
_RUNTIMES_LOCK = threading.Lock()


def _default_url_file() -> Path:
    env = Environment.discover(bootstrap=True)
    return env.workspace_path / _CACHE_RELATIVE / _URL_FILENAME


def _default_cache_root() -> Path:
    env = Environment.discover(bootstrap=True)
    return env.workspace_path / _CACHE_RELATIVE / "agent_func"


def _warn_unclosed() -> None:
    logger.warning("agent_func: a runtime was not closed; its sessions may leak")


class AgentRuntime:
    """One dsh connection, one session map per module, owned by one event loop."""

    def __init__(self) -> None:
        self._connection: DshConnection | None = None
        self._owns_connection = False
        self._connection_lock = asyncio.Lock()
        self._modules: dict[int, _ModuleState] = {}
        self._project_path = Environment.discover(bootstrap=True).project_path
        self._url_file = _default_url_file()
        self._cache_root = _default_cache_root()
        self._closed = False
        self._finalizer = weakref.finalize(self, _warn_unclosed)

    @property
    def closed(self) -> bool:
        return self._closed

    # -- connection ---------------------------------------------------- #

    async def connection(self) -> DshConnection:
        if self._closed:
            raise AgentFuncError("agent_func runtime is closed")
        current = self._connection
        if current is not None and current.is_running():
            return current
        async with self._connection_lock:
            current = self._connection
            if current is not None and current.is_running():
                return current
            borrowed = self._borrow_connection()
            if borrowed is not None and borrowed.is_running():
                self._connection = borrowed
                self._owns_connection = False
                return borrowed
            config = self._read_config()
            connection = DshConnection(config)
            try:
                await connection.__aenter__()
            except BaseException as exc:
                with contextlib.suppress(Exception):
                    await connection.__aexit__(type(exc), exc, exc.__traceback__)
                raise AgentFuncUnavailableError(
                    f"agent_func: cannot connect to dsh at {config.base_url}: {exc}"
                ) from exc
            self._connection = connection
            self._owns_connection = True
            logger.info("agent_func: connected to dsh %s", config.base_url)
            return connection

    def _borrow_connection(self) -> DshConnection | None:
        """The ghost's own launcher, when this process is the ghost."""
        try:
            project = Project.discover(Environment.discover(bootstrap=True))
            # bootstrap before touching the container: get() on a fresh project warns.
            project.bootstrap()
            return project.container.get(DshConnection)
        except Exception:
            return None

    def _read_config(self) -> DshConnectionConfig:
        if not self._url_file.is_file():
            raise AgentFuncUnavailableError(
                f"agent_func: no dsh url at {self._url_file}; start the ghost's dsh first"
            )
        url = self._url_file.read_text(encoding="utf-8").strip()
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url)
        if not parsed.hostname:
            raise AgentFuncUnavailableError(f"agent_func: invalid dsh url {url!r}")
        token = parse_qs(parsed.query).get("token", [None])[0]
        return DshConnectionConfig(host=parsed.hostname, port=parsed.port or 3083, token=token)

    # -- module state -------------------------------------------------- #

    def module_state(self, module: ModuleType) -> _ModuleState:
        key = id(module)
        state = self._modules.get(key)
        if state is None:
            state = _ModuleState(module=module, lock=asyncio.Lock())
            self._modules[key] = state
        return state

    # -- sessions ------------------------------------------------------ #

    async def lease(self, state: _ModuleState, config: AgentFuncConfig) -> _SessionLease:
        if config.session_id:
            session = state.explicit_session
            created = session is None or not session.is_running()
            if created:
                session = await self._attach(config.session_id)
                state.explicit_session = session
                state.turns = 0
            return _SessionLease(
                session=session, key=config.session_id, created=created, owned=False
            )

        key = resolve_session_key(config, state.module.__name__, state.source_hash)
        if config.mode is AgentMode.PER_CALL:
            session = await self._create(state, config)
            state.turns = 0
            return _SessionLease(session=session, key=key, created=True, owned=True)

        session = state.sessions.get(key)
        if session is not None and session.is_running():
            return _SessionLease(session=session, key=key, created=False, owned=False)
        session = await self._create(state, config)
        state.sessions[key] = session
        state.turns = 0
        return _SessionLease(session=session, key=key, created=True, owned=False)

    async def _attach(self, session_id: str) -> DshSession:
        connection = await self.connection()
        session = connection.attach_session(session_id, follow=True)
        await session.__aenter__()
        await session.wait_follow_snapshot(timeout=10.0)
        return session

    async def _create(self, state: _ModuleState, config: AgentFuncConfig) -> DshSession:
        connection = await self.connection()
        if config.agent_preset:
            await self._validate_preset(connection, config.agent_preset)
        cwd = str(Path(config.cwd).expanduser()) if config.cwd else str(self._project_path)
        value = await connection.client.session_create(
            SessionCreateParams(cwd=cwd, agentPreset=config.agent_preset or None)
        )
        session = connection.attach_session(str(value.sessionId), follow=True)
        await session.__aenter__()
        await session.wait_follow_snapshot(timeout=10.0)
        with contextlib.suppress(Exception):
            await session.set_policy(sandbox="read-only", approval="never")
        if config.model or config.effort:
            await self._apply_model(session, config.model, config.effort)
        with contextlib.suppress(Exception):
            await session.rename(title=module_title(state.module))
        logger.info(
            "agent_func: opened session %s for %s", session.session_id, state.module.__name__
        )
        return session

    async def _validate_preset(self, connection: DshConnection, preset: str) -> None:
        if preset == "ego":
            raise AgentFuncError("agent_func: the ego preset cannot be started as a module agent")
        try:
            roster = await connection.client.agent_presets()
        except Exception as exc:
            logger.warning("agent_func: cannot read presets: %s", exc)
            return
        row = next((item for item in roster.presets if item.id == preset), None)
        if row is None:
            known = ", ".join(item.id for item in roster.presets) or "(none)"
            raise AgentFuncError(f"agent_func: unknown preset {preset!r}; available: {known}")
        if row.broken:
            raise AgentFuncError(f"agent_func: preset {preset!r} is broken: {row.broken}")

    async def _apply_model(self, session: DshSession, model: str, effort: str) -> None:
        if not model:
            if effort:
                selection = await session.model_selection()
                await session.select_model(
                    provider=selection.provider,
                    model=selection.model,
                    reasoning_effort=effort,
                )
            return
        provider, _, name = model.partition("/")
        if not name:
            raise AgentFuncError(
                f"agent_func: model {model!r} must be 'provider/model' to select it per session"
            )
        await session.select_model(
            provider=provider, model=name, reasoning_effort=effort or None
        )

    async def debug_artifact(self, session_id: str, turn: int, text: str) -> Path:
        path = self._cache_root / session_id / f"debug-{turn}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    # -- lifecycle ----------------------------------------------------- #

    async def aclose(self) -> None:
        """Close sessions first, then the connection; archive what this runtime created."""
        if self._closed:
            return
        self._closed = True
        for state in self._modules.values():
            for session in list(state.sessions.values()):
                with contextlib.suppress(Exception):
                    await session.archive_session(stop=True)
                with contextlib.suppress(Exception):
                    await session.close()
            state.sessions.clear()
            explicit = state.explicit_session
            state.explicit_session = None
            if explicit is not None:
                with contextlib.suppress(Exception):
                    await explicit.close()
        self._modules.clear()
        connection = self._connection
        owns = self._owns_connection
        self._connection = None
        if owns and connection is not None:
            with contextlib.suppress(Exception):
                await connection.__aexit__(None, None, None)
        self._finalizer.detach()
        logger.info("agent_func: runtime closed")


def _runtime() -> AgentRuntime:
    loop = asyncio.get_running_loop()
    with _RUNTIMES_LOCK:
        runtime = _RUNTIMES.get(loop)
        if runtime is None or runtime.closed:
            runtime = AgentRuntime()
            _RUNTIMES[loop] = runtime
        return runtime


async def close_agent_func_runtime() -> None:
    """Close this loop's runtime; called by the ghost runtime before its dsh exits."""
    loop = asyncio.get_running_loop()
    with _RUNTIMES_LOCK:
        runtime = _RUNTIMES.pop(loop, None)
    if runtime is not None:
        await runtime.aclose()


# ── the decorator ───────────────────────────────────────────────────────────


def agent_func(
    func: _F | None = None,
    *,
    agent_preset: str = "",
    model: str = "",
    effort: str = "",
    cwd: str | None = None,
    mode: AgentMode | str = AgentMode.PER_MODULE,
    session_id: str = "",
    session_key: str = "",
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    max_debug_rounds: int = _DEFAULT_MAX_DEBUG_ROUNDS,
    timeout: float = _DEFAULT_TIMEOUT,
    debug_result_cap: int = _DEFAULT_DEBUG_CAP,
    archive_on_close: bool = True,
):
    """Declare a function whose implementation is settled by a module agent per call."""
    config = AgentFuncConfig(
        agent_preset=agent_preset,
        model=model,
        effort=effort,
        cwd=cwd,
        mode=AgentMode(mode),
        session_id=session_id,
        session_key=session_key,
        max_attempts=max_attempts,
        max_debug_rounds=max_debug_rounds,
        timeout=timeout,
        debug_result_cap=debug_result_cap,
        archive_on_close=archive_on_close,
    )

    def decorate(fn: _F) -> _F:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError("@agent_func requires an async def function")
        module = inspect.getmodule(fn)
        if module is None:
            raise TypeError(f"@agent_func cannot find the module of {fn.__qualname__}")
        if "<locals>" in fn.__qualname__:
            raise TypeError(
                f"@agent_func supports module-level functions only, got {fn.__qualname__}"
            )
        expected = inspect.signature(fn)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            active = _ACTIVE_MODULES.get()
            key = id(module)
            if key in active:
                raise AgentFuncReentryError(
                    f"agent_func: {module.__name__}.{fn.__qualname__} re-entered its own module"
                )
            token = _ACTIVE_MODULES.set(active | {key})
            outer = _CURRENT_CTX.get()
            effective = _merge_config(config, outer.overrides if outer is not None else {})
            ctx = AgentContext(
                module=module.__name__,
                function=fn.__qualname__,
            )
            ctx_token = _CURRENT_CTX.set(ctx)
            record = AgentCallRecord(
                module=module.__name__,
                function=fn.__qualname__,
                mode=effective.mode.value,
            )
            started = time.time()
            try:
                runtime = _runtime()
                state = runtime.module_state(module)
                async with state.lock:
                    return await _call(
                        runtime, state, fn, expected, effective, ctx, record, args, kwargs
                    )
            except BaseException as exc:
                record.status = record.status or "error"
                record.error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record.duration = time.time() - started
                _RECORDS.append(record)
                _CURRENT_CTX.reset(ctx_token)
                _ACTIVE_MODULES.reset(token)

        wrapper.__agent_func__ = True  # type: ignore[attr-defined]
        wrapper.__agent_func_config__ = config  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorate(func)
    return decorate


def _merge_config(config: AgentFuncConfig, overrides: dict[str, Any]) -> AgentFuncConfig:
    allowed = {
        "agent_preset",
        "model",
        "effort",
        "cwd",
        "mode",
        "session_id",
        "session_key",
        "max_attempts",
        "max_debug_rounds",
        "timeout",
        "debug_result_cap",
    }
    values = {key: value for key, value in overrides.items() if key in allowed}
    if "mode" in values:
        values["mode"] = AgentMode(values["mode"])
    return replace(config, **values) if values else config


async def _call(
    runtime: AgentRuntime,
    state: _ModuleState,
    fn: Callable[..., Any],
    expected: inspect.Signature,
    config: AgentFuncConfig,
    ctx: AgentContext,
    record: AgentCallRecord,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    changed = state.refresh()
    if changed:
        state.namespace = None
        state.context_text = ""
        state.pending_notes.clear()
    namespace = state.namespace
    if namespace is None:
        namespace = build_module_namespace(state.module)
        state.namespace = namespace
        state.context_text = await render_module_context(namespace)
    lease = await runtime.lease(state, config)
    ctx.session_id = lease.session.session_id
    record.session_id = ctx.session_id
    record.source_hash = state.source_hash
    try:
        result = await _answer(
            runtime, state, lease, namespace, fn, expected, config, ctx, record, args, kwargs
        )
        record.status = "ok"
        return result
    finally:
        leftover = ctx._drain_notes()
        if leftover:
            state.pending_notes.extend(leftover)
            record.notes = list(leftover)
        if lease.owned and config.mode is AgentMode.PER_CALL:
            if config.archive_on_close:
                with contextlib.suppress(Exception):
                    await lease.session.archive_session(stop=True)
            with contextlib.suppress(Exception):
                await lease.session.close()


async def _answer(
    runtime: AgentRuntime,
    state: _ModuleState,
    lease: _SessionLease,
    namespace: ModuleType,
    fn: Callable[..., Any],
    expected: inspect.Signature,
    config: AgentFuncConfig,
    ctx: AgentContext,
    record: AgentCallRecord,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    first = state.turns == 0 or lease.created
    call_message = _call_message(state.module, fn, expected, args, kwargs, state.pending_notes)
    if first:
        instruction = build_instruction(
            state.module, source=state.source, context=state.context_text
        )
        message = f"{instruction}\n\n{call_message}"
    else:
        message = call_message
    state.pending_notes.clear()
    run = await _run_turn(lease.session, message, config.timeout)
    state.turns += 1
    attempt = 0
    debug_rounds = 0
    while True:
        text = run.final_response or ""
        parsed = parse_answer(text)
        if parsed.run is not None:
            try:
                implementation = compile_generated(namespace, parsed.run, "__run__", expected)
            except AgentFuncProtocolError as exc:
                attempt += 1
                record.attempts = attempt
                if attempt >= config.max_attempts:
                    raise
                run = await _run_turn(
                    lease.session, _protocol_retry(exc, text, ctx), config.timeout
                )
                state.turns += 1
                continue
            record.generated_code = parsed.run[:_DEFAULT_DEBUG_CAP]
            try:
                return await implementation(*args, **kwargs)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await lease.session.cancel()
                raise
            except Exception as exc:
                if not _is_generation_error(exc):
                    raise
                attempt += 1
                record.attempts = attempt
                if attempt >= config.max_attempts:
                    raise
                run = await _run_turn(
                    lease.session, _execution_retry(exc, ctx), config.timeout
                )
                state.turns += 1
                continue
        if parsed.debug is not None and debug_rounds < config.max_debug_rounds:
            debug_rounds += 1
            record.debug_rounds = debug_rounds
            result = await _run_debug(namespace, parsed.debug, args, kwargs)
            result = await _render_debug_result(
                runtime, lease.session.session_id, state.turns, result, config.debug_result_cap
            )
            run = await _run_turn(
                lease.session,
                _debug_message(result, ctx),
                config.timeout,
            )
            state.turns += 1
            continue
        attempt += 1
        record.attempts = attempt
        if attempt >= config.max_attempts:
            raise AgentFuncProtocolError(
                f"agent_func: {fn.__qualname__} produced no <run> block; last answer: "
                f"{text[:800]!r}"
            )
        run = await _run_turn(
            lease.session, _protocol_retry("no <run> or <debug> block found", text, ctx), config.timeout
        )
        state.turns += 1


async def _run_turn(session: DshSession, message: str, timeout: float):
    try:
        return await session.run(message, timeout=timeout)
    except asyncio.TimeoutError as exc:
        with contextlib.suppress(Exception):
            await session.cancel()
        raise AgentFuncTimeoutError(
            f"agent_func: dsh turn timed out after {timeout}s"
        ) from exc


def _call_message(
    module: ModuleType,
    fn: Callable[..., Any],
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    notes: list[str],
) -> str:
    note_block = ""
    if notes:
        note_block = "\n<notes>\n" + "\n".join(notes) + "\n</notes>\n"
    return f"""<call module="{module.__name__}" function="{fn.__qualname__}">
signature: {signature}
args: {_safe_repr(args)}
kwargs: {_safe_repr(kwargs)}
docstring: {inspect.getdoc(fn) or ""}
</call>{note_block}"""


def _protocol_retry(exc: object, previous: str, ctx: AgentContext) -> str:
    notes = ctx._drain_notes()
    note_block = ("\n<notes>\n" + "\n".join(notes) + "\n</notes>\n") if notes else ""
    return (
        "<protocol-error>\n"
        f"{exc}\n"
        "</protocol-error>\n"
        "Your previous answer was:\n"
        f"{previous[:1200]}\n"
        "Answer with exactly one <run> block (or one <debug> block)."
        f"{note_block}"
    )


def _execution_retry(exc: BaseException, ctx: AgentContext) -> str:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if len(text) > _DEFAULT_DEBUG_CAP:
        text = text[-_DEFAULT_DEBUG_CAP:]
    notes = ctx._drain_notes()
    note_block = ("\n<notes>\n" + "\n".join(notes) + "\n</notes>\n") if notes else ""
    return (
        "<run-error>\n"
        f"{text}\n"
        "</run-error>\n"
        "The implementation raised. Fix it and answer with a new <run> block."
        f"{note_block}"
    )


def _debug_message(result: str, ctx: AgentContext) -> str:
    notes = ctx._drain_notes()
    note_block = ("\n<notes>\n" + "\n".join(notes) + "\n</notes>\n") if notes else ""
    return f"<debug-result>\n{result}\n</debug-result>{note_block}"


async def _run_debug(
    namespace: ModuleType,
    code: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    try:
        debug = compile_generated(namespace, code, "debug")
    except AgentFuncProtocolError as exc:
        return f"debug block rejected: {exc}"
    try:
        value = await debug(*args, **kwargs)
    except Exception as exc:
        return f"debug raised {type(exc).__name__}: {exc}"
    return value if isinstance(value, str) else _safe_repr(value)


async def _render_debug_result(
    runtime: AgentRuntime,
    session_id: str,
    turn: int,
    text: str,
    cap: int,
) -> str:
    if len(text) <= cap:
        return text
    path = await runtime.debug_artifact(session_id, turn, text)
    head = text[: cap // 2]
    tail = text[-(cap // 2):]
    return (
        f"{head}\n... ({len(text)} chars; full result at {path}) ...\n{tail}\n"
        f"Use your read-only file tools to read the full result."
    )


_GENERATION_ERRORS = (
    SyntaxError,
    IndentationError,
    NameError,
    UnboundLocalError,
    ImportError,
    ModuleNotFoundError,
    AttributeError,
    TypeError,
)


def _is_generation_error(exc: BaseException) -> bool:
    return isinstance(exc, _GENERATION_ERRORS)
