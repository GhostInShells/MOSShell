"""
Module-level sandbox for safe, stateful Python code execution.

Motivation
----------
Existing Compiler creates a ModuleType, compiles source into it once, and
discards the container. It has no builtins control, no lifecycle hooks, and
no way to share variables between related execution contexts.

Sandbox answers a different need: give an AI model a persistent Python REPL
where it can write arbitrary code across multiple turns, with dangerous
builtins blocked by default, and with the ability to create child sandboxes
that share the parent's namespace (injected domain objects propagate both ways).

This is the foundation for ModuleEvalChannel — the "thin shell" pattern where
an AI controls a domain object (Playwright browser, pandas DataFrame, ROS node)
by writing raw Python code rather than calling pre-wrapped command functions.

Security boundary (read before trusting this with untrusted code)
-----------------------------------------------------------------
Sandbox is NOT a security boundary. Blocking ``__import__``/``open``/``eval``
is a guardrail against accidental, naive misuse — a model reflexively writing
``import os`` or ``open()`` gets a clear failure. It does not stop an adversary.

Escape needs none of the blocked builtins. ``object`` is always in scope, so::

    ().__class__.__base__.__subclasses__()   # every class in the process
    # -> any Python-defined class -> __init__.__globals__['sys'] -> sys.modules
    # -> os / subprocess / open / ... -> arbitrary host code

Any Python function/class reachable from the namespace (including injected
domain objects) leaks its defining module's ``__globals__``, which reference
the host's real builtins. No builtin list can close this — the object protocol
is orthogonal to it.

Real isolation lives outside the Sandbox:

- a separate process / container with a narrow protocol (see tools._eval_server)
- narrowing the reachable surface, not the builtins: record imports at compile
  time and replay only those at exec time (see agents._imports.replay_import)

Trust model: the writer is non-adversarial. Sandbox raises friction on
accidental dangerous builtin use; it is not a jail.

Relationship to Compiler & Executor
------------------------------------
Compiler: one-shot ModuleType creation + source compilation. No persistence.
Executor: wraps Compiler with stdout capture and function calling. Still one-shot.
Sandbox:  persistent ModuleType namespace. Cumulative exec(). Builtins control.
          Parent-child __dict__ sharing. Lifecycle hooks. Self-introspection.

All three share the same ExecutionResult type. They form a spectrum:
  Compiler → Executor → Sandbox
  (stateless) → (one-shot capture) → (stateful REPL)

Usage
-----
Basic::

    s = Sandbox()
    s.exec("x = 1")
    s.exec("y = x + 2")
    r = s.exec("__result__ = y")  # r.returns == 3

With domain object injection (the ModuleEvalChannel pattern)::

    parent = Sandbox()
    parent.set("sqlite3", sqlite3)
    parent.set("conn", sqlite3.connect(":memory:"))

    child = Sandbox(parent=parent)
    # AI writes code in child; side effects visible in parent
    child.exec("conn.execute('CREATE TABLE t (id INT)')")

Builtins control::

    s = Sandbox()                     # default safe set (no __import__, open, ...)
    s = Sandbox(builtins=None)        # unrestricted — full Python builtins
    s = Sandbox(builtins={...})       # custom whitelist

Introspection (for channel api() command)::

    s.get_interface()                 # list all names with types
    s.get_interface("conn")           # detail on a specific name
    s.get_source("my_func")           # source of a sandbox-defined function

Lifecycle::

    def init(sb): sb.set("db", connect_db())
    def destroy(sb): sb.get("db").close()

    with Sandbox(on_init=init, on_destroy=destroy) as sb:
        sb.exec("db.execute(...)")
    # destroy() called automatically on __exit__
"""

import builtins as _builtins
import inspect as _inspect
import sys as _sys
import traceback as _traceback
from contextlib import redirect_stdout
from io import StringIO
from types import ModuleType
from typing import Any, Callable

from .executor import ExecutionResult

__all__ = ['SANDBOX_BUILTINS', 'Sandbox']

# Builtins that are blocked in the default safe configuration.
# __import__ is the most critical — it gates access to the entire stdlib
# and filesystem. Without it, code inside the sandbox cannot import new modules
# (callers pre-import what's needed and inject via on_init or set()).
_DANGEROUS_BUILTINS = frozenset({
    '__import__',   # gates all module imports
    'open',          # filesystem access
    'eval',          # arbitrary code execution from strings
    'exec',          # arbitrary code execution from strings
    'compile',       # bytecode compilation
    'input',         # blocking stdin read
    'breakpoint',    # debugger drop-in
})

# A copy of Python's builtins with dangerous functions removed.
# Safe functions (print, len, range, isinstance, Exception, etc.) are all present.
# This constant is read-only — never mutated. Sandbox.__init__ copies from it
# into the module's __builtins__ rather than referencing it directly.
SANDBOX_BUILTINS: dict[str, Any] = {
    k: v for k, v in _builtins.__dict__.items()
    if k not in _DANGEROUS_BUILTINS
}

# Sentinel for "caller didn't pass a builtins value".  Using a sentinel
# avoids the mutable-default-argument anti-pattern (passing the SANDBOX_BUILTINS
# dict literal as a default would risk silent mutation of the global constant).
_UNSET = object()


class Sandbox:
    """Safe, stateful execution environment backed by a ModuleType namespace.

    Think of it as a Python REPL with controlled builtins and lifecycle.
    Each exec() call is cumulative — variables, functions, and classes defined
    in one call are visible in the next. This is the key property that makes it
    suitable for AI-driven incremental code writing.

    Parent-Child Sharing
    --------------------
    When a child Sandbox is created with ``parent=``, the child's
    ``module.__dict__`` IS the parent's ``module.__dict__`` — same reference,
    not a copy. This means:

    - Variables set in the child are immediately visible in the parent
    - Variables set in the parent are immediately visible in the child
    - child.close() does NOT clear the namespace (it belongs to the parent)
    - parent.close() cascades to all children, then clears the namespace

    This design serves the ModuleEvalChannel pattern: the parent holds domain
    objects (Playwright page, sqlite3 connection, pandas DataFrame) that the
    child (the AI's execution context) reads and mutates. When the child is
    done, the parent still has all accumulated state.

    Thread Safety
    -------------
    Sandbox is NOT thread-safe. It's designed for single-threaded use within
    a Channel's execution context. For cross-thread code execution (e.g.,
    Playwright's requirement to run on the main thread), use a Janus bridge
    (queue.Queue + threading.Event) in the layer above — see ModuleEvalChannel.
    """

    def __init__(
        self,
        name: str = "__sandbox__",
        *,
        parent: "Sandbox | None" = None,
        builtins: dict[str, Any] | None = _UNSET,
        on_init: Callable[["Sandbox"], None] | None = None,
        on_destroy: Callable[["Sandbox"], None] | None = None,
        source: str = '',
    ):
        """
        Parameters
        ----------
        name:
            Logical name for the sandbox. Used in error messages and as the
            module's ``__name__``. Not required to be unique.
        parent:
            If set, this sandbox shares the parent's ``module.__dict__``.
            Side effects propagate both ways. The parent must not be closed.
        builtins:
            Dict to use as the module's ``__builtins__``. When not provided,
            a child inherits the parent's builtins; a root sandbox uses
            SANDBOX_BUILTINS (safe set: no __import__, open, eval, etc.).
            Pass a custom dict to further restrict. Pass ``None`` for
            unrestricted access (full Python builtins). A child that
            explicitly sets this overrides the shared __dict__'s builtins
            (affecting the parent too — they share the namespace).
        on_init:
            Called immediately after the module is created, before any exec().
            Use this to inject domain objects via ``sandbox.set()`` — e.g.,
            database connections, Playwright page, API clients.
        on_destroy:
            Called during close(), after children are closed but before the
            namespace is cleared. Use this to release external resources.
        source:
            Module source code.  Stored for reflection — get_interface()
            delegates to Reflector which shows source + imported attrs.
            Default ``''``: no source, Reflector output is attr-blocks only.
        """
        self._name = name
        self._parent = parent
        self._on_destroy = on_destroy
        self._children: set["Sandbox"] = set()
        self._closed = False

        # Resolve builtins default: inherit from parent if unset, otherwise
        # use the safe default for root sandboxes.
        if builtins is _UNSET:
            if parent is not None:
                builtins = parent.module.__dict__.get('__builtins__')
            else:
                builtins = SANDBOX_BUILTINS

        # Snapshot into an owned dict. Never hand exec code a reference to the
        # shared SANDBOX_BUILTINS constant (or the process builtins dict in the
        # builtins=None case): a sandbox mutating `__builtins__` (e.g.
        # __builtins__['x'] = ... / __builtins__.pop('len')) must not poison
        # other sandboxes or the module constant.
        if builtins is not None:
            builtins = dict(builtins)

        if parent is not None:
            if parent._closed:
                raise ValueError("parent sandbox is closed")
            # Share the exact __dict__ reference — not a copy.
            # This is the core of the parent-child design: mutations in child
            # are instantly visible in parent, and vice versa.
            self._module = parent._module
            # If the caller explicitly passed a builtins value (not _UNSET),
            # apply it to the shared namespace. This intentionally affects
            # the parent — they share the same __dict__.
            if builtins is not None:
                self._module.__builtins__ = builtins
            parent._children.add(self)
        else:
            self._module = ModuleType(name)
            # Set __builtins__ on the module BEFORE any code runs.
            # CPython's exec() checks globals['__builtins__'] first;
            # if absent, it inserts builtins.__dict__. By setting it here
            # we preempt that default and control what builtins are available.
            # The unrestricted case snapshots the process builtins dict so exec
            # code mutating __builtins__ cannot corrupt the host.
            self._module.__builtins__ = (
                builtins if builtins is not None else dict(_builtins.__dict__)
            )

        if on_init:
            on_init(self)

        from .reflector import Reflector
        self._reflector = Reflector(self._module, modulename=name, source=source)

    # -- public read-only properties -------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def module(self) -> ModuleType:
        """The underlying ModuleType whose __dict__ is the execution namespace.

        Exposed so callers can inspect the full namespace (module.__dict__)
        or pass the module to Compiler/Executor for one-shot operations
        within the same persistent namespace.
        """
        return self._module

    # -- core API --------------------------------------------------------

    def exec(self, code: str) -> ExecutionResult:
        """Execute *code* in the sandbox namespace and capture stdout.

        This is the primary API. Each call compiles and executes the given
        code string in the sandbox's persistent namespace. Variables, imports,
        functions, and classes defined in one call are visible in all subsequent
        calls — exactly like a Python REPL.

        To return a value, assign to ``__result__`` in the code. This convention
        is shared with Executor and keeps the return channel simple: the model
        writes ``__result__ = expr`` and the caller reads ``result.returns``.

        All output to stdout (print, tracebacks, etc.) is captured and returned
        as ``result.std_output`` — the code cannot write directly to the real
        stdout of the host process.

        If *code* raises an exception, it is caught and returned in
        ``result.exception`` and ``result.traceback`` — never propagated.
        The traceback is filtered to show only frames from the sandboxed code
        and any libraries it calls, omitting sandbox.py internals (cognitive
        isolation — the model sees only its own call stack, not ours).

        The only case where exec() raises is when the sandbox itself is closed
        — a usage error, not a code error.
        """
        if self._closed:
            raise RuntimeError(f"sandbox {self._name!r} is closed")

        result = ExecutionResult()
        buffer = StringIO()
        try:
            with redirect_stdout(buffer):
                exec(compile(code, self._name, 'exec'), self._module.__dict__)
                result.returns = self._module.__dict__.get('__result__', None)
        except Exception:
            result.exception = _traceback.format_exception_only(
                *_sys.exc_info()[:2]
            )[-1].strip()
            result.traceback = self._filter_traceback()
        result.std_output = buffer.getvalue()
        return result

    def get(self, name: str) -> Any:
        """Read a variable from the sandbox namespace by name.

        Raises AttributeError if the name doesn't exist. For bulk inspection
        of the namespace, use ``sandbox.module.__dict__`` directly, or call
        ``get_interface()`` for a human-readable summary.
        """
        if name not in self._module.__dict__:
            raise AttributeError(f"{self._name!r} has no attribute {name!r}")
        return self._module.__dict__[name]

    def set(self, name: str, value: Any) -> None:
        """Write a variable into the sandbox namespace.

        This is how domain objects are injected before the AI starts writing
        code — database connections, API clients, Playwright pages, etc.
        The AI's exec() calls then reference these objects by the injected name.
        """
        self._module.__dict__[name] = value

    # -- introspection (for channel api() / vars() commands) -------------
    #
    # All reflection delegates to the codex Reflector, which follows the
    # same pipeline as "moss codex get-interface":
    #   - source (module_source) shown as-is
    #   - imported objects → standard reflection via reflect_imported_attr
    #   - local (exec-defined) objects → inspect-based fallback
    #
    # This keeps sandbox consistent with how every other module is reflected.

    def get_interface(self, name: str | None = None) -> str:
        """Reflect on the sandbox namespace.

        Without *name*, returns the full Reflector output: module source +
        <attr> blocks for imported objects.  Local (exec-defined) objects
        are already visible in the source and are not duplicated.

        With *name*, returns detail for a single object: full source for
        imports, signature + docstring + public methods for locals.

        Designed to back the ``api()`` / ``vars()`` commands in
        ModuleEvalChannel.
        """
        if name is not None:
            ns = self._module.__dict__
            if name not in ns:
                return f"'{name}' is not defined in sandbox {self._name!r}"
            return self._reflect_single(name, ns[name])

        return self._reflector.reflect()

    def get_source(self, name: str) -> str:
        """Return the source code of *name* if available.

        Works for imported objects (real modules with source files).
        For local (exec-defined) objects, raises ValueError with a pointer
        to the module source / instruction.
        """
        if name not in self._module.__dict__:
            raise AttributeError(f"{self._name!r} has no attribute {name!r}")
        obj = self._module.__dict__[name]
        try:
            return _inspect.getsource(obj)
        except (TypeError, OSError):
            if self._is_local(obj) and self._reflector.source:
                raise ValueError(
                    f"'{name}' is defined in the sandbox source — "
                    f"see the module source / instruction for its definition"
                ) from None
            raise ValueError(f"source not available for {name!r}") from None

    # -- traceback filtering (cognitive isolation) -----------------------
    #
    # The sandbox is a runtime for model-written code, like CPython is a
    # runtime for user Python.  When the model's code raises, the traceback
    # should not expose our exec() frame — just as Python tracebacks don't
    # expose C-level eval frames.  We keep only frames whose filename is
    # NOT this file (sandbox.py).

    def _filter_traceback(self) -> str:
        """Return the current exception's traceback with sandbox.py frames removed."""
        tb = _sys.exc_info()[2]
        if tb is None:
            return ''
        frames = _traceback.extract_tb(tb)
        # Resolve lazily — __file__ is available at import time.
        our_file = _inspect.getfile(type(self))
        relevant = [f for f in frames if f.filename != our_file]
        if not relevant:
            return ''
        return ''.join(_traceback.format_list(relevant))

    # -- internal reflection helpers ------------------------------------

    def _is_local(self, value: Any) -> bool:
        """Check if *value* is local to this sandbox (defined via exec)."""
        from ._utils import get_modulename_of_value
        modname = get_modulename_of_value(value)
        if modname is None:
            return True
        if modname == self._module.__name__:
            return True
        return False

    def _reflect_single(self, name: str, value: Any) -> str:
        """Detail for a single name.  Imported objects go through the standard
        codex pipeline; local objects get inspect-based fallback."""
        from ._reflect import reflect_imported_attr

        prompt = reflect_imported_attr(name, value, self._module.__name__)
        if prompt:
            return prompt

        return self._reflect_local_detail(name, value)

    @staticmethod
    def _reflect_local_detail(name: str, value: Any) -> str:
        """Inspect-based detail for a local (exec-defined) object.

        Does NOT attempt inspect.getsource — local objects have no source
        file.  Uses inspect.signature, inspect.getdoc, and dir() which
        always work.
        """
        lines = []

        if _inspect.isfunction(value) or _inspect.ismethod(value):
            try:
                sig = _inspect.signature(value)
                lines.append(f"def {name}{sig}")
            except (ValueError, TypeError):
                lines.append(f"def {name}(...)")
            doc = _inspect.getdoc(value)
            if doc:
                lines.append(f'    """{doc}"""')
            return '\n'.join(lines)

        if _inspect.isclass(value):
            try:
                sig = _inspect.signature(value)
                lines.append(f"class {name}{sig}")
            except (ValueError, TypeError):
                lines.append(f"class {name}")
            doc = _inspect.getdoc(value)
            if doc:
                lines.append(f'    """{doc}"""')
            public = sorted(
                m for m in dir(value)
                if not m.startswith('_') and callable(getattr(value, m, None))
            )
            if public:
                lines.append(f"    # public methods: {', '.join(public)}")
            return '\n'.join(lines)

        # Instance or plain value — show type + repr + public methods
        typ = type(value)
        try:
            r = repr(value)
            if len(r) > 120:
                r = r[:117] + "..."
            lines.append(f"{name}: {typ.__qualname__} = {r}")
        except Exception:
            lines.append(f"{name}: {typ.__qualname__}")
        public = sorted(
            m for m in dir(value)
            if not m.startswith('_') and callable(getattr(value, m, None))
        )
        interesting = [m for m in public if not m.startswith('__')]
        if interesting:
            lines.append(f"    # public methods: {', '.join(interesting)}")
        return '\n'.join(lines)

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Destroy the sandbox and release resources.

        Sequence:
        1. Mark closed (subsequent exec() calls raise RuntimeError)
        2. Close all children recursively (they share our __dict__)
        3. Call on_destroy hook (release external resources)
        4. If this is a root sandbox (no parent), clear the module __dict__
        5. If this is a child, detach from parent (parent's namespace lives on)

        Idempotent — calling close() multiple times is safe.
        """
        if self._closed:
            return
        self._closed = True

        # Close children first. They share our __dict__, so closing them
        # doesn't clear the namespace — it just runs their on_destroy hooks
        # and marks them closed.
        for child in list(self._children):
            child.close()

        if self._on_destroy:
            self._on_destroy(self)

        if self._parent:
            self._parent._children.discard(self)
        else:
            # Only the root sandbox actually clears the namespace.
            # Children share the parent's __dict__; clearing it would
            # destroy the parent's state.
            self._module.__dict__.clear()

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
        return False

