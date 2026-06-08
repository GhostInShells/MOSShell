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
                builtins = parent._module.__dict__.get('__builtins__')
            else:
                builtins = SANDBOX_BUILTINS

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
            self._module.__builtins__ = (
                builtins if builtins is not None else _builtins.__dict__
            )

        if on_init:
            on_init(self)

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
    # Delegates to the existing codex reflection pipeline:
    #   get_value_self_prompt  — __prompt__ protocol (objects describe themselves)
    #   reflect_prompt_from_value — value-level reflection (classes, functions)
    #   get_callable_definition    — clean function signature + docstring
    #
    # The sandbox namespace is a hybrid: it contains both injected imports
    # and variables defined via exec().  We can't use reflect_module() or
    # reflect_imported_locals_by_modulename() directly because those assume
    # a real module with source code and filter out same-module locals.
    # Instead we call the value-level primitives directly.

    def get_interface(self, name: str | None = None) -> str:
        """Reflect on the sandbox namespace — the model's "what can I use?"

        Delegates to the codex reflection pipeline (__prompt__ protocol →
        reflect_prompt_from_value → get_callable_definition).

        Without *name*, returns a compact listing: one line per public name
        with type info — suitable as a quick orientation.  Functions show
        their def signature (via get_callable_definition).  Classes and
        modules show their qualified name.  Simple values show type + repr.

        With *name*, returns detailed information about that object: source
        code for classes, signature + docstring for functions, type + repr
        for plain values.  Objects that implement the __prompt__ protocol
        describe themselves.

        Designed to back the ``api()`` / ``vars()`` commands in
        ModuleEvalChannel.
        """
        ns = self._module.__dict__

        if name is not None:
            if name not in ns:
                return f"'{name}' is not defined in sandbox {self._name!r}"
            return self._reflect_detail(name, ns[name])

        lines = [f"# sandbox: {self._name}"]
        for key in sorted(ns):
            if key.startswith('_'):
                continue
            lines.append(self._reflect_summary(key, ns[key]))
        return '\n'.join(lines)

    def get_source(self, name: str) -> str:
        """Return the source code of *name* if available.

        Uses inspect.getsource — the same mechanism the Reflector uses for
        classes and functions.  Only works for objects defined in the sandbox
        via exec() or injected objects that have accessible source files.
        Raises ValueError when source is not retrievable.
        """
        if name not in self._module.__dict__:
            raise AttributeError(f"{self._name!r} has no attribute {name!r}")
        obj = self._module.__dict__[name]
        try:
            return _inspect.getsource(obj)
        except (TypeError, OSError):
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

    # -- internal reflection helpers (delegate to codex._reflect) --------

    @staticmethod
    def _reflect_detail(name: str, value: Any) -> str:
        """Detailed description for a single name, using the reflection pipeline."""
        from ._reflect import get_value_self_prompt, reflect_prompt_from_value

        # 1. __prompt__ protocol: object self-describes
        prompt = get_value_self_prompt(value)
        if prompt:
            return prompt

        # 2. Standard reflection: abstract classes, pydantic/dataclass, functions
        prompt = reflect_prompt_from_value(value)
        if prompt:
            return prompt

        # 3. Fallback: type + repr for plain values
        return f"{name}: {type(value).__qualname__} = {value!r}"

    @staticmethod
    def _reflect_summary(key: str, value: Any) -> str:
        """Compact one-line summary for namespace listing.

        Uses get_callable_definition for functions (just the def line),
        basic type hints for everything else.
        """
        from ._utils import get_callable_definition

        if _inspect.isfunction(value) or _inspect.ismethod(value):
            try:
                definition = get_callable_definition(value)
                if definition:
                    # Take only the first line (def signature)
                    return f"  {definition.split(chr(10))[0]}"
            except (TypeError, OSError):
                pass

        if _inspect.isclass(value):
            return f"  {key:<20} class {value.__qualname__}"
        if _inspect.ismodule(value):
            return f"  {key:<20} module ({value.__name__})"

        typ = type(value)
        try:
            return f"  {key:<20} {typ.__qualname__} = {value!r}"
        except Exception:
            return f"  {key:<20} {typ.__qualname__}"

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

