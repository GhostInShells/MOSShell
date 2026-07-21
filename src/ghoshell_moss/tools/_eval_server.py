"""Generic Sandbox eval server — stdin/stdout JSON-line protocol.

Receives MODULE_FILE env var, compiles it, wraps in two-layer Sandbox,
and serves exec requests over JSON-line.

Protocol:
  in  — {"code": "..."}
  out — {"returns": ..., "std_output": ..., "exception": ..., "traceback": ...}

Special codes:
  __SHUTDOWN__  — exit cleanly
  __vars__      — return namespace summary (not via exec, sandbox.__dict__)
  __api__       — return full sandbox.get_interface()
  __api_single__ — return sandbox.get_interface(name), name in request["name"]

Layer design:
  Compiler (builtins unrestricted) — compile module, execute imports
  init_sandbox (builtins=None)       — holds compiled namespace objects
  sandbox (parent=init, SANDBOX_BUILTINS) — AI exec namespace, no __import__
"""

import json as _json
import os
import sys
import traceback as _traceback
from pathlib import Path

from ghoshell_moss.core.codex.compiler import Compiler
from ghoshell_moss.core.codex.sandbox import Sandbox, SANDBOX_BUILTINS

# ── Resolve module file ─────────────────────────────────────────────────

_module_file = os.environ.get("MODULE_FILE")
if not _module_file:
    print(_json.dumps({"error": "MODULE_FILE not set"}), flush=True)
    sys.exit(1)

_module_path = Path(_module_file)
if not _module_path.is_file():
    print(_json.dumps({"error": f"file not found: {_module_file}"}), flush=True)
    sys.exit(1)

_module_name = os.environ.get("MODULE_NAME", _module_path.stem)
_source = _module_path.read_text()

# ── Compile module (full builtins, executes imports/side effects) ──────

try:
    _compiler = Compiler(
        source=_source,
        modulename=_module_name,
        filename=str(_module_path),
        compile_soon=True,
    )
    _compiled = _compiler.compiled
except Exception:
    _json.dump(
        {"error": "module compilation failed", "traceback": _traceback.format_exc()},
        sys.stdout,
    )
    sys.stdout.flush()
    sys.exit(1)

# ── Two-layer sandbox ──────────────────────────────────────────────────

_init_sandbox = Sandbox(
    name=_module_name,
    builtins=None,
    source=_source,
)

# Copy compiled namespace (domain objects, imports) into init_sandbox
for _k, _v in _compiled.__dict__.items():
    if not _k.startswith("__"):
        _init_sandbox.set(_k, _v)

_sandbox = Sandbox(
    name=_module_name,
    parent=_init_sandbox,
    builtins=SANDBOX_BUILTINS,
    source=_source,
)

# ── Eval loop ──────────────────────────────────────────────────────────

print("ready", flush=True)

for _line in sys.stdin:
    _request = _json.loads(_line)

    # -- protocol commands (not exec) --
    _code = _request.get("code", "")

    if _code == "__SHUTDOWN__":
        break

    if _code == "__vars__":
        _ns = {
            k: type(v).__name__
            for k, v in sorted(_sandbox.module.__dict__.items())
            if not k.startswith("_")
        }
        print(_json.dumps({"vars": _ns}), flush=True)
        continue

    if _code == "__api__":
        _name = _request.get("name")
        try:
            _output = _sandbox.get_interface(_name)
        except Exception:
            _output = f"api error: {_traceback.format_exc()}"
        print(_json.dumps({"api": _output}), flush=True)
        continue

    # -- exec --
    _result = _sandbox.exec(_code)
    _output = {
        "returns": repr(_result.returns) if _result.returns is not None else None,
        "std_output": _result.std_output,
        "exception": _result.exception,
        "traceback": _result.traceback,
    }
    print(_json.dumps(_output), flush=True)

# ── Cleanup ────────────────────────────────────────────────────────────

# Call close()/stop() on domain objects before destroying sandboxes.
# This handles resources like browser/playwright that need explicit teardown.
for _k, _v in list(_init_sandbox.module.__dict__.items()):
    if _k.startswith("_"):
        continue
    for _method in ("close", "stop"):
        _m = getattr(_v, _method, None)
        if callable(_m):
            try:
                _m()
            except Exception:
                pass
            break

_sandbox.close()
_init_sandbox.close()
