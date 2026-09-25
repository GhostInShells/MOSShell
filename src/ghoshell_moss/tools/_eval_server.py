"""Generic eval server — stdin/stdout JSON-line protocol.

Receives MODULE_FILE env var, compiles it, and serves exec requests over
JSON-line. Requests carry an ``id`` that is echoed in the response so the
parent can match responses to in-flight requests (pipelining + timeout).

Protocol:
  in  — {"id": "...", "code": "..."}
  out — {"id": "...", "returns": ..., "std_output": ..., "exception": ..., "traceback": ...}

Special codes:
  __SHUTDOWN__  — exit cleanly

Builtins are unrestricted (``builtins=None``): the domain module's own imports
are the declared authorization boundary, surfaced to the model as instruction.
No whitelist — containment is not this layer's job.
"""

import json as _json
import os
import sys
import traceback as _traceback
from pathlib import Path

from ghoshell_moss.core.codex.compiler import Compiler
from ghoshell_moss.core.codex.sandbox import Sandbox

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

# ── Sandbox (unrestricted builtins) ────────────────────────────────────

# Root sandbox holds the compiled domain objects; builtins=None → full Python
# builtins. The AI exec namespace shares the same __dict__ and inherits them.
_init_sandbox = Sandbox(
    name=_module_name,
    source=_source,
    builtins=None,
)

for _k, _v in _compiled.__dict__.items():
    if not _k.startswith("__"):
        _init_sandbox.set(_k, _v)

_sandbox = Sandbox(
    name=_module_name,
    parent=_init_sandbox,
    source=_source,
)

# ── Eval loop ──────────────────────────────────────────────────────────

print("ready", flush=True)

for _line in sys.stdin:
    _request = _json.loads(_line)

    _code = _request.get("code", "")
    _rid = _request.get("id")

    if _code == "__SHUTDOWN__":
        break

    _result = _sandbox.exec(_code)
    _output = {
        "id": _rid,
        "returns": repr(_result.returns) if _result.returns is not None else None,
        "std_output": _result.std_output,
        "exception": _result.exception,
        "traceback": _result.traceback,
    }
    print(_json.dumps(_output), flush=True)

# ── Cleanup ────────────────────────────────────────────────────────────

# Call close()/stop() on domain objects before destroying the sandboxes.
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
