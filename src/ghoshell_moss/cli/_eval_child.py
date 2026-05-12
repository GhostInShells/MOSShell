"""
Child process entry point for moss eval.

Reads a JSON request from stdin:
    {"code": "...", "module": "ghoshell_moss.host.moss_runtime" or null}

Executes code in an isolated module (optionally inheriting a module's context),
captures stdout, and writes a JSON result to stdout:
    {"returns": ..., "std_output": "..."}

On error, prints traceback to stderr and exits with code 1.
"""

import sys
import json
import types
from typing import Optional


def main() -> None:
    try:
        raw = sys.stdin.read()
    except Exception:
        print("Failed to read stdin", file=sys.stderr)
        sys.exit(1)

    try:
        request = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON request: {e}", file=sys.stderr)
        sys.exit(1)

    code: str = request.get("code", "")
    module_path: Optional[str] = request.get("module")

    if not code.strip():
        print("No code to execute", file=sys.stderr)
        sys.exit(1)

    origin: types.ModuleType
    if module_path:
        import importlib
        try:
            origin = importlib.import_module(module_path)
        except ModuleNotFoundError:
            print(f"Module not found: {module_path}", file=sys.stderr)
            sys.exit(1)
    else:
        origin = types.ModuleType("__moss_eval__")

    try:
        from ghoshell_moss.core.codex.executor import Executor
        executor = Executor(origin=origin)
        result = executor.execute(code)
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    returns = result.returns
    try:
        json.dumps(returns)
    except (TypeError, ValueError):
        returns = repr(returns)

    output = {
        "returns": returns,
        "std_output": result.std_output,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
