"""
moss eval — execute arbitrary Python code in the live MOSS runtime.

Spawns a child process to isolate code execution from the CLI layer.
The child reads a JSON request from stdin, executes code via
Compiler + Executor in a clean process, and writes JSON result to stdout.
Errors are written to stderr and displayed as-is by the parent.
"""

import typer
import json
import sys
import subprocess
from pathlib import Path
from typing import Optional

from .utils import console, print_error


def eval_code(
    code: str = typer.Argument(default="", help="Python code to execute"),
    module: Optional[str] = typer.Option(
        None, "--module", "-m",
        help="Import path of module to use as execution context",
    ),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f",
        help="Read code from file instead of argument",
    ),
):
    """
    Execute Python code in the live MOSS runtime environment.

    Code runs in an isolated child process with full access to the host.
    stdout is captured automatically. Assign to __result__ to return a value.

    Examples:
        moss eval "__result__ = type(42)"
        moss eval --module ghoshell_moss.host.moss_runtime "print(dir())"
        moss eval --file debug_script.py
    """
    if file:
        try:
            code = file.read_text()
        except FileNotFoundError:
            print_error(f"File not found: {file}")
            raise typer.Exit(code=1)

    if not code.strip():
        print_error("No code to execute. Provide code as argument or via --file.")
        raise typer.Exit(code=1)

    request = json.dumps({"code": code, "module": module})

    child = subprocess.run(
        [sys.executable, "-m", "ghoshell_moss.cli._eval_child"],
        input=request,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if child.returncode != 0:
        print_error(f"Code execution failed:\n{child.stderr.rstrip()}")
        raise typer.Exit(code=1)

    # Child succeeded — output its JSON result
    console.print(child.stdout.rstrip())
