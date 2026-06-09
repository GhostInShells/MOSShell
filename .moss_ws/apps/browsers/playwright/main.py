"""Playwright Browser App — subprocess Sandbox eval server.

Spawns eval_server.py as a child process, communicates via stdin/stdout
JSON-line protocol.  Playwright runs in the child (no asyncio), Sandbox
provides builtins safety + persistent namespace.
"""

import json
import subprocess
import sys
import threading

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel


class EvalServer:
    """Manages the eval_server.py child process."""

    def __init__(self, server_script: str):
        self._proc = subprocess.Popen(
            [sys.executable, "-u", server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._lock = threading.Lock()

        # Wait for "ready" signal
        ready_line = self._proc.stdout.readline()
        if ready_line.strip() != "ready":
            stderr = self._proc.stderr.read()
            raise RuntimeError(f"Eval server failed to start: {ready_line!r} stderr={stderr!r}")

    def send(self, code: str) -> dict:
        """Send code to the server, return the JSON result dict."""
        request = json.dumps({"code": code})
        with self._lock:
            self._proc.stdin.write(request + "\n")
            self._proc.stdin.flush()
            response_line = self._proc.stdout.readline()
        return json.loads(response_line)

    def shutdown(self):
        """Send shutdown signal and wait for the child to exit."""
        try:
            self._proc.stdin.write(json.dumps({"code": "__SHUTDOWN__"}) + "\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


# ── Channel ───────────────────────────────────────────────────────────

def build_channel():
    server_script = __file__.replace("main.py", "eval_server.py")
    server = EvalServer(server_script)

    chan = new_channel(
        name="playwright",
        description=(
            "Playwright browser control — subprocess Sandbox eval server. "
            "exec(code), vars(), api(name, *methods)."
        ),
    )

    @chan.build.close
    async def cleanup():
        server.shutdown()

    @chan.build.command(name="exec", always_observe=True)
    async def exec_code(text__: str) -> str:
        """Execute Python code in the browser namespace.  text__: code string.

        Use open-close tags:
            <playwright:exec>
            page.goto("https://example.com")
            print(page.title())
            </playwright:exec>

        Variables persist across calls.  print() output is captured.
        """
        result = server.send(text__)
        parts = []
        if result["std_output"]:
            parts.append(result["std_output"].rstrip())
        if result["exception"]:
            parts.append(f"Error: {result['exception']}")
            if result["traceback"]:
                parts.append(result["traceback"].rstrip())
        if result["returns"]:
            parts.append(f"__result__: {result['returns']}")
        return "\n".join(parts) if parts else "(executed, no output)"

    @chan.build.command(name="vars", always_observe=True)
    async def list_vars() -> str:
        """List objects in the browser namespace (via dir() reflection)."""
        result = server.send("import json; print(json.dumps({k: type(v).__name__ for k, v in [(k, v) for k, v in list(locals().items()) if not k.startswith('_')]}))")
        return result["std_output"] or "(empty namespace)"

    return chan


_channel = build_channel()


async def main(matrix: Matrix):
    await matrix.provide_channel(_channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
