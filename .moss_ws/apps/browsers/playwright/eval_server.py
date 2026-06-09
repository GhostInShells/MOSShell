"""Persistent Sandbox eval server — stdin/stdout JSON-line protocol.

Protocol:
  in  — code string (one "line", may contain embedded newlines as \n)
  out — JSON: {"returns": ..., "std_output": ..., "exception": ..., "traceback": ...}

Send __SHUTDOWN__ to exit cleanly.
"""

import sys
import json
import urllib.parse

from playwright.sync_api import sync_playwright

from ghoshell_moss.core.codex.sandbox import Sandbox, SANDBOX_BUILTINS

# ── Init Playwright ───────────────────────────────────────────────────

_playwright = sync_playwright().start()
_browser = _playwright.chromium.launch(headless=False)
_context = _browser.new_context()
_page = _context.new_page()

# ── Build sandbox, inject objects ─────────────────────────────────────

_init_sandbox = Sandbox(name="playwright", builtins=None, source="")
_init_sandbox.set("page", _page)
_init_sandbox.set("browser", _browser)
_init_sandbox.set("context", _context)
_init_sandbox.set("json", json)
_init_sandbox.set("urllib", urllib)

_sandbox = Sandbox(
    name="playwright",
    parent=_init_sandbox,
    builtins=SANDBOX_BUILTINS,
    source="",
)

# ── Eval loop ─────────────────────────────────────────────────────────

print("ready", flush=True)

for line in sys.stdin:
    request = json.loads(line)
    code = request["code"]
    if code == "__SHUTDOWN__":
        break

    result = _sandbox.exec(code)
    output = {
        "returns": repr(result.returns) if result.returns is not None else None,
        "std_output": result.std_output,
        "exception": result.exception,
        "traceback": result.traceback,
    }
    print(json.dumps(output), flush=True)

# ── Cleanup ───────────────────────────────────────────────────────────

try:
    _browser.close()
    _playwright.stop()
except Exception:
    pass
_init_sandbox.close()
