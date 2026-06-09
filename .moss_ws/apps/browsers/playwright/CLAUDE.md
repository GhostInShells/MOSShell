# Playwright Browser App — browsers/playwright

AI-controlled browser via subprocess Sandbox eval server.

## Architecture

```
main.py (Channel, async)          eval_server.py (child, sync)
─────────────────────────         ────────────────────────────
EvalServer.__init__()             import playwright
  Popen → wait "ready"              init Sandbox
                                    inject page/browser/context
exec command:                      eval loop:
  server.send(code)                  stdin.readline → JSON request
  stdin → JSON request               sandbox.exec(code)
  stdout ← JSON result               stdout.write → JSON result
```

## Protocol

```
Request  →  {"code": "page.goto('...')"}
Response ←  {"returns":null, "std_output":"Example Domain", "exception":null, "traceback":null}
```

## Design decisions

- **Subprocess, not thread** — Playwright sync API cannot run inside asyncio event loop
- **JSON-line, not terminal** — No prompt matching, no ANSI stripping, self-delimiting
- **Sandbox in child** — SANDBOX_BUILTINS restricts __import__/open/eval/exec
- **Module-level EvalServer init** — Happens before Matrix event loop starts

## Related

- FEATURE.md: `.ai_partners/features/workstreams/2026/06/module-eval-channel/`
- Architecture: `.design/2026-06-09_subprocess_sandbox_eval_protocol.md` (待写)
