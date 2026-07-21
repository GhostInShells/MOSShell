# Playwright Browser App — browsers/playwright

AI-controlled browser via ModuleEval subprocess Sandbox.

## Architecture

```
main.py (Channel, async)          _eval_server.py (child, sync)
─────────────────────────         ─────────────────────────────
ModuleEval.start()                Compile playwright_domain.py
  matrix.spawn() → child            import playwright, launch browser
  wait "ready"                      inject page/browser/context
                                  Sandbox wrapper
exec command:                      eval loop:
  eval.exec(code)                    stdin.readline → sandbox.exec(code)
  JSON request →                     stdout.write → JSON result
```

## Files

- `main.py` — App entry. Builds channel, provides to Matrix
- `playwright_domain.py` — Domain module. Module-level init creates browser/page
- `eval_server.py` — Removed. Replaced by generic `ghoshell_moss.tools._eval_server.py`

## Protocol

The eval server is generic (`tools/_eval_server.py`), not Playwright-specific.
Domain logic lives in `playwright_domain.py` — its source IS the instruction.

```
Request  →  {"code": "page.goto('...')"}
Response ←  {"returns":null, "std_output":"Example Domain", "exception":null, "traceback":null}
```

## Design decisions

- **Subprocess, not thread** — Playwright sync API cannot run inside asyncio event loop
- **Generic eval_server** — Any .py file can be a domain module; the eval server is a 120-line shared script
- **Source = instruction** — Code as Prompt: the AI sees the domain module source directly
- **Two-layer Sandbox** — init_sandbox (builtins=None) holds domain objects; sandbox (SANDBOX_BUILTINS) runs AI code

## Related

- FEATURE.md: `.ai_partners/features/workstreams/2026/06/module-eval-channel/`
- Channel type: `ghoshell_moss.channels.module_eval_channel`
- Core: `ghoshell_moss.tools.module_eval`
- Eval server: `ghoshell_moss.tools._eval_server.py`
