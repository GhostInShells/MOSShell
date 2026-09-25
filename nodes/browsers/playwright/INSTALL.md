# Install

The playwright node requires its own venv (heavy dependency).

```bash
cd nodes/browsers/playwright
uv sync
playwright install chromium
```

After install, run:

```bash
moss nodes install nodes/browsers/playwright
```
