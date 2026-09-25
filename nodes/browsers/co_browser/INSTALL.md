# Install

The co_browser node shares playwright's heavy dependency — it needs its own venv.

```bash
cd nodes/browsers/co_browser
uv sync
playwright install chromium
```

After install, run:

```bash
moss nodes install nodes/browsers/co_browser
```
