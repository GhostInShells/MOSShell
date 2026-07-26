# Screen Node — Installation

Requires Qt6 with WebEngine (Chromium-based webview).

```bash
uv pip install PySide6
```

Verify:
```bash
python -c "from PySide6.QtWebEngineWidgets import QWebEngineView; print('ok')"
python nodes/screens/main.py --standalone    # GUI smoke test
```
