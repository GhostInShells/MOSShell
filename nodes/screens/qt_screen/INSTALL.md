# Screen Node — Installation

This node has its own Python dependencies including PySide6 with QtWebEngine.

```bash
cd nodes/screens/qt_screen
uv sync
```

Verify WebEngine availability:
```bash
.venv/bin/python -c "from PySide6.QtWebEngineQuick import QtWebEngineQuick; print('ok')"
```

Then mark as installed:
```bash
moss nodes install nodes/screens/qt_screen
```

GUI smoke test (standalone QML, no Matrix):
```bash
.venv/bin/python main.py --standalone
```
