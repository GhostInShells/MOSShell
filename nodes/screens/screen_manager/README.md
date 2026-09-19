# screen_manager

MOSS 的屏幕躯体 —— window 语义（item / group / arrange / fullscreen / veil /
background）的零依赖 webview 合成器。一个进程、两张脸共享一个 store：channel
（ghost 的控制面）和 web surface（人类的观看与操控面）。

语义与设计见
`.ai_partners/features/workstreams/2026/07/screen-manager/web_manager.md`。

## Run

```bash
moss nodes run nodes/screens/screen_manager
# 或直接调试
python nodes/screens/screen_manager/main.py
```

人类 surface 在 `http://127.0.0.1:8766`。端口用 `--port N` 或
`MOSS_SCREEN_MANAGER_PORT` 覆盖。

## Test

```bash
pytest nodes/screens/screen_manager/tests/
```
