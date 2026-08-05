# Install

The voice node requires its own venv with dependencies.

```bash
cd nodes/sensors/voice
uv sync
```

Dependencies (pyproject.toml):

- `ghoshell-moss[matrix]` — channel builder, mindflow, topics
- `miniaudio` — microphone capture
- `click` — CLI entry
- Volcengine ASR (`websockets`, `scipy`) — bundled via ghoshell-moss host layer

After `uv sync`, run:

```bash
moss nodes install nodes/sensors/voice
```
