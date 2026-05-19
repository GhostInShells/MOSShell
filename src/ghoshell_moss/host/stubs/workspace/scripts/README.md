Scripts are lightweight, one-shot Python files launched via `moss script run <name>`.
They connect to the running matrix network, perform a task, and exit.

Unlike apps, scripts:
- Have no persistent runtime (no Circus, no respawn)
- Reuse the moss Python runtime (no pyproject.toml isolation)
- Are dev-time tools — debugging, probing, sending signals
- Live in `scripts/<name>/SCRIPT.md` + `main.py`

## Convention

Each script is a subdirectory under `scripts/` containing:
- `SCRIPT.md` — manifest (YAML frontmatter + optional description)
- The entry point (default: `main.py`, configurable in SCRIPT.md)

## Launch

```
moss script run <name>
```

This discovers the workspace, builds MOSS env vars (MOSS_CELL_ADDRESS=script/<name>),
and runs the script as a foreground subprocess.
