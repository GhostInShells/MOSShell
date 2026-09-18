# terminal

Shell commands as **cards**. The ghost issues a command and gets a receipt
immediately — it never blocks on a human's response time. Each command streams
onto a card on a web surface, where a human watches it appear and answers
`accept` / `deny` / `ask`. What happens next reaches the ghost as a signal, not
a return value.

One process, two faces over one store: the channel (the ghost's control surface)
and the web surface (the human's card stream). See `NODE.md` for the full model —
cards, threads, per-thread trust, rules, the ground field, and analyze.

## Run

```bash
# as a node (named mount under matrix.mesh.<name>)
moss nodes run nodes/os/terminal

# or directly, for debugging
python nodes/os/terminal/main.py
```

The human surface serves at `http://127.0.0.1:8768`. Override the port with
`--port N` or `MOSS_TERMINAL_PORT`.

## Test

```bash
pytest nodes/os/terminal/tests/
```
