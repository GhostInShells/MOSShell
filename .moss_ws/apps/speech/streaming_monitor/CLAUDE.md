# Speech Streaming Monitor

Key files: `APP.md`, `main.py`, `CLAUDE.md`

Runtime: `moss apps test speech/streaming_monitor`

What main.py does: receives Matrix, subscribes to `speech/streaming` Topic via Session,
prints each sentence with batch marker in terminal.
