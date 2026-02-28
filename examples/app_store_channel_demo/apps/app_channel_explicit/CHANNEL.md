---
name: "explicit_chan"
description: "A channel-mode app (loads __channel__ from a python file)."
main: "channel_def.py"
mode: "channel"
---

## Usage

- Command: `ping() -> str`
- Command: `echo(text) -> str`

This app is launched in `mode=channel`, meaning the app's `main` file defines a `__channel__` instance.
