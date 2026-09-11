---
name: 'signal_receiver'
description: 'dogfood signal drainer — subscribes to the session signal bus and prints each signal as full JSON'
singleton: true
exec:
  command: python
  args: main.py
---

Signal drainer probe. Subscribes to `matrix.session.on_signal` and drains the
queue, printing each signal's full JSON to stdout. No channel — works
standalone, no Ghost needed. Read output via `matrix.nodes:read_output`.

Run in the same network scope as the emitter:

    moss nodes run .moss/system_test_nodes/signal_receiver/

Pair with `.moss/system_test_nodes/signal_sender/` to verify the bus, or with
`moss audio listen` to observe listener signals (first/clause/tail).
