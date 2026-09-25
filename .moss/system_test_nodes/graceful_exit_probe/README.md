# graceful_exit_probe

Regression probe for the `cell-graceful-exit` contract: a second SIGINT arriving
while the matrix tears down must still exit gracefully — exit code 0, no
`ERROR` traceback in the cell log.

## Run

```bash
python .moss/system_test_nodes/graceful_exit_probe/main.py
```

or, via the node CLI:

```bash
moss nodes run .moss/system_test_nodes/graceful_exit_probe
```

The node terminates itself — a background thread fires two SIGINTs once the
channel is live. Do not leave it running.

## Pass criteria

- process exit code is `0`
- no new `ERROR` entry from `project.py:845` in the cell log
  (`moss.node__graceful_exit_probe.log` when run as a node, `moss.log` when run
  directly via `python main.py`)
