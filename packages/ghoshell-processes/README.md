# ghoshell-processes

Pure mechanism-layer governance of child processes: spawn, observe, signal, and reap.

A subprocess never outlives its owner. The owner-bound `Subprocesses` instance is the
ledger; consumers that only spawn get `SubprocessFacade` and structurally cannot close a
shared instance. Output capture, exit callbacks, and process-group termination are part
of the contract; the default implementation is asyncio-based.

No dependency on MOSS or ghoshell-common. Runtime dependencies are `pydantic` and
`typing-extensions` only.
