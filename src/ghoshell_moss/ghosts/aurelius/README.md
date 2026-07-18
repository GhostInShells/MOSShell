# Aurelius Ghost

Aurelius is the persistent counterpart to Atom. Atom keeps a process-local linear
history; Aurelius reconstructs every model history from an owner-scoped Memento and
writes completed Moments back after articulation.

Aurelius freezes completed Moments into owner-scoped Memento commits. Its mechanical
checkpoint is followed by a non-blocking, retryable reflection that appends a
new interpretation note without changing frozen records; startup catches up any
unreflected checkpoint (including legacy empty notes).

Above that ledger, `MemoryProjection` rebuilds evidence-backed Claims from the
current branch. Only explicitly parseable user and authenticated-tool percepts
can become active Claims; model logos and reflection notes remain non-authoritative
candidates. Factual memory questions receive a bounded evidence packet, and their
generated answer is verified before it is yielded. Unknown or conflicting facts
fail closed instead of being guessed.

`AureliusDesktop` owns existing `DefaultGrounds` during the Ghost lifecycle. Its
`DESKTOP.md` law enters the current instruction and its Pin frame enters only the
current prompt; neither becomes a durable Claim. This adapter does not change the
Desktop, Memento, Ghost, or GhostRuntime contracts.

`MemoryConfig` controls the history window, checkpoint/reflection policy, knowledge
packet budget, trusted percept sources, and automatic Ground opening. The `ghost`
CTML channel exposes bounded Memento inspection/control, Claim recall/audit, and
Ground open/pin/update/frame actions for the current owner only. The storage root
defaults to `{GhostWorkspace.home}/memento`.
