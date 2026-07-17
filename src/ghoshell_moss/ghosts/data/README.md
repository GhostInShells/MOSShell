# Data Ghost

Data is the persistent counterpart to Atom. Atom keeps a process-local linear
history; Data reconstructs every model history from an owner-scoped Memento and
writes completed Moments back after articulation.

Data freezes completed Moments into owner-scoped Memento commits. Its mechanical
checkpoint is followed by a non-blocking, retryable reflection that appends a
new interpretation note without changing frozen records; startup catches up any
unreflected checkpoint (including legacy empty notes).

`MemoryConfig` controls the context window, automatic checkpoint cadence and
reflection policy. The `ghost` CTML channel exposes bounded inspection,
semantic checkpoints, reinterpretation and timeline fork/switch operations for
the current owner only. The storage root defaults to
`{GhostWorkspace.home}/memento`.
