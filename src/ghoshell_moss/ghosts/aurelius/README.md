# Aurelius Ghost

Aurelius is the persistent counterpart to Atom. Atom keeps a process-local linear
history; Aurelius reconstructs every model history from an owner-scoped Memento and
writes completed Moments back after articulation.

Aurelius freezes completed Moments into owner-scoped Memento commits. Its mechanical
checkpoint is followed by a non-blocking, retryable reflection that appends a
new interpretation note without changing frozen records; startup catches up any
unreflected checkpoint (including legacy empty notes).

Factual recall is agentic, not intercepted. There is no regex Claim/verifier layer
parsing Moment bodies (`core.memento` payloads stay opaque). Instead the model reads
its own trajectory: `memory_search` greps the owner's frozen commits and staging as
plain text, `memory_show` expands a frozen commit like a page fault, and an injected
discipline instruction tells the model to search-and-cite before answering fact
questions — and to say it found no evidence rather than guess. The system reports;
the model decides.

A sister bypass, `AureliusCurator`, mirrors reflection: a background model reads the
frozen trajectory and rewrites a human-readable `facts.md` that is pinned into the
memory Ground. It is append-only, fallible, off the articulate hot path, and never
parses payload semantics. Disabling it loses nothing — the model can still search.

`AureliusDesktop` owns existing `DefaultGrounds` during the Ghost lifecycle. Its
`DESKTOP.md` law enters the current instruction and its Pin frame enters only the
current prompt; neither is persisted into Memento. This adapter does not change the
Desktop, Memento, Ghost, or GhostRuntime contracts.

`MemoryConfig` controls the history window, checkpoint/reflection policy, curation
policy, the memory-discipline instruction, and automatic Ground opening. The `ghost`
CTML channel exposes model-visible retrieval (`memory_search`/`memory_log`/`memory_show`)
plus hidden human-operator control (commit/reinterpret/fork/switch/reflect/curate) and
Ground open/pin/update/frame actions for the current owner only. The storage root
defaults to `{GhostWorkspace.home}/memento`.
