# Data Ghost

Data is the persistent counterpart to Atom. Atom keeps a process-local linear
history; Data reconstructs every model history from an owner-scoped Memento and
writes completed Moments back after articulation.

The first version deliberately runs Memento's degraded mode: one current branch,
mechanical commits, no fork, no reflection daemon, and no CTML memory channel.
The storage root defaults to `{GhostWorkspace.home}/memento`.
