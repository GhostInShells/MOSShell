# MEMENTO Specification — the cognitive trajectory

Memento is the layer that lets a mind persist across sessions.

A session is finite: its context is built, used, and then gone. What must survive is not
the transcript, but the ability to **reach** what happened — a span of experience with a
stable address, and the note saying what it meant while it was still fresh. Memento is
that addressability.

It stores no conversation. A commit holds the *key* to restore a span; the span itself
lives where the session lived. Memento's job is to make the span findable, immutable in
its identity, and interpretable forever.

This document is the contract. It says what must be true, never how it is stored: any
implementation that satisfies it — files, a table, a database — is valid. Where this
document and the code disagree, the code is authoritative and this document is wrong.

## Why it is shaped this way

**The unit of memory is a span, not a message.** A session does not remember sentence by
sentence; it remembers "that stretch where we settled the design". So the commit is
written at a boundary the writer chooses, and it addresses the whole stretch between two
boundaries. Nothing between them needs to be replayed to reach it.

**Facts and interpretations have different lifetimes.** What happened is frozen. What we
think it meant keeps changing. Memento keeps them apart: the commit is immutable, the
note is last-wins and may be rewritten forever. This is what lets memory be re-read
without rewriting history — and history be trusted while interpretation improves.

**Continuity is linear; divergence is a reference.** A branch is what happened next. When
a direction splits off, it starts a new branch *from a coordinate* — a pointer, never a
copy. The parent's history is neither duplicated nor rewritten, so both sides stay cheap
and both stay true.

**Addressing beats enumeration.** A coordinate is short, stable and printable. Reaching
one commit never requires loading the trajectory, and a model and a human use the same
address for the same place.

**The record must survive being read.** A commit that cannot be read keeps its position
and surfaces as broken; it never shifts the coordinates of everything after it. A
trajectory that refuses to advance because one record is unreadable is worse than one that
reports what it lost.

**The write side is governed at the owner, not per record.** One owner has at most one
writer at a time, and the host chooses it. Memento has no per-write lock, no retry, and no
arbitration between writers — a second writer is a configuration error, not a race to
survive.

## The model

```
owner
 └── branch — name, identity, position
      ├── commit 1 ── note
      ├── commit 2 ── note        (absorbs references that arrived since commit 1)
      │     └── child: another branch started here
      └── commit 3 ── note
```

| term | what it is | what it guarantees |
|---|---|---|
| **Owner** | the identity boundary: one trajectory scope and its write gate | at most one writer at a time; readers never write, never create |
| **Branch** | a continuous history with a name, a position, and an append-only sequence of commits | a stable identity; a name that may move |
| **Commit** | one frozen point: the key to restore a span, plus the references it absorbed | immutable, never deleted |
| **Note** | the message of a commit — what that span was | last-wins, droppable, rebuildable |
| **Coordinate** | `{branch_position}-{commit_seq}`, e.g. `27-1027` | stable for the life of the trajectory |
| **Reference** (confluence) | something another branch sent, absorbed by this branch's next commit | carries the source address and title, never the source content |
| **Child** | a branch created from this one, recorded at the position where it happened | says that and when, never why |
| **Fork** | a new branch starting from a coordinate on an existing one | a reference, never a copy |
| **Node space** | an unmanaged place a commit may own for its own materials | memento reports whether it is claimed; never reads it |

## Lifecycle

1. **Open an owner.** One process opens it writable and keeps it; readers may open it
   read-only, and a read-only owner performs no write at all — it does not initialize,
   repair, or create anything.

2. **Create a branch.** A name and a description; identity and position are assigned. The
   name must not exist. The name is a label and may later move; the identity never does.

3. **Commit a span.** Freeze what has happened since the last commit. The commit carries
   the consumer's restore metadata — opaque to memento — and absorbs the references that
   arrived since the previous commit.

4. **Note it.** Write what the span was. A note may arrive after the commit: summarization
   is slower than the boundary, and a late note must attach to the right commit. Rewriting
   a note is normal; the latest version wins. A note that has not arrived leaves the commit
   readable but silent.

5. **Fork.** Open a new branch from a coordinate. The new branch keeps the parent as a
   pointer, so a reader can walk back without the parent being copied.

6. **Reference across branches.** A note may name recipients; the reference is staged for
   each of them and absorbed by that branch's next commit. The recipient's history is not
   altered otherwise: a reference is a fact recorded at a position, not a merge of parent
   chains.

7. **Archive.** Release the name and keep everything else. An archived branch and its
   commits stay readable by coordinate; the name may be given to a new branch.

## Identity, position, name

Three fields, three jobs, and they must not take each other's place.

- **Identity** — an opaque string the implementation assigns at creation: unique within
  the owner for a branch, unique within the branch for a commit, and unchanged for life.
  It is a key, never an address. Nothing may parse it, and its form is not part of this
  contract.
- **Position** — the halves of the coordinate: the branch's creation ordinal in the
  owner, and the commit's ordinal in the branch. Both are frozen at production, and both
  sequences are append-only, which is why a coordinate is permanently stable.
- **Name** — for the eye. Names rot: an archived name can be given to a new branch. A
  relationship must never resolve through a name.

Models, humans and tools all address by coordinate. The identity is never shown, and never
needed to reach anything.

## Reading a trajectory

- **Window** — the tip of a branch, or the trajectory as it stood before a given
  coordinate: the recent commits in detail, the past folded into counts, ranges and a few
  titles. A window never materializes the whole history.
- **History** — the commits of one branch over a time range, one line each, with notes.
- **Raw span** — the conversation behind one commit. It lives in the session store, so
  reading it belongs to the layer that holds that store; memento only holds the key.
- **Talk** — one round of inference with the context as it stood at a commit. The other
  side sees only what had happened by then.
- **Node space** — the materials a commit owns, when it has claimed them.
- **Events** — the owner announces what happens (a branch created, a commit made, a
  reference landed, a branch archived), so a watcher can react without polling.

Folded history is a deliberate compromise: far commits become facts (count, range, a few
nearest titles) instead of objects. The fold is deterministic, so two readers of the same
window see the same thing.

## Guarantees

1. A commit is immutable. A coordinate, once printed, always resolves to the same commit.
2. Commits are never deleted; archive removes a name pointer only.
3. The note is the only home of a commit's message. It is last-wins, may be rebuilt, and
   may be lost without breaking the trajectory.
4. A commit that could not be read keeps its position, carries its failure on itself,
   and does not shift any other coordinate. Nothing is dropped silently: unreadable
   references are reported on the commit that would have absorbed them.
5. A fork never copies or rewrites its parent.
6. A reference carries the source address and the title, never the source content.
7. A commit's node space is never read, cleaned, or versioned by memento.
8. At most one writer per owner; a read-only owner writes nothing at all.

## Non-goals

- No semantic search, no embeddings, no automatic relevance ranking.
- No merge of parent chains. References are facts, not merges.
- No versioning of node-space content — that belongs to whoever writes there.
- No cross-owner coordination: uniqueness and resolution are owner-scoped.
- No interpretation of restore metadata. The consumer defines it; memento only stores and
  returns it.

## Implementation freedom

The contract fixes behavior, not layout. Storage may be files, a table, a database, or
something else; identities may be counters, keys, or names generated by the store. What it
must not do is leak into the surface: no field of this model carries a storage detail, and
no caller may depend on one. An implementation that changes its layout without changing
the guarantees has not changed the contract.
