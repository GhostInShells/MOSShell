---
name: explorer
description: The explorer mechanism — a one-turn ego fork dispatched by name, whose answer is committed to its own memento branch
---

# explorer

> **Status:** design settled D17 (2026-11-17), not built. This is a construction plan, not a
> discussion. The workstream's `FEATURE.md` is the index.

## What it is

> **An explorer dispatch = a session forked from the ego at one step edge, run to
> `turn/end`, whose final answer is committed by the runtime as one commit on that
> explorer's own memento branch.**

Every comma in that sentence is checkable. The decisive one is the last: **an explorer
leaves a commit.** That is also what separates it from `beside`, which leaves nothing —
beside is a sidecar; an explorer is a recorded case.

`explorer` is the type. The domain — what it looks at, how deep, what shape of answer it
brings back — is the *instruction inside its branch*, not code.

## Why the spec lives in a memento branch, not in ghost home

thirdgerb's argument, quoted:

> 这个不能在 ghost home 里的原因是, ghost home 目录的拓扑会因同类机制反复扩大. 要放 ghost
> home 也应该做分层治理, 然后你还得找个办法关联 commit 历史 和 版本. 可以 archive 又保住了
> 历史资产的 memento branch, 我个人觉得就是最好的做法.

The consequence he named in the same turn: **the branch is the version.** Not a field in a
file — the branch identity itself.

The memento contract already says this, from the other direction (`abcd.py:117`):

> `branch_id` is the identity. **Every resolution starts here.** `branch_index` is the
> position... `branch_name` is for the eye. Names rot (a deleted name can be given to a new
> branch), so a name must never be used to resolve a relationship — only to show one.

So a version promotion is a pointer swap, not a copy: the new version takes the name, the old
one keeps its `branch_id`, its commits, and its directory — archived, readable, done.
`ghost_home` never grows a directory per explorer, and nothing in the cognitive field
(`ground`) is polluted by a mechanism that is not cognition.

**Missing capability #1: `amend_ref`.** The `Memento` ABC has `create_branch`,
`archive_branch`, `list_branch_catalog`, `resolve_commit`, `children_of` — and no way to edit
a branch's ref row. Rename and description are the *same* edit (`BranchRef` carries `name` and
`description`; identity is `branch_id`), so one verb covers both. It writes the ref, deletes
the old one, touches no commit.

## The five invariants

1. **Context** — fork from the ego at a step edge (`fork_into(cwd, at_seq=edge)`), never from
   the ghost inception. The explorer answers inside *this task's* frame, and the shared prefix
   keeps the cache useful.
2. **Permission** — `read-only` sandbox for its whole life; MOSS CTML tools never mounted.
   It can look at the world and cannot touch it.
3. **Duration** — one `run()` to `turn/end`. No follow-up question, no second turn.
4. **Record** — one commit on its own branch; the answer *is* the note (title = the question,
   body = the answer). One text, not two.
5. **Reproducibility** — every commit is self-sufficient: `(anchor, spec)` reconstructs the
   call without consulting anything else.

## What a commit carries

`CommitRef.metatype` / `CommitRef.metadata` are the extension point, and memento's own
docstring says what they are for: *"holds the keys sufficient to restore one session ... memento
only gets/sets, never parses"* (`abcd.py:207`). The ego's convention is `metatype="session"`,
`metadata={"ref": DshSessionRef, "prev_turn": int}` (`_ego_memento.py:428`). An explorer's
commit is a different metatype with a different convention — no core change:

```
metatype = "explorer"
metadata = {
    "anchor": DshSessionAnchor(...),   # the ego-side fork point — the start
    "ref":    DshSessionRef(...),      # the forked session + its turn span — the end is the
                                       # final answer turn
    "ask":    "<the question>",        # same source as the title, for scripts
    "spec":   {"name": ..., "sha": ..., "version": ...},
}
```

Three things fall out of this shape:

- **The whole exploration is readable.** `read(coord)` is the existing route — ref → turn
  slice, live or cold (`index.ts:1471`, `loadSourceEvents(ctx, ref.session_id)`). The explorer's
  tool calls and reasoning are the case's asset, not extra storage to decide about.
- **A case is replayable by a plain script.** Anchor + ask + spec is everything.
- **A/B is a coordinate plus a name.** Inside one owner, `3-9` is a case; run it with another
  branch's spec and compare.

**Missing capability #2: nothing.** `metatype`/`metadata` already exist; the explorer simply
defines its own convention.

## Storage: its own memento owner

One owner = one directory (`Memento` docstring). The explorer owner lives at
`<ghost_home>/.memento/explorers/`, so:

- the ego's branch list does not fill up with explorers;
- archive-with-history comes free;
- the branch list is not a cognitive surface, so nothing here grows the ground.

There are no cross-owner coordinates. An ego coordinate (`1-761`) does not exist in the
explorer owner; the portable object between owners is an **anchor**
(`DshSessionAnchor`, session id + seq), never a coord.

## Command surface

`ghost.beside.explorer` — a gated child of `beside` (the same `mount_child` pattern listener
and mindflow already use). Not mounted → one existence line in the notice, no facade cost.

| Verb | What it does |
|---|---|
| `list` | explorers: name, one line, draft/alive, version, recent case count |
| `new(name, description, from="")` | create the branch + write the `EXPLORER.md` template; `from` copies another spec and bumps `version` |
| `read(name)` | the file text + its absolute path (edits go through the ego's own file tools) |
| `dispatch(name, task)` | fork at the current edge, one turn, commit |
| `archive(name)` | archive the branch; the asset stays readable |

**Draft rule:** an empty instruction is a draft — not dispatchable. No `status` field, no
approval gesture. Writing the file *is* the grant (the dsh sandbox already governs who may
write it, and config adds a second layer).

### Two high-order verbs, behind a flag

`test` and `replay` are the iteration surface, and they are gated by an `available` fn (a
config flag), so a default ghost never sees them.

| Verb | Input | Writes a commit? |
|---|---|---|
| `test(name, task)` | a new proposition | no |
| `replay(coord, name="")` | an existing case | no |

They are not one verb with an optional argument: one takes a proposition, the other takes a
coordinate. Both leave the branch untouched — iterating on a prompt must not fill the log with
trial runs.

## The plugin side: side sessions are classes, not a ternary

`index.ts:779` currently decides by one predicate — `stepAgent.id !== doloresEgoSessionId` →
every non-primary session is a *bypass*: read-only sandbox every pre-step, MOSS CTML tools
denied, one bypass instruction injected once. The explorer fits most of that by accident, and
that is the problem: three different things share one branch of an if-statement.

- `bypass` — note / chat / beside: one model call, nothing persisted;
- `explorer` — a first-class session: own cwd, own log, own commit;
- `clone` / `superego` (future) — writable side sessions bound to the main path's decisions
  (the comment already names them).

So the plugin gains a **registry of session classes** keyed by preset id, each with its own
tool policy and its own one-time instruction. The explorer gets its own preset; the domain
prompt still comes from the branch's file.

**Open:** whether a case can reuse dsh's own subagent machinery instead of a plain
`fork_into`. Not settled; does not block the first slice.

## Lifecycle of one dispatch

1. `dispatch(name, task)` reads the spec from the branch directory; empty instruction → error
   (a draft is not dispatchable).
2. Fork the ego session at the current step edge into `project_home` (cwd). Policy: read-only.
3. Build the container — `emanation-reminder` + `emanation_position` + the branch's
   `EXPLORER.md` content as the instruction + the task.
4. `run()` to `turn/end`; take the last assistant text as the answer.
5. Commit to the explorer's branch: title = the question, body = the answer, metadata as above.
6. Deliver the answer to the ego as one signal; archive the session (`archive_session`) — the
   log stays readable by id.
7. A failed run (no text, or a tool call where no executor stands) is content: `<error>` in
   the delivery, and **no commit** — an empty case must not enter the log.

## Not doing

- Write access. Not for the explorer, not "temporarily to fix something". An explorer that
  can write is a side effect nothing accounts for.
- Adding tools to the dsh plugin this phase. `bash` + `read`/`grep`/`glob` +
  `web_search`/`web_fetch`, with a read-only sandbox, is already unbounded: the branch's
  `EXPLORER.md` can point at bash skills (`moss codex`, `moss features`, `rg`, `git log`).
  thirdgerb's line: *"bash 现在在无状态交互上, 已经远远超过了 plugin / mcp / ctml 这些. ctml
  核心服务于有状态交互."*
- Linear cognition. The explorer does not read its own past answers; the branch is an
  append-only log, not a memory.
- Nested explorers (an explorer dispatching an explorer).
- Git-versioning the spec. The branch is the version; the file is edited in place, and
  `spec.sha` is the post-hoc evidence of what was in force.

## Open

- `test` / `replay` flag name and where it lives (config vs channel gate).
- Whether `new(from=...)` is the right promotion shape, or promotion is `amend_ref` + a new
  branch created from the same template.
- Whether the explorer's own dsh session should also be archived when the ego shuts down
  (leaked live sessions), or only when the case completes.
