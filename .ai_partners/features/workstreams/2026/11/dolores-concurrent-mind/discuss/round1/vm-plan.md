# Concurrent Mind VM — technical plan

Status: **round 1 candidate — superseded.** Sections 0-13, no code. This is the record of what we
thought the machine was on D04 01:11, kept because the reasoning is still the deepest we have on
shape. The current plan is `../../FEATURE.md`: it lists what survived, what was overturned, and in
which round.

Provenance: authored by **moss the Ghost** on 2026-11-04, in dialogue with thirdgerb. This is
the first complete design plan produced entirely by the ghost; thirdgerb has read it and
marked the places that still need a better orthogonal model (§13).

## 0. Shape

One sentence: **the ghost's mind is a process tree over one shared plan graph. The main is
the graph's only writer; a clone is a process forked from an immutable anchor; the
mindflow bus carries the signals that move the tree.**

Three stores:

- **session log (dsh)** — raw turns. An anchor is a `DshSessionRef` (session_id + turn
  interval); it is a coordinate, not a snapshot.
- **memento** — branches / commits / notes. The execution store; addressing by
  `{branch_index}-{seq}`.
- **plan** — the graph of nodes / edges / ops. The plan store; every view is a projection.

Three runtimes:

- **main** — the interaction loop; plans (writes ops); owns the body.
- **scheduler** — the main's hand; folds the plan, computes the ready set, spawns and
  cancels, backprops results.
- **clone** — a process; `reply` (one-turn, no session) or `session` (own dsh session +
  branch + workspace).

## 1. Invariants

I1. **Anchor immutability.** A node's context is seeded from an immutable anchor (a dsh
turn interval or a memento coordinate). The anchor never moves; the prefix is cached.

I2. **The graph is the plan.** A decision is an op; the graph is `fold(ops)`; no decision
lives only in a context.

I3. **One writer per branch.** memento's write lock. A clone writes its branch and its
workspace only. The main is the only writer of the plan graph; the scheduler acts as its
hand.

I4. **Ready = closed.** A node runs only when all five hold:
deps bound, claim free, trigger fired, accept known, brief compiles.
The three semantic conditions: context sufficient, process compressible, observation clear.

I5. **A run is a projection.** A running process is a projection of its node. A patch that
invalidates the node cancels the process, discards the result, and records `cancelled`.

I6. **No text for state.** Graph maintenance uses ops only. Natural language is material
(task labels, node bodies) and the human surface — never the protocol.

I7. **The human has two veto points.** Every node is visible before it runs (the graph
view); every side effect has a card (the action). The graph is the pre-veto, the card is
the in-flight veto.

I8. **Every op is a delta.** Ops are append-only and ordered; the fold is deterministic;
`diff(rev)` is a projection delta; a view is never the truth.

## 2. Data model

### 2.1 Plan graph

`G = (N, E)` plus an append-only op log `O`.

Node:

```
Node = {
  id,                     # stable identity (kebab, unique in the plan)
  kind,                   # main | clone | reply | tool
  anchor,                 # DshSessionRef | CommitCoord — immutable
  task,                   # declaration: label + body (material)
  deps: [edge_id],        # inbound questions
  when,                   # trigger: event | commit | time | manual | dep-bound
  accept,                 # schema + predicate — the observation point
  budget: {cost, value, depth, time},   # cost estimate; value = priority; depth = max levels
  status,                 # pending | ready | claimed | running | done | failed | cancelled | parked
  branch,                 # memento branch ref (optional)
  workspace,              # path (optional; strict orthogonal, no nesting)
  claim: {worker, until}, # virtual loss (scheduler-owned)
  result: {exit, out, err}  # exit: done|failed|blocked|needs_input; out: value/ref; err: trace ref
}
```

Edge:

```
Edge = {
  id, from, to,
  kind,                   # dep | question | data
  status,                 # bound | unbound   (unbound = a question)
  value_ref               # the answer's ref when bound
}
```

Op:

```
add_node(node) | link(edge) | patch(node_id, fields) | claim(node_id, worker, ttl)
| release(node_id, result) | backprop(node_id, value) | cancel(node_id, reason)
| compact(branch) | confluence(branch, message)
```

Fold: deterministic, rev-ordered; `diff(rev)` gives what changed; the d3 view is one
projection, never the truth.

### 2.2 Mapping to memento

- **Branch = a node's execution store.** A clone's commits are its checkpoints.
- **Fork = reference, not copy.** A clone's branch forks from the anchor's commit (or
  starts root for an anchorless task); the parent's and the child's `commits.jsonl` splice
  into one trajectory.
- **Coordinate** `{branch_index}-{seq}`. Addressing practice: `name` for humans and
  clones (stable label), `id` for storage and internals, `index` for display only, never
  persisted.
- **Confluence (reintroduced).** On archive, the branch writes
  `confluence/<branch_id>.md` into the parent; the parent's next commit absorbs it
  (atomic rename) and the commit view lists it. The archive message is the confluence.
- **Compaction is callable.** `compact(branch)` = commit the span + start a new epoch;
  the old session stays warm as a standby; the branch's active ref moves. Identity
  (branch id) never changes.

### 2.3 Mapping to mindflow

- A clone's **nucleus** emits events: `clone.started`, `clone.progress`, `clone.final`,
  `clone.needs_input`.
- Signals carry `target` (`branch id` or `@name`); routing is the bus's job. The main sees
  only addressed events + a registry delta — clone-to-clone traffic never enters the
  main's impulse queue.
- **Commit signals** carry `{coord, title}`; subscribers inject only new titles (their
  last-seen coord lives in their own branch).
- A clone may **raise an impulse** (BACKGROUND by default); the main's attention decides.

### 2.4 Plan language

A compilable Python subset:

```python
async def research(topic: Ref, depth: int = 2) -> Report:
    """collect sources; accept: >= 5 sources, each with url and claim"""
```

Compile:

- `research` = node id; `topic`/`depth` = deps (edges); docstring = accept; `-> Report` =
  the output type; `async` = parallelizable.
- `await` / `gather` = joins; `main` = the current session.
- The body is declarative (the clone fills it) or a composition of calls.
- The compiler rejects hidden state: only params, refs, calls, await/gather, no globals,
  no side effects outside declared deps.

The `split` tool (the discipline gate):

```
split(task, ref) -> {"kind": "single"} | {"kind": "graph", "ops": [...]}
```

Gated by the three conditions; conservative — `single` only when a condition fails.
The state machine calls it first in every planning cycle.

## 3. Protocol

### 3.1 Operations

```
op                    actor                 precondition -> effect
add_node(n)           main | human          id unique -> node pending
link(e)               main | human          endpoints exist -> edge, deps updated
patch(n, fields)      main | scheduler | human  -> invalidates a running node (I5)
claim(n, w, ttl)      scheduler             ready && claim nil -> claimed, virtual loss
release(n, result)    scheduler             claimed -> status from exit; backprop
cancel(n, reason)     main | scheduler      -> signal the clone; discard; cancelled
compact(branch)       main                  -> commit + new epoch; active ref moves
confluence(branch,m)  clone | main          -> parent inbox; absorbed at next commit
split(task, ref)      main                  -> single | ops
dig(claim)            main | clone          -> evidence | gap (depth protocol)
```

### 3.2 Node state machine

```
pending --(deps bound & when fired & accept known)--> ready
ready   --(claim)--> claimed --(spawn)--> running
running --(release done)--> done
running --(release failed)--> failed
running --(patch invalidates)--> cancelled
pending/ready --(human park)--> parked --(human resume)--> pending
done/failed/cancelled --(archive)--> archived (branch kept, ref dropped)
```

### 3.3 Claim (virtual loss)

- `claim(n, worker, ttl)` appends a claim op; the scheduler's selection sees the node as
  taken (virtual loss), so other workers pick elsewhere.
- TTL expiry: the claim becomes re-claimable; the stale run is cancelled by its own
  signal timeout.
- One claim per node; the scheduler is the only claimer.

### 3.4 Spawn

```
spawn(node):
  brief = compile(node)        # anchor + bound deps' out + task + accept + budget
  if node.kind == reply:
      result = bypass_run(anchor, brief)          # one turn, no session
  else:
      branch = memento.fork(anchor)               # or create root
      session = dsh.session_seed(anchor, brief)   # verbatim prefix + brief
      run task loop; checkpoint at state changes; emit clone.* signals
  release(node, result)
```

The clone's trace is its branch; the parent reads the ref on demand (I6).

### 3.5 Cancel

A `patch` that changes deps / accept / task / status of a running node:
scheduler emits `cancel` to the clone's nucleus; the clone interrupts; its branch keeps
its commits; the node becomes `cancelled`; a re-issue is a new claim when ready again.

### 3.6 Compact

`compact(branch)` = the model calls it when the span is worth a checkpoint:
commit (note or summarizer) -> new epoch -> the recap carries the fold -> the old session
stays warm. Compaction is not automatic; it is a callable capability.


## 4. The main's loop

Each turn:

1. A moment arrives; read the plan diff (`diff(rev)`) and the registry delta.
2. `split(task, ref)` — the gate. `graph`: write the ops. `single`: continue.
3. Speak / act — the interaction language (direct actions, cards).
4. Write ops for new or changed nodes; the scheduler reacts to `plan.changed`.
5. Commit when a checkpoint is worth it; `compact()` when the span is worth it.

The main never spawns; it only edits the plan. The scheduler spawns.
The discipline is structural: the state machine's first planning step is `split()`;
there is no path to serial work that bypasses it.

The main's context holds a **projection** of the plan (ids, statuses, values, edges), the
interaction material, and refs — never the full graph, never the raw node bodies of
others.

## 5. Scheduler

Hosted as a **mindflow nucleus** (`SchedulerNucleus`) plus a debounced tick task. It never
busy-loops; it coalesces.

Inputs: `plan.changed`, `clone.final`, `clone.started`, claim timeouts.

```
tick():
  g = fold_incremental(ops)              # O(new ops)
  ready = {n | deps bound && when fired && claim free && accept known && budget ok}
  batch = select(ready, k)               # k = min(workers, budget)
  for n in batch: claim(n, self, ttl); spawn(n)
```

Selection v1:

```
score(n) = value(n) / max(cost(n), eps) * (1 + unblocks(n)) + c * sqrt_log(total) / (1 + visits(n))
```

`unblocks(n)` = how many nodes become ready when n completes (graph centrality on the
folded graph). UCT supplies exploration; value/cost supplies exploitation.

Budgets:

- per-node: `budget.cost` (tokens/time), enforced by the clone;
- per-tick: `k` claims and a token ceiling;
- per-plan: a total budget; when exhausted, only human-addressed nodes run.

Invalidation:

- a patch to a running node (deps / accept / task / status) -> `cancel(n)`;
- cancelling an ancestor cancels its descendants (the subgraph is invalid);
- a cancelled node's result is discarded; its branch keeps its commits.

Backprop:

- on release, `value(n) = accept(result) in [0,1]`; `visits += 1`; the delta propagates to
  ancestors (a node's value is the mean of its descendants' accepted values);
- accepted results bind the out edges; dependents become ready in the next tick.

Crash recovery:

- the scheduler is stateless: on restart it folds the log, expires claims older than `ttl`,
  and marks running nodes whose sessions are gone as `failed` (or re-queues them when
  `resume: auto`).

Invariants:

- never spawn a non-ready node; the claim op is the atomic point;
- one claim per node; the scheduler is the only claimer;
- a tick is idempotent: re-running it on the same fold spawns nothing new.

## 6. Clone runtime

Two modes:

**reply** — one-turn bypass: `bypass_run(anchor, brief)`. No session, no branch, no
workspace. Output `{exit, out, err}`; a timeout is `failed`.

**session** — own dsh session + memento branch + workspace:

- branch: forked from the anchor's commit (or root for anchorless tasks);
- seed: `seed_from_log(anchor)` — the verbatim prefix — plus the brief as the first user
  message;
- workspace: `workspaces/<node-id>`; strict orthogonal, no nesting; writes confined;
  reads broad;
- checkpoints: commit when the state changed (the per-idle check); the final answer is an
  implicit checkpoint;
- resume: `resume: auto|manual` in the node; auto re-claims and re-seeds from the branch
  tip on ego restart.

Brief (compiled, never written by the main):

```
brief = {node, parent, task, accept, anchor, deps: [{edge, out_ref}],
         workspace, budget, rules {write_scope, proxy}}
```

The clone knows it is a clone: the brief says so; the shared ground/personality is
read-only; its loop is a worker's (`goal -> plan -> tasks -> per-idle`), not the
interaction loop.

Tool surface:

- read-only MOSS channels (facade, files, search);
- workspace writes;
- boundary-crossing writes are **proxy intents** `{op, path, diff, reason}` sent to the
  main; the main evaluates and executes as its own action (a card) or denies. No approval
  loop for the human; the main is the arbiter.

Lifecycle:

- bound to the ghost: shutdown cancels in-flight clones; their branches keep their commits;
- states: claimed -> seeded -> running -> finalizing -> released | cancelled;
- on release the result triple goes to the scheduler; the branch stays; archive later.

Signals: `clone.started {node, branch}`, `clone.progress {node, coord, title}`,
`clone.needs_input {node, question, ref}`, `clone.final {node, exit, out, err}`.

Anchor resolution: `anchor: "@tip"` resolves at spawn time (the current tip of the
parent's branch); a fixed coordinate pins a historical state.

## 7. Memory

- **Checkpoint cadence.** Commit when the state changed; the per-idle check compares the
  task state. The final answer is an implicit checkpoint. Compaction is callable, never
  automatic.
- **Compact.** `compact(branch)` = commit the span + new epoch; the recap carries the
  fold; the old session stays warm; the active ref moves; the branch id is the identity.
- **Confluence.** On archive the branch writes `confluence/<branch_id>.md` into the
  parent; the parent's next commit absorbs it (atomic rename); the commit view lists the
  confluences. The archive message and the final answer are the same artifact.
- **Addressing practice.** `name` for humans and clones, `id` for storage, `index` for
  display only. A plan op that cites a commit stores the coordinate, not the id.
- **The main's commits** carry `metadata.plan_rev`: each main commit pins the plan state
  it saw; the plan log stays separate but recoverable from the commit.

## 8. Communication

Common schema: `{name, target, source, priority, messages, metadata}`.

| signal | payload | target |
|---|---|---|
| `clone.started` | node, branch | main |
| `clone.progress` | node, coord, title | main |
| `clone.needs_input` | node, question, ref | main |
| `clone.final` | node, exit, out, err | main |
| `plan.changed` | rev, cells | scheduler |
| `commit` | coord, title, branch | subscribers |

- **Routing.** `target` = branch id or `@name`; the bus routes. The main subscribes to
  `@main` plus a coalesced registry delta; clones subscribe to their own node plus
  `commit`. Clone-to-clone traffic never enters the main's impulse queue.
- **Commit signal.** Emitted by the memento hook; subscribers inject only new titles; the
  last-seen coord lives in their own branch.
- **Flood control.** Per-source rate limit; hop limit; the main's registry delta is
  coalesced (one summary per tick).
- **Impulses.** `clone.final` / `clone.needs_input` may raise an impulse (BACKGROUND by
  default) carrying the result ref, not the full text; the main's attention decides.

## 9. Surface

- **Plan view.** A projection of the plan (ideas-style graph): nodes colored by status,
  sized by value; edges as deps/questions; claims as rings; batches highlighted. The
  human's edits (flag, weight, drop) are ops like any other.
- **Diff per turn.** The main shares the plan diff since the last turn (a card or notice);
  parallelizable nodes surface there.
- **Veto.** Pre-run: the node is visible; the human flags or denies before it is claimed.
  In-flight: the action card. The graph is the pre-veto; the card is the in-flight veto.
- **Warm data.** Clone progress rides the channel notice; the main reads state, never
  polls.

## 10. Economics

- **Node budget**: cost (tokens/time), value (importance), depth (max levels), eta.
- **Batches**: nodes sharing a blocker (`blocked_by`) form a batch; the idle hook computes
  "building the blocker unlocks N nodes".
- **Idle accounting**: on idle, recompute ready, reorder pending, prune stale, and surface
  a candidate (knock/notice) with its value/cost and the batch it unblocks.
- **split/dig budgets**: split is cheap (small context, low effort, cached `single`);
  dig's depth is budgeted by the claim's value.
- **Metrics**: ready utilization, tokens per accepted node, wasted spend on cancelled
  nodes, plan growth, human corrections.

## 11. Acceptance

Invariants as tests:

| invariant | test |
|---|---|
| I1 anchor immutability | a re-seed from the same anchor is byte-identical |
| I2 graph is the plan | the same ops fold to the same graph (deterministic) |
| I3 single writer | two concurrent commits on a branch: one wins |
| I4 ready = closed | a non-ready node is never claimed or spawned |
| I5 run is a projection | patching a running node cancels it and discards the result |
| I6 no text for state | every graph change is an op; no prose op |
| I7 human veto | a node can be flagged or denied before claim |
| I8 delta | replay from the log reproduces the graph and the diff |

Demo: the main splits "survey the ACP ecosystem" into three nodes; two clones run in
parallel; one is cancelled by a patch; results merge with backprop; a confluence is
absorbed; the human flags one node before it runs.

Failure injection: crash/restart (claims expire, branches survive), orphan clone, anchor
unavailable, duplicate id, log corruption (the fold fails loudly).

## 12. Build order

Vertical slices; each is demoable.

1. **Plan store** — op log, fold, projection, diff. No execution.
2. **Reply clone** — spawn a node, run a one-turn bypass, release. The smallest end-to-end.
3. **Scheduler** — ready set, claim, spawn, backprop.
4. **Session clone** — branch, seed, workspace, checkpoints.
5. **Invalidation** — cancel propagation.
6. **Communication** — signals, addressing, commit signals.
7. **Surface** — plan view, diff, human ops.
8. **Compaction + confluence.**
9. **Economics** — budgets, batches, idle accounting.
10. **split / dig** — the discipline gates.

The first vertical slice is (1) + (2): a plan with one reply node, executed and merged.

## 13. Decisions and remaining risks

Decisions made here:

- the plan store is one log per ghost root plus one per clone branch (sub-plans); the root
  log is the main's;
- the scheduler is a nucleus (bus-facing) plus an idle hook (accounting);
- a clone proposes plan patches; the main or the scheduler applies them;
- human ops share the same log and are attributed;
- the plan view is a projection written for the ideas/d3 surface, never the truth.

Remaining risks and mitigations:

- **split says `single` falsely** -> the human flag; a periodic audit (a reflection-style
  check) over nodes that ran serially and took long.
- **graph explosion** -> depth and breadth budgets; stale pruning on idle; a hard cap on
  live nodes.
- **claim TTL vs long runs** -> progress signals refresh the claim; a heartbeat.
- **scheduler in the ego event loop** -> the tick is async and bounded; heavy folds run in
  a thread.
- **anchor drift** -> `@tip` resolves at spawn; a pending node never holds a live cursor.
- **the main's serial bias** -> `split` is a structural gate; the graph diff is visible
  every turn.

Status: **round 1 candidate — no code written, not the current plan.** The live plan is
`../../FEATURE.md`.
