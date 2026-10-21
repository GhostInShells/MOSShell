---
name: service-operator
description: Understand and extend MOSS's cell-level capability wiring — how a page, device surface, or data source becomes a discoverable service on the Matrix mesh without inventing a new wiring path.
moss_version: beta2
verified: 2026-10-17
author: deepseek-flash
evidence: The webview service kind ran in live streams from 2026-09-19 through 2026-10-17, served by terminal, file_editor, camera, push, artifacts, co_browser and dsh, and consumed by the screen node. Zero-context validation on 2026-10-17 — a fresh agent given a live-metrics proposition and this skill alone landed on a new kind with query-snapshot plus pub-stream, identity/state split, and subscribe-before-pull.
---

# Matrix Service — the operator and the kind

## 1. The problem

A process in the mesh has a capability other processes and the ghost need to use: a page a
human should see, a device surface, a stream. Channels solved this for commands, but every
new capability shape drifted toward its own wiring — resources had their own path, webviews
had none. Adding a capability meant going back into Matrix and soldering another path, and
the knowledge of how to do that lived in the kernel rather than in a surface a model could
read.

The real question: what is the smallest uniform contract any cell-level capability can stand
on, so that discovering, addressing, and following it do not depend on what the capability is?

## 2. The motivation

Two facts about the mesh drive the answer:

- It is broadcast and full-duplex. There is no central broker to register with, and no
  component holds a global view of the mesh.
- A capability has two irreducible communication shapes: "ask this specific thing and get an
  answer" and "tell me whenever this changes."

So the uniform layer is not a protocol. It is two primitives plus a declaration of identity.
"Unify the boring layer, specialize the interesting layer": the operator owns addressing,
discovery, lifecycle and the async bridge, so no capability re-solves them; the kind owns the
meaning of the payload.

The second motive is the implementer's surface. A model should be able to add a capability by
reading the abstraction and writing ordinary async handlers — not by learning the transport's
threading model.

## 3. The solution shape

- **Two primitives.** Query is addressed request-reply; it also covers fire-and-forget when
  the caller does not await. Pub/sub is the streaming shape. Every capability is a composition
  of these. A "send/listen" pair is not a third primitive.
- **Three server slots, three client operations.** Serve side: queryable (answer), pub
  (stream out), listen (receive events). Client side: get, sub, emit. That symmetry is the
  whole wire surface.
- **Four roles, one switchboard.** Declaration says what the capability is. Provider is its
  serve-side runtime — the slots. Server wraps lifecycle and owns business handlers. Client
  wraps the consumer lifecycle and the state it tracks. Operator is the switchboard: provide,
  discover, get/sub/emit, lifecycle callbacks. The operator never knows a business key.
  Matrix exposes it through a lazy accessor plus lifecycle helpers (serve and connect), so a
  node or consumer asks the Matrix rather than constructing the operator itself; exact entry
  points and signatures come from interface reflection.
- **Identity is data, not a registry.** A kind is a self-describing declaration: it knows its
  own kind, projects itself into meta, and reconstructs itself from meta. Discovery is a
  closed loop — any meta must validate back into the declaration. Failure is a kind bug, not a
  missing registration. There is no central registry.
- **Identity and condition have different lifetimes.** A declaration is frozen when the
  provider is created and served to every discoverer forever. Anything that changes while the
  capability lives belongs in a condition payload, not in the declaration; otherwise it is
  stale for the identity's whole life.
- **Address is the anchor.** The cell address identifies the service; business keys identify
  slots within it. Transport key expressions are derived inside the operator and never leak to
  implementers.
- **Discovery has two directions.** Push: start/stop callbacks. Pull: enumerate live services
  by kind or address. A consumer that needs both a snapshot and a stream should subscribe
  first, then pull, so a change between the two is not lost.
- **Concurrency is the implementer's contract.** Handlers may be invoked concurrently. A
  handler with await points and shared state owns its own lock. The operator isolates failures
  and replies with errors; it does not serialize work on the implementer's behalf.
- **The async bridge is inside.** Transport callbacks run on foreign threads; the operator
  turns them into loop work. Implementers write async functions and never touch those threads.
- **What the mechanism deliberately does not decide.** Layout, ordering, rendering, priority,
  and "who is watching" are consumer policy. Channel wiring is not backfilled onto this
  mechanism, and nested liveness is not part of this layer. These are open decisions, not
  missing pieces to patch inside a kind.

The recognition rule: if adding a capability makes you want to add another wiring path to
Matrix, the capability probably needs a kind on this mechanism instead.

## 4. Where the framework already explains it

This is not a new surface to learn — the polished abstraction already carries it:

- `moss codex blueprint matrix` — the cognitive map: cells, bus, communication, discovery.
- `core/blueprint/service.py` — the ABCs and the exact contract, with zero transport
  dependency. Interface reflection is the authority.
- `moss codex get-interface ghoshell_moss.core.blueprint.service` and `.matrix` — signatures
  and docstrings, current by construction.
- `channel_builder` — the contrasting case. Knowing where the command channel stops is what
  makes the service boundary legible.

Read those first. This skill adds only the shape that is easy to miss: two primitives, four
roles, the identity/condition split, and closed-loop discovery.

## 5. Reference directions

- **Reference kind:** `src/ghoshell_moss/services/webview/`. Look at how declaration, server,
  and client divide; how condition is pulled once and followed thereafter; how the server's
  own transitions are methods rather than wire messages.
- **Openbox producers:** `nodes/os/terminal`, `nodes/os/file_editor`, `nodes/visions/camera`,
  `nodes/visions/push`, `nodes/webview_apps/artifacts`, `nodes/browsers/co_browser`, and the
  dsh runtime. Look for where the declaration is created at node startup and which real
  moments trigger state changes.
- **Openbox consumer:** `nodes/screens/screen_manager` — it discovers the kind, adopts live
  views, and keeps policy (where things go) outside the mechanism.
- **Executable acceptance:** start the screen and any producer. The screen should report the
  view on its own, with nobody opening a URL. If it does not appear, check discovery and the
  declaration round-trip before suspecting transport.

## 6. Version and identity

- Verified against MOSS beta2 on 2026-10-17.
- Author: deepseek-flash.
- Evidence: the webview kind served by seven producers and consumed by the screen node across
  live streams; no unit test substitutes for that run.
- Stale check: re-run interface reflection. If the operator/provider/client ABCs changed,
  re-read before trusting this shape. If the executable acceptance no longer happens, update
  or prune this skill.
