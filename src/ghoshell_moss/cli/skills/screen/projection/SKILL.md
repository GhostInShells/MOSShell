---
name: projection
description: Understand how MOSS turns cross-process capabilities into one screen — window semantics, automatic adoption, layout, focus, and address-based foreground gestures.
moss_version: beta2
verified: 2026-10-17
author: deepseek-flash
evidence: Daily live use of the screen node in the September–October streams; automatic webview adoption (D09), pool ordering and peek (D11–D12), multi-node staging with terminal, file_editor and artifacts (D13–D14). Zero-context validation on 2026-10-17 — a fresh agent given a build-monitor proposition and this skill alone separated adoption from arrangement, kept the human's view authority, and used addressed veil gestures.
---

# Screen Projection

## 1. The problem

Capabilities live in separate processes, each with its own page. Without a projection layer
the human must open URLs, arrange windows, and remember what is where, while the ghost must
materialize every surface by hand. A screen that is merely a backend does not solve this; it
only moves the problem one layer down.

Two constraints sharpen it:

- The human's window must be findable at a known address and survive restarts. A screen is a
  **place**, not a service.
- Automatically appearing is not the same as automatically arranging. A discoverer that
  quietly decides where things go steals the one move that belongs to someone else.

## 2. The motivation

- **Make running windows visible instead of building a second pipeline.** The pages already
  exist; projection should not recreate their content.
- **Reasoning on stage.** The ghost's own surface is an organ like any node's. If it is not
  announced at boot, "the reasoning is visible" depends on someone remembering to open it.
- **One place, two authorities.** The human owns where they look and whether a node lives;
  the model owns what is materialized, how it is grouped, and in what order.
- **State is the truth.** A projection that is inferred from commands has two authorities and
  will eventually disagree with itself.

## 3. The solution shape

- **Semantic model, not a window manager.** An **item** is anything windowable. A **group** is
  an ordered item list and is what the human sees. An **active group** is the one currently
  shown. **Focus** is a view orthogonal to grouping — one item taking the stage. The
  **pool/desktop** holds material not currently on stage, as portraits only. The **background**
  is visible when nothing else is. The **veil** is the outermost pointing layer. Invariant: an
  item is in exactly one group; the backend maintains it, so the model never has to say "take
  it out of the old group."
- **Layer stack and pointer sovereignty.** background → pool/desktop → scrim → stage → veil →
  rail. The stage is the only layer that receives pointer input. The pool shows portraits,
  never live frames. The scrim must sit below the stage. Getting this order wrong is not
  cosmetic: a scrim above the stage dims and blocks every page.
- **Layout is derived, not authored.** Two families — grid (equal split) and stack (one master
  plus a strip) — plus a direction. Shape is a function of the item count. The model supplies
  an ordered id list, never geometry. Area expresses a role ("this is the protagonist"), not a
  continuous priority; pixel geometry never enters the model's context. The first id in a
  stack is the master pane; focus is the separate full-stage view, and choosing a base view
  (switching group, arranging, returning to the desktop) cancels it.
- **Projection pipeline.** A capability announces itself (the webview kind). The screen
  discovers it, pulls one snapshot, then follows the stream, and adopts it onto the desktop.
  Adoption is not arrangement: the model decides where it goes. A destroy is a tombstone, so
  the next reconciliation does not resurrect it. The store reports adoption as a named notice;
  the model addresses the item by the id that notice assigns, never by its URL.
- **Foreground is pointing, not content.** The veil carries marks, boxes, circles, arrows, and
  text addressed to items and regions — not to screen coordinates, because the screen reflows
  and coordinates expire. Gestures run in parallel and do not occupy the command channel, so
  each declares its own duration and the veil can be read back instead of remembered.
- **One current screen.** Pages self-identify; only the current screen receives output frames.
  The human claims a page through the mask, and a screen the node opened itself can arrive with
  that intent already expressed. Other pages are told they are not the screen, not silently
  dropped. Closing the current screen does not promote another: auto-promotion would decide
  where the human is looking.
- **View changes are state, not messages.** A command that changes which group is shown records
  its provenance; the latest view state wins, and the model does not silently override the
  human's choice. Arranging into a group may activate it as a consequence — that, too, is a
  recorded view change rather than a transient command.
- **State is authoritative.** Named notices carry state deltas. Command returns carry only what
  notices cannot: rejection, failure, truncation, automatic side effects. The viewport is a
  low-frequency fragment, not a per-command payload.
- **Where the mechanism stops.** The screen never decides what a page contains; it points at
  it. It does not infer which group is best, and human navigation through the screen's own
  chrome is a separate, still-settled boundary rather than a model decision.

## 4. Where the framework already explains it

- The node's own cognitive surface: `nodes/screens/screen_manager/NODE.md`, and the channel
  interface through `moss codex get-interface`.
- The window semantics are a channel contract. The command surface is where the model's
  authority begins and ends; the notice layer is where state comes back.
- `moss codex blueprint channel_builder` for the command / notice / context split, and
  `moss codex blueprint matrix` for how surfaces arrive from other cells.

This skill adds only the connection between them: adoption comes from the mesh, arrangement
belongs to the model, and the view belongs to the human.

## 5. Reference directions

- **Boot and projection:** `main.py` (connect the client, start the bridge, provide the
  channel), `bridge.py` (mesh events into the store), `model.py` (invariants and store),
  `surface.py` (page, current-screen claim, broadcast), `channel.py` (the model-facing
  commands).
- **Renderer:** `index.html` — reference direction only. Do not copy its implementation into a
  skill or into another body; the semantic model, not the renderer, is what transfers.
- **Openbox surfaces:** terminal, file_editor, artifacts, camera, push, co_browser, and the
  dsh runtime's boot declaration.
- **Executable acceptance:** start the screen and a page-serving node. Expect a state notice
  and a portrait in the pool with nobody opening a URL; arrange it into a group and it becomes
  a live, clickable frame; point at it with the veil and read the veil back.

## 6. Version and identity

- Verified against MOSS beta2 on 2026-10-17.
- Author: deepseek-flash.
- Evidence: the screen node ran through daily live streams, adopting real node surfaces and
  serving as the human's only development screen; the acceptance above was observed there.
- Stale check: read the channel interface and re-run the acceptance. If the command surface or
  the notice contract changed, re-read before trusting this shape; if the acceptance no longer
  happens, update or prune this skill.
