---
title: co_playwright — supervised playwright node
status: in-progress
priority: P2
created: 2026-09-21
updated: 2026-09-21
depends: []
milestone:
description: >-
  Sibling of nodes/browsers/playwright with a web surface that streams every
  exec as a frame and gates page-navigation on human accept.
---

# co_playwright

> Use `moss features set-status co-playwright <status> -m "note"` to update state.
> Node lives at `nodes/browsers/co_browser/` (short path — the "playwright"
> tag lives on this workstream, not on the directory).

## Motivation

The `playwright` node lets a model drive a live browser through arbitrary
Python. Trust is binary — either the human lets the model use it, or they
don't. Between those two extremes there is a room where the model does
useful work while the human keeps an eye. That room needs a shared
surface. Without one, the human's only lever is "don't give playwright to
the model at all"; with one, they can leave it on because they can see.

Framing (from the human collaborator): "当模型做操作时, 我无法审计, 无法
理解时, 就倾向于不让模型用 playwright. 倒过来能看到, 能对话, 这个问题就
消解了." Watching **is** the mechanism.

## Key Decisions

**Sibling node, not a wrapper.** `co_browser` runs its own `ModuleEval` +
its own headed browser. It does not proxy to the existing `playwright`
node over the matrix bus. Reason: two nodes both owning a browser is
cleaner than one node stamping approvals on another's actions; and it
lets `playwright` stay as-is for headless / unsupervised use.

**Approval gate scope: none. Observation replaces consent.** The first draft
gated page-open calls behind a human accept. That was removed on the
second pass — the human's own framing is "audit is what dissolves the
trust question, not consent". Rejected alternatives:
- *Approve page-open only* (static regex on `goto`/`new_page`/`new_context`)
  — shipped, then cut. It made the human a gate at exactly the moment
  they had the least information (before seeing anything). Now they see
  the result and can turn the whole thing off.
- *Approve every exec* (like terminal cards) — terminal assumes a discrete
  shell line. Playwright exec bodies are Python; approving every one
  drowns the human.
- *Instrument the domain* (wrap `goto` inside `domains/playwright.py` so
  it phones home) — breaks module_eval's "domain source IS the boundary"
  invariant and creates a mid-execution pause the sandbox subprocess has
  no story for.

**The only fine-grained control is a master switch.** One flag on the
store, wired into every command's `available=`. Flipping it off removes
the commands from the model's interface entirely (verified: the facade's
`<interface>` block disappears, calls fail with "command not found").

**`available` does not threaten the stateful runtime — verified, not
assumed.** The concern raised: if a channel wraps a live subprocess,
`available=False` might tear the runtime down and lose all accumulated
state. Measured on the mesh path:

- Browser + module subprocess: **survive** both the toggle and `clear`.
  After off→on, `page.url` was still `https://example.com/`.
- Code: `_is_available()` (`py_channel.py:928`) is a pure predicate over
  `main_state.is_available()`; `own_commands()` returns `{}` when false.
  The runtime lifecycle is bound to `chan.build.startup`/`close`, not to
  `available`.

So there was no capability loss. **The real criterion is not "is the
channel stateful" but "is the runtime lifecycle bound to `available`"** —
here it isn't.

**`clear`/`interrupt` semantics — measured.** Both go through
`clear_children` → `ChannelTree.clear` → `clear_own` + recurse. What that
does, confirmed empirically:

| Layer | Result |
|---|---|
| Browser / module subprocess | **untouched** — `page.url` unchanged |
| In-flight channel-layer task | **cancelled** (`cancelled: 1`) |
| Code already running in the child | **runs to completion** — an `aexec`'s `print` landed in `history` after the clear |

`clear` is an execution-state contract: it cancels the queue, running
tasks, and scopes. It never touches lifecycle, so it never touches
external instances. Lifecycle belongs to `chan.build.close` and the
subprocess owner (`matrix.processes`).

One real gap this exposed: `exec` (blocking) awaits `eval.exec(...)`, so a
clear cancels that await and the result can never come back even though
the child finishes the work. Handled by catching `CancelledError` in
`_run_frame` and settling the frame as `error` with a `[cleared]` note
rather than leaving it stuck in `running` forever.

**Visual: newest frame is pinned to the top.** The list is newest-first,
but a human reads a newest-first list the way they read a chat log — and
the eye goes to the bottom, which here is the *oldest* frame. Fixed with
a green "latest" divider above the newest frame, an "older" divider
below it, a glowing dot on that row, and an accent outline on the card.
Anchoring beats a label: the problem wasn't ambiguity about which was
newest, it was the absence of a stop.

## Implementation Notes

**Separate `.venv` (follow-up).** `nodes/browsers/co_browser` has its
own `pyproject.toml` + `.venv`, mirroring how `nodes/browsers/playwright`
is set up today. But `nodes/tools/` and `nodes/visions/` show the
group-shared-venv pattern (one `pyproject.toml` at the group level, all
sibling nodes share `.venv`). For heavy deps like chromium (~200 MB),
`nodes/browsers/` should adopt the same pattern — consolidate playwright
+ co_browser + future browsers under `nodes/browsers/pyproject.toml`
and delete the per-node venvs. Left for a follow-up commit; the human
flagged it while co_browser's venv was already downloading.

**aexec + gate.** `aexec` gated frames don't fire until accepted, but
the receipt returns immediately with a note ("awaiting human accept").
The result lands in `eval.history()` when it eventually runs — the model
reads it there. This preserves fire-and-forget semantics from the
caller's side.

**Denial returns text, doesn't raise.** `exec` returns
`"[co_browser #N] page-open denied by human"` rather than raising. This
matches how `module_eval_channel` handles other soft failures and keeps
the model's mental model simple: `exec` returns a string, always.

**Frame store trim.** Only settled frames are evicted (mirrors terminal).
A running frame at position 0 will not be dropped even at the cap.

## Files

- `nodes/browsers/co_browser/NODE.md` — model-facing description
- `nodes/browsers/co_browser/main.py` — Matrix entry point
- `nodes/browsers/co_browser/domains/playwright.py` — headed browser domain
- `nodes/browsers/co_browser/src/ghoshell_co_browser/frame.py` — Frame + FrameState
- `nodes/browsers/co_browser/src/ghoshell_co_browser/store.py` — FrameStore + master enabled flag
- `nodes/browsers/co_browser/src/ghoshell_co_browser/channel.py` — nav regex + channel builder
- `nodes/browsers/co_browser/src/ghoshell_co_browser/surface.py` — WS server + snapshot
- `nodes/browsers/co_browser/index.html` — one-file frame stream + toggle
