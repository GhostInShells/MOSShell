---
title: Dolores Reflexive Control
status: in-progress
priority: P1
created: 2026-10-23
updated: 2026-10-23
depends: []
milestone:
description: >-
  The knobs the ghost turns on itself — what model it thinks with, how deep it
  thinks — readable and writable from inside its own CTML surface, with exactly
  one write path.
---

# Dolores Reflexive Control

> Use `moss features set-status dolores-reflexive-control <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

The ghost could not reach its own brain, and the gap was structural rather than
informational.

Its model was fixed at boot: two variables in the ghost home `.env` were asserted into
dsh's `settings.yaml` as the default model, so changing it meant editing files and
restarting. Thinking depth was the only thing it could move at runtime — and even that
had two writers: the plugin, plus a python-side `_push_effort_officially` that went
through `session/selectModel` and silently rewrote the human's global default.

thirdgerb's question was the whole feature: "你自己知道 dsh 上有哪些模型吗". The answer was
no — nothing in the CTML tree exposed models at all. Every claim the ghost could make
about its own capabilities was a claim about source code it had read, not about surface
it actually had.

This workstream is the umbrella: **self-control is a class of organs, and each one is
built the same way** — a warm notice for what is running, a cold read for what is
available, and a single write path at a boundary that belongs to the shell rather than
to the model. The model organ is the first; depth, voice, tone, memory policy follow.

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

- **Read warm, catalog cold, write single.** What the ghost is running is a named notice
  on the `ghost` channel (`model`), re-sent only when it changes — it arrives as a
  facade delta, so no polling. The catalog has to be asked for: `ghost.model:list`
  calls `session/modelCatalog` through the ego's own dsh session. `ghost.model:select`
  writes nothing; the declaration rides the next thinking frame, and the plugin composes
  provider/model/effort into one `selectForNextRequest` at the turn boundary.
- **`selectForNextRequest`, not `selectModel`.** `selectModel` saves the selection as the
  global default; using it for the ghost's own switch would recreate the two-write-path
  ambiguity this feature exists to remove. The human's default is the human's.
- **The ego agent must be created with `agentOptions`.** It is built directly with
  `ctx.agents.create`, not through `ensureSession`, so the plugin's private selection was
  its *only* source of provider and model — not merely a second authority. Deleting that
  block without installing the official path first made the very first request fail with
  "agent has no provider/model"; the patch was rolled back once for exactly this.
- **No new observation surface.** Read rides the existing channel's notice instead of
  opening a status page: "不另开一个观察面，读走温数据".
- **Depth travels with the model.** The declaration carries provider + model + effort as
  one tuple, because a depth the new model does not offer is a coherent unit of failure,
  not something to drop silently.

## Exploration paths

- Attempt #1 (rolled back by thirdgerb): delete the plugin's private `EgoSelection`
  (~110 lines), read the official projection, call `sessionController.selectModel`.
  Failed at boot — the ego agent had no provider/model at all.
- The lesson's shape: "第二份权威" and "唯一来源" were the same object. A duplicate that
  looks redundant from the outside can be the only supply line from the inside.

## Methods

- **Verification is a two-way round trip.** A same-model declaration proves nothing,
  because the notice will not change. The real test is flash → v4-pro → flash, watching
  the `model` notice and the per-turn model notice move each time. Verified 2026-10-23:
  both directions landed, and the ghost reported each switch from its own surface.

## Implementation Notes

- `_runtime.py::_sync_default_model` is where the ghost's *base* model still comes from at
  boot — ghost home `.env` → dsh `settings.yaml` `agent-default-model`. Switching the
  base model remains two lines and a restart; switching at runtime is now the organ.
- Sight is a property of the model, not a channel: the catalog does not report modality,
  so choosing a model without image input turns the eyes off silently.
- Plugin assets are copied into the dsh profile at boot, so plugin changes need a restart;
  python-side changes do not.
- Any frame may carry the declaration, including background frames — the plugin writes it
  before deciding whether the frame drives a turn.
