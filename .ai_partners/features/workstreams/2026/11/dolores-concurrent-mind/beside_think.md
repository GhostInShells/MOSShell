---
name: beside-think
description: The first beside slice — ghost.beside.think, per-step issuance, off-record directions, and the delivery-nucleus bug it exposed
---

# beside think

> **Status:** design settled 2026-11-16; the delivery nucleus, the `memento.chat` reroute,
> the three-slot delivery fix and the beside container's position are committed and accepted
> (5/5 concurrent chats, D17 00:01). **The beside slice is implemented** (D17 00:2x): the
> `step/end` lifecycle hook, `EgoBesideMindConfig`, `BesideRuntime`, and the
> `ghost.beside` factory gate. Not yet exercised on a live shell — acceptance needs a restart.
> This is a construction record, not a discussion. The workstream's `FEATURE.md` is the index.

## What it is

`ghost.beside.think` is the first **`beside` timing primitive**: the ego keeps interacting,
and a group of pure-thinking directions runs in parallel on the same session context.

It is not:

- a subagent / explorer — no tools, no read, no CTML;
- a background reflection (thinking frame) — it is issued by the ego and its result is
  awaited as a signal;
- checkout — the ego does not move.

The timing model (from the five-mechanism discussion): `now` = think yourself; `beside` =
issue now, start at the next step, return as a signal. `after` / `ahead` / `maintained` /
`elsewhere` are later mechanisms, not this slice.

## Command surface

```
ghost.beside.think(text__, sep="||") -> beside-think #k: N direction(s)
```

- `text__` holds the directions, separated by `sep` (`||` by default); each direction gets
  one off-record call.
- **No `label`.** The returned `seq` is the group identity; the result carries the same `seq`.
- The receipt is **plain text** — `beside-think #k: N direction(s)`, plus
  `, M dropped by the cap` when the cap bit. It carries the parsed count on purpose: a wrong
  separator is otherwise a silent success (see *Live run*).
- Result shape:

  ```
  <beside-think seq="k">
    <direction i="1">...</direction>
    <direction i="2">...</direction>
    <error i="3">...</error>
  </beside-think>
  ```

- Docstring: positive (interaction continues, deep thinking beside), negative (if you can
  focus now, think directly — do not use this), constraint (do not re-think the same
  directions; wait for the signal).
- Cap the direction count (2–4) and per-direction output.

## Lifecycle

- Module lives under `ghosts/dolores/mindset/beside/` (channel + runtime).
- The channel is bound by the shell; if it is not bound, the runtime does not start.
- On channel startup: the runtime enters and registers `on_step_end` on the ego lifecycle bus.
- On `think`: record a ticket `(seq, directions, issued_at_step)`; return the receipt.
- On `step/end`: consume tickets issued at or before this step; for each ticket, spawn one
  off-record call per direction. The step-end handler is a **fast enqueue/spawn**; it must
  never await the model calls, because the dsh session event dispatch is sequential.
- On completion: build `<beside-think seq="k">` and send a body-carrying ego signal
  (`role='mind'`, `priority=BACKGROUND`, notify mode). It must be queued, not dropped.
- On channel close: discard the hook and cancel in-flight group tasks.

## The dsh edge (verified)

`dsh-agent-loop/lib/index.js`:

- `step/start` after `preStep` (L968);
- `await this.step(decision)` (L1041) runs the model + `executeToolCalls`; `tool/result`
  lands inside the step;
- `finally` appends `step/end` (L992);
- only then does the loop target the next step and call `preStep` / `step/start`.

So `step/end` is exactly the edge where the step's tool result is already in the log and the
next step has not begun. `StepEnd` carries `turn`, `step`, `meta.seq`; the Python side
subscribes with `on_session_event_model(StepEnd, cb)`. The anchor is
`DshSessionAnchor(session_id, seq=step_end.seq, turn=step_end.turn)`; the off-record call sees
the step's tool call/result and the `beside.think` receipt.

The session event dispatch is sequential (`DshSession._dispatch_session_event`): a slow
`step/end` handler blocks the event stream, though not the agent loop. Hence the fast-spawn
rule above.

dsh is not serial: concurrent `chat_off_record` calls are allowed (`follow=False` one-shot
handles, no `session/follow` stream). The 2026-11-16 probe issued 3 and 4 concurrent
`memento.chat` calls; the requests ran, but delivery was lossy — see the next section.

## The delivery-nucleus bug (fix before/with beside)

`memento.chat` sends `new_dolores_ego_signal` → `dolores/ego` → `DoloresEgoNucleus`. That
nucleus's contract is **one-shot wake**:

- `peek()` pops the oldest impulse;
- `suppress()` discards on lost arbitration (`fire-and-forget`);
- `clear()` wipes the queue when the attention ends;
- the timeout failure signal goes through the same path, so failures can vanish too.

Observed 2026-11-16: 3 chats to different refs → 2 answers; 3 chats to the same ref → 2
answers; 4 chats to the same ref → 2 answers. The missing requests either completed or timed
out, but the delivery was dropped — the chats themselves are not the problem.

`NotifyNucleus` is closer (its losing side buffers), but it **merges** bursts into one pending
impulse; N chat answers would collapse into one message list.

**Fix:** a dedicated message-preserving FIFO nucleus, signal names `memento/chat` and
`beside/think`:

- `add_signal`: one independent impulse per answer, no merge;
- `peek`: return the head, **without** discarding it;
- `attended`: remove the head;
- `suppress`: keep the impulse for the next re-rank (optionally with a cooldown), never
  discard it;
- `clear`: only on explicit shutdown, not on attention end;
- `status`: pending count.

**Landed 2026-11-16, accepted 2026-11-17:** `MindDeliveryNucleus` in
`core/mindflow/mind_delivery_nucleus.py` (signal `mind/delivery`), registered in
`matrix/openbox/nuclei.py` and injected by the host from `MOSS.manifests.nuclei` —
`new_default_mindflow` registers no global nucleus at all; `memento.chat` sends
`mind/delivery` for both answers and failures.

The lost frames had a second cause, found by reading the accepted run's session log: the
plugin's `thinking/enter` read two of the three moment slots (`context`, `inputs`) while
`moment/push` read all three, so a `role='mind'` delivery arriving as the first frame of an
attention vanished, while the same delivery mid-turn arrived. Every path now goes through one
`perceptFrame()`, and an enter frame carrying perceptions but no inputs opens a turn instead
of waiting to be carried by the next real input.

A beside container also states its position now: `at` (the ask instant, on the body's clock)
and `coord` (the span asked about). A fork inherits the source view's last clock, so without
them a question asked hours — or months — later reads as if it had been asked back then.
Helper: `mindset/emanation.py::emanation_position`; every beside path uses it.

## Build order

1. ~~Add `on_step_end` to `EgoLifecycle` / `EgoLifecycleBus`~~ — `StepEndEvent(turn, step, seq)`,
   non-blocking, fired by the ego's `step/end` subscription.
2. ~~Re-export `StepEnd` from the seam~~ — `StepEnd` / `StepStart` are on the v0.2.0 face.
3. ~~The FIFO delivery nucleus; `memento.chat` routed to it~~ — accepted 5/5.
4. ~~`BesideRuntime` + `ghost.beside`~~ — ticket queue, `on_step_end` consumption,
   `chat_off_record` fan-out, one `<beside-think seq>` delivery per group.
5. ~~The `beside.think` prompt asset~~ — ghost-home editable; the runtime falls back to the
   module seed when a home has no asset.
6. ~~Tests~~ — registration/discard, ticket consumption at the edge, fan-out, failure content,
   cap, no-session, and the factory's three gates.

### Where it lives

- `mindset/ego_lifecycle.py` / `_bus.py`: `on_step_end` + `fire_step_end` (non-blocking task).
- `mindset/beside/config.py`: `EgoBesideMindConfig` (`enabled` / `directions_max` / `effort` /
  `max_tokens` / `timeout`), materialized in `<ghost_home>/.dolores/configs`.
- `mindset/beside/runtime.py`: `BesideRuntime` — tickets, the step edge, the group delivery.
- `mindset/beside/channel.py`: `EgoBesideChannel.factory` — `IoC → config store → config`;
  disabled (or no runtime built) returns `None`, so no channel, no listener.
- `_runtime.py`: builds the runtime after the ego, binds it (and the config store) into the
  ghost channel for the factory; `channel.py` imports the factory.

## Acceptance

- Three concurrent chats to the same ref all return, including one forced failure; no silent
  drops.
- `ghost.beside.think` with 3 directions: the receipt returns immediately; the calls start at
  the next step end; the group result arrives as one `<beside-think seq>` delivery; the main
  path does not re-think the directions.
- Channel not bound → no `on_step_end` listener, no runtime.
- A slow/failed direction does not block the session event stream; the group reports the
  error and the other conclusions.

## Live run — what the first real acceptance cost (D17 00:5x)

Five directions issued in one `think`, read back as `<beside-think>` deliveries. Four of the
five came back and were substantive; the rest of the run exposed three defects, all of them
mechanism, none of them the model.

1. **A wrong separator is silent.** The first call separated five directions with newlines;
   the runtime split on `|`, saw one part, and queued one group. The input was valid, so
   nothing failed — only `beside think #1 delivered: 1 direction(s)` in the log said so, after
   the fact. Fixed: `sep` is a parameter (`||` by default), and the receipt carries the parsed
   count as plain text — `beside-think #2: 4 direction(s), 1 dropped by the cap`. The receipt
   is read once, to check what the call understood.
2. **The cap was invisible.** `directions_max` defaulted to 4 and dropped the fifth direction
   without a word; finding that out meant reading `config.py`. Fixed: the cap is in the
   channel instruction (read live from the config store on each refresh) and in the receipt.
3. **The tools menu leaks into an off-record round.** 6 of 10 direction rounds came back as
   *"asked to run a tool instead of answering: bash"* — including one whose whole text was
   `只回一个字：好`. The seed asks the emanation not to run tools; an ask is not a switch.
   Clearing tools (`tools=[]`) is the wrong fix: the tool menu is part of the prefix, so
   clearing it invalidates the prefix cache — the very thing an off-record round exists to
   reuse. The failure therefore stays content (`<error i="n">`, what was asked for), and the
   seed keeps the `NEED-WORLD` fallback as the intended shape of the answer.

What the run did establish: the emanation container reads correctly (the four answers each
described their own position, not a misdelivery), the group arrives as one delivery, and the
identity line holds under a question that invites identity confusion.

## Open

- Whether the delivery nucleus is shared by `memento/chat` and `beside/think` or split per
  signal name. One shared FIFO is simpler; per-name isolation is safer if the failure modes
  differ.
- Whether `ghost.beside` is always built or behind a `.dolores.yml` flag. Channel binding is
  the lifecycle; a flag is only cost control.
- The `seq` counter is in-memory; a process restart kills in-flight groups, so there is no
  cross-process collision.
