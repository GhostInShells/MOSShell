---
created: 2026-10-21
depends: []
description: Two live failures on D21 evening — a blocking command that never completes permanently occupies the __main__ queue (voice dies and clear_speech cannot recover), and moss_wait_next cannot yield while percepts keep arriving (the thinking turn never ends). Both need a watchdog that names the culprit instead of a restart.
milestone: null
priority: P0
status: in-progress
status_note: D21 — strong speech clear landed (stream.close always fully closes; wait_played interruptible; clear verifies and logs leftovers) + moss_shell_clear ego tool (bypasses the __main__ queue). The dispatch stall itself is still unidentified.
title: Command Dispatch Stall — 命令卡在 pending 永不执行, 声音跟着一起死
updated: '2026-10-21'
---

# Command Dispatch Stall — 命令卡在 pending 永不执行, 声音跟着一起死

> Use `moss features set-status command-dispatch-stall <status> -m "note"` to update state.

## What happened (D21, 17:13–17:14, from `moss.log` timestamps)

The live session, reconstructed from evidence — this is the only record, the state is gone after the 17:15:58 restart.

1. `17:13:04` I emitted one `<all>` block:

   ```ctml
   <all>
   <matrix.mesh.webview_screen:_> 7 × <box duration=5..7/> </matrix.mesh.webview_screen:_>
   <_> 7 × <say> …one short demo line each… </_>
   </all>
   ```

2. The 7 says played **serially** through `17:13:10 → 17:13:35` (each `start_play` closes the previous
   stream — that part is the documented contract and it worked). `matrix.mesh.webview_screen` idled at
   `17:13:17` with `pending tasks 0`.
3. `17:13:58.771` the next `<say>` was prepared — batch created, text fed, TTS synthesised, audio received
   (`17:14:11.6`). **But the command body never ran:** no `start task`, no `Starting playing TTS stream`,
   no `wait play done`.
4. `17:14:10` — he reports he cannot hear me. `17:14:12` `say replace="true"` prepared a new stream
   (cleared the old one), `17:14:13` `<clear_speech/> + <say>` prepared another. Every one of them was
   **prepared but never dispatched**.
5. At shutdown `17:15:58` the runtime reports the abandoned command:
   `capture command err CLEARED: cleared by runtime at _tree_channel_runtime.py:492` — i.e. it was sitting
   in `_pending_tasks` of `__main__` the whole time, never consumed by `_main_loop`.

So the observable failure is: **`__main__`'s consume loop stops consuming; every later command (including
`clear_speech`) queues forever, so the voice cannot be recovered from inside the conversation.**
"音频崩了" is a symptom of the dispatch stall, not a fault in the speech pipeline.

## The mechanism (from `_tree_channel_runtime.py`)

`_consume_task` (line ~149) does:

```python
if self._executing_blocking_task is not None and not self._executing_blocking_task.done():
    await self._executing_blocking_task.wait(throw=False)
    self._executing_blocking_task = None
```

The consume loop is serial: one never-completing blocking task blocks every later task on that channel,
silently, with no timeout and no log line. `push_task` also has no watchdog.

Candidate culprits at `17:13:35.5`, when the stall begins — two blocking tasks started 4 ms apart:

| task | note |
|---|---|
| `01M5EVRSTHPPX0BKKMT2JNE750` | `__scope_exit__` of the inner `<_>` |
| `01M5EVRSXV60SK0AD8S91E29QN` | `__scope_exit__` of the `<all>` |

Nothing else on `__main__` ran until `17:14:13.881` — and that was only a scope **compile** step
(`open scope` / `commit scope` are done while parsing), not a dispatched task.

## Reproduction attempts (both pass — the trigger is still unknown)

- `tests/ghoshell_moss/default/ctml/shell/test_scope_child_channel_scope.py`
  `<all><child:_><quick/></child:_><_><quick/><quick/></_></all>` — closes fine.
- `tests/ghoshell_moss/default/ctml/shell/test_scope_proxy_child_scope.py`
  same shape with a `create_thread_bridge` proxied child — closes fine.

So "an `<all>` containing a child-channel scope never closes" is **not** sufficient. What the live case had
and the tests do not: 7 commands inside the child scope, 7 say commands in the sibling scope, and real
speech streams whose `start_play` closes the previous stream. The next attempt should add those
(especially the many-commands case, which is what `_sub_task_group` and `wait_close`'s
`asyncio.gather(*wait_group)` are actually stressed by).

## Proposal — make the failure name itself (needs thirdgerb's go-ahead)

1. **Dispatch watchdog.** Keep a timestamp when a blocking task becomes `_executing_blocking_task`; if it
   is still alive after N seconds (30?), log a WARNING with the task's `caller_name()` / `cid` and a stack
   dump of `asyncio.all_tasks()` for the shell loop. One line is enough to turn the next occurrence from
   "restart and guess" into a named task. Without this, every recurrence costs another evening.
2. **Do not let an unclosed task hold the channel forever.** A scope whose `wait_close` outlives its
   children by some margin should be closed with a reason and logged, and the dispatch stall released.
   This must be a contract decision, not a hack — the reason a `<all>` scope exists is to wait, so the
   bound has to be either `timeout` (opt-in) or a governance-level ceiling.
3. **`clear_speech` must not be a queued command.** It is the escape hatch for a wedged speech channel; if
   it queues behind the wedge it is worthless. It should be `call_soon` / out-of-band
   (`_tree_channel_runtime.py:427` `meta.call_soon` already exists for exactly this shape).

## Second failure of the same evening — `moss_wait_next` cannot yield

In the same session `moss_wait_next` returned, three times in a row:

> `a percept arrived before the yield — the moment is pushed and this turn continues instead of ending`

Each time the turn continued instead of ending. The code is `_handle_wait_next` in
`src/ghoshell_moss/ghosts/dolores/_run.py` (~line 713): it calls `_wait_actions_racing_barge_in()` and, if
any percept landed during the wait, returns `cancel=False` with that message — the yield loses the race
and the turn goes on by design ("Unless someone speaks first").

The design assumes percepts are rare and human. With a chatty shell — the stalled command above emits a
status moment roughly every second — the race always loses, so the turn can never close: the model
reasons, a percept lands, the turn re-enters, and from outside the shell looks frozen while tokens burn.
That is the "DSH 那边也卡死了" he saw. Fix direction: keep the barge-in rule for human voice, but give the
yield a hard close path that queues the percept for the next turn instead of cancelling the yield —
priority, or "how many times in a row this yield has been defeated", is the knob to decide.

## Open questions

- Does `_main_loop` survive a blocking task that raises without completing? (line 149 has no timeout.)
- Is `say` a `call_soon` command? If it ever becomes one, line 439–444's `priority=None` path cancels every
  pending blocking task on the same channel — worth checking before it bites.

## What landed on D21 (thirdgerb's direction: "让 speech 的 clear 强势一点" + "ego 加一个 moss_shell_clear 的 tool")

Two changes, both chosen because they hold even when the trigger is unknown:

1. **The speech stream can always be closed.** `TTSSpeechStream.close()` used to early-return
   when synthesis had not started — batch left open, `_closed_event` never set, so `is_closed()`
   stayed False forever and that stream became a ghost holding a reference. Now close always sets
   the closed event, closes the batch, and logs any failure instead of swallowing it.
   `wait_played()` now races `_tts_batch.wait_done()` against the closed event: a `say` pinned on a
   TTS session that never comes back can be freed by `clear()` — before this, clear set the
   play-done event while the command was waiting on something else entirely.
   `BaseTTSSpeech.clear()` now also covers `_last_stream`, reports each stream that failed to close,
   and logs loudly if anything is still open afterwards.
2. **`moss_shell_clear` — the escape hatch that cannot queue.** CTML `<clear/>` is itself a command
   on `__main__`, so in the exact failure it is meant to fix it queues behind the wedge. The new ego
   tool calls `shell.clear()` directly (speech + every runtime own-and-children + stop the running
   interpretation), keeps the turn open, and emits no moment.

Tests: `tests/.../core/speech/test_stream_tts_speech.py` (stuck-on-batch, never-started stream),
`src/ghoshell_moss/ghosts/dolores/test_dolores.py::test_shell_clear_reaches_shell_directly`.

**Acceptance (thirdgerb's plan, for after a restart):** emit many short `<say>`s, ask out loud to
stop while the first is still playing, then use `say replace="true"` — the queue must cut and the
new line must be the one that plays.

Still open: the dispatch stall itself (the watchdog and the scope ceiling above), and the
`moss_wait_next` yield race.
