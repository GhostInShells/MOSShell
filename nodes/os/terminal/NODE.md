---
name: 'terminal'
description: 'shell commands as cards — the human watches them stream, questions them, and accepts or denies; the model is never blocked on the answer'
category: os
singleton: true
exec:
  command: python
  args: main.py
---

Terminal runs shell commands for the ghost, and shows each one to a human as a
**card**. The pose is that the human is a busy collaborator, not an approval gate:
`exec()` returns a receipt immediately — the model never waits on a person's
response time. What happens to the card next (accepted, denied, talked about,
withdrawn) and what the process did (exited, failed, was stopped) both arrive as
signals, so `read(id)` is how the model learns its fate.

Commands run inside a **thread** — a named working context with a cwd. The
default thread `root` is always open at the node's own root, so `exec()` with no
thread runs there. `open()` names another location; every cwd must live inside
the root.

```
<terminal:open thread="dev" cwd="." description="build and test"/>
<terminal:exec thread="dev" desc="list the repo"><![CDATA[ls -la]]></terminal:exec>
<terminal:read card_id="1"/>
<terminal:ground thread="dev"/>
```

Two faces share one store: this channel (the ghost's side) and a web surface (the
human's side). The surface binds an **ephemeral port by default**; read its live
URL from this channel's `url` notice — never assume a fixed port. To pin one,
start with `--port N` (or set `MOSS_TERMINAL_PORT`).

**Trust is per-thread and per-pattern, not a global switch.** A command runs
without asking when its thread is `auto` (the human flips that thread's "auto"
button) or it matches an accepted rule — `rule()` proposes the regex, the human
accepts it. The two global postures are `approval` (the default) and `disabled`
(commands drop out of the interface).

**Cards and actions.** `accept` and `deny` are decisions; `ask` is talk — it
leaves the card pending and comes to the model as a question. `accept`/`deny`
arrive as **aside** signals (buffered, no interrupt); when the last pending card
settles, one **notify** tells the model the review round is over. `ask` and
process completion are notifies.

**Ground.** Every thread has a cognitive field (the nearest `GROUND.md`). The
model reads it with `ground(thread)`; the human views the same field on demand
from the surface's thread panel (a "ground" button that renders it live). The
field is the model's concern — the human only shares the view.

**Analyze.** Every card's detail panel has a zero-context "analyze" box: a
side-channel model reads the command text (not the issuing model's intent) and
answers the human's question. This is a second opinion for trust, kept entirely
out of the ghost's message stream. It goes through the moss message protocol
(`call_messages`): the prompt is one `Message` carrying an XML document that
grows with each turn (`<cwd>` + `<command>` + one `<turn>` per question/answer —
no assistant messages). If no LLM func engine is configured, it answers
"unavailable" and the box hides.

**Reading results.** Output is pushed to the card as it is produced. Past a size
threshold the full text is written to a file under the node's `runtime/outputs/`,
and `read()` returns that path instead of the text.

Boundaries worth knowing: output arrives at line granularity — a partial line with
no newline yet is never visible — and this node orchestrates **processes**, not
persistent shell sessions (no `cd`, no environment carried between commands; each
command gets its explicit cwd).
