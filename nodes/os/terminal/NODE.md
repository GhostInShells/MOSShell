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

Commands run inside a **thread** — a named working context with a cwd. There is no
hidden current thread: every command names one.

```
<terminal:open thread="dev" cwd="." description="build and test"/>
<terminal:exec thread="dev" desc="list the repo"><![CDATA[ls -la]]></terminal:exec>
<terminal:read card_id="1"/>
```

Two faces share one store: this channel (the ghost's side) and a web surface at
`http://127.0.0.1:8768` (the human's side) — the card stream, the three actions
accept / deny / ask, stop and stop-all, and the mode switch.

**Cards.** A card is `id / type / title / description / content` plus whatever
happened to it. `accept` and `deny` are decisions; `ask` is talk — it leaves the
card pending and comes to the model as a question. Rules are cards too: `rule()`
proposes an auto-approval regex, and once a human accepts it, matching commands
run unattended while the terminal is in `auto` mode.

**Modes.** `approval` (every command waits for a human), `auto` (accepted rules
decide), `disabled` (this node's commands are not in the model's interface at
all). The mode and the pending/running counts live in this channel's notice.

**Reading results.** Output is pushed to the card as it is produced. Past a size
threshold the full text is written to a file under the node's `runtime/outputs/`,
and `read()` returns that path instead of the text.

Boundaries worth knowing: output arrives at line granularity — a partial line with
no newline yet is never visible — and this node orchestrates **processes**, not
persistent shell sessions (no `cd`, no environment carried between commands; each
command gets its explicit cwd).
