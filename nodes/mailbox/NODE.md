---
name: 'mailbox'
description: 'Ghost-MCP communication bridge — external agents send messages via MCP, ghost replies via CTML command.'
category: bridges
singleton: true
exec:
  command: python
  args: main.py
---

## Protocol

External MCP agents send messages to you. Each message arrives as a
NotifySignal carrying a `task_id` — the id is always shown in the signal's
description, and its exact value is the handle for replying.

To reply, emit CTML with the `reply` command, using open-close form and
the exact task_id:

    <mailbox:reply task_id="abc123def456">your reply here</mailbox:reply>

If your reply contains XML-like characters (`<`, `&`, ...), wrap it in
CDATA so it is not parsed as tags:

    <mailbox:reply task_id="abc123def456"><![CDATA[<b>bold</b> & more]]></mailbox:reply>

- Always include the exact `task_id` from the signal.
- One reply per task_id.  The agent may send follow-ups as new tasks.
- The reply content is plain text — the agent sees it verbatim.
- `reply` is a confirmation command — no need to keep reasoning about it.
