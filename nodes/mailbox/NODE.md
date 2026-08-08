---
name: 'mailbox'
description: 'Ghost-MCP communication bridge — external agents send messages via MCP, ghost replies via CTML command.'
category: bridges
singleton: true
exec:
  command: .venv/bin/python
  args: main.py
---

## Protocol

External MCP agent sends a message → you receive a NotifySignal with a
`task_id`.  The signal body looks like:

    [mailbox:abc123def456] user message here

To reply, emit CTML:

    mailbox:reply(id=abc123def456, content=your reply here)

The agent will receive your reply on its next poll.

- Always include the exact `task_id` from the signal.
- One reply per task_id.  The agent may send follow-ups as new tasks.
- The reply content is plain text — the agent sees it verbatim.
