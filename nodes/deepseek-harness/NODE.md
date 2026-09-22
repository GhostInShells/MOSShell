---
name: 'deepseek-harness'
description: 'dsh agent control plane — drive dsh sessions as a code-first surface: send/wait/interrupt/status/read, watch for notifications, run(code) escape'
category: harness
singleton: false
exec:
  command: python
  args: main.py
---

dsh node: connects to a running dsh web and exposes one flat `dsh` channel —
an agent control plane borrowing the perception shape of an IM session list.

Start (token is the auth — the model can only start a node for an address whose
token it knows; `--token` is optional if `DSH_WEB_TOKEN` is set):

    moss nodes run nodes/deepseek-harness -- --host 127.0.0.1 --port 3080 --token <token>

Drive via CTML — a few lifted verbs, everything else is code. The tags below
name the channel as it names itself; from the ghost's **host cell** the same
channel mounts as a child of the mesh projection, so the tag carries the mount
path — `<matrix.mesh.<alias>:...>`. `<alias>` is reserved at spawn: pass
`name="dsh"` to `nodes:run(target, name)`, or declare `{target: ..., alias: dsh}`
in the mode's `bringup_nodes`, and the path is exactly `matrix.mesh.dsh`. A node
started without one — CLI `moss nodes run`, a `bringup_nodes` entry with no
alias, or a re-mount after the host restarted — falls back to the cell's
uid-suffixed short address (`matrix.mesh.dsh_<uid6>`), whose suffix changes on
every spawn.

    <dsh:new name="p1"/>                       # create + attach a session
    <dsh:send name="p1" text="看下 runtime.py"/>
    <dsh:wait name="p1"/>                      # block in the room until the turn ends

    <dsh:watch name="p1" level="notify"/>      # push results as signals (row/notify/next)
    <dsh:send name="p1" text="继续"/>

    <dsh:read name="p1" n="10"/>               # pull context (clears unread)
    <dsh:status name="p1"/>                    # running / tokens / last line
    <dsh:interrupt name="p1"/>                 # cancel the in-flight turn
    <dsh:runs/>                                # background-run receipt ledger

Code escape (injects the connection surface without `name`, the session surface with it):

    <dsh:run name="p1"><![CDATA[
    async def run(session):
        return await session.history(max_messages=20)
    ]]></dsh:run>

Attention discipline (100 parallel agents still hold): results are silent by
default — they land in `runs()` and the notice counts. Only `watch` pushes, and
`next` is an explicit escalation, never the default. Topological dependencies
(fan-in/fan-out/ordering) are written as code, not channel verbs.

Read the code surface: `moss codex get-interface ghoshell_moss.deepseek_harness.surfaces`.
