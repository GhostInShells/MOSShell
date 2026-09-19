---
name: 'deepseek-harness'
description: 'dsh connection control plane — code-driven workspace/session management + agent driving'
category: harness
singleton: false
exec:
  command: python
  args: main.py
---

dsh node: connects to a running dsh web and exposes a `dsh` channel the model
drives by writing Python against connection/session surfaces.

Start (token is the auth — the model can only start a node for an address whose
token it knows):

    moss nodes run nodes/deepseek-harness -- --host 127.0.0.1 --port 3080 --token <token>

Drive via CTML:

    <dsh:run><![CDATA[
    async def run(connection):
        return [s.sessionId for s in await connection.sessions()]
    ]]></dsh:run>

    <dsh:open session_id="s1"/>
    <dsh.s1:run><![CDATA[
    async def run(session):
        r = await session.run("reply ok")
        return r.final_response
    ]]></dsh.s1:run>

Read both surfaces: `moss codex get-interface ghoshell_moss.deepseek_harness.surfaces`.
