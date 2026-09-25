---
name: 'co_browser'
description: 'observable playwright — every exec streams onto a shared web page so the human sees the source and result; one master switch turns the channel off'
category: browsers
singleton: true
exec:
  command: .venv/bin/python
  args: main.py
---

Co-browser is the "there is a human in the room" sibling of the `playwright`
node. The runtime is the same live browser (headed by default), but every
`exec`/`aexec` streams onto a small web surface — the human sees the code
that ran and the result. The surface page pops open automatically at
startup.

There is no per-command approval. Audit is the mechanism: the human trusts
the model to work because they can see, not because they nod at every call.
The only control is one **master switch** on the surface. When the human
flips it off, the channel's `available()` returns false and every command
drops out of the model's interface until it is switched back on.

```
<co_browser:exec><![CDATA[
page.goto("https://example.com")
print(page.title())
]]></co_browser:exec>
<co_browser:aexec><![CDATA[
page.wait_for_load_state("networkidle")
]]></co_browser:aexec>
<co_browser:history/>
```

The surface binds an **ephemeral port by default**; read its live URL from
the channel's `url` notice. To pin, start with `--port N` or set
`MOSS_CO_BROWSER_PORT`. Pass `--no-open` to skip the auto-launch.
