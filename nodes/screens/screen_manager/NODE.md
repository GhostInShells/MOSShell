---
name: 'screen_manager'
description: 'screen body — window semantics (item / group / arrange / fullscreen / veil / background) as an iframe compositor, zero-dependency webview backend'
category: screens
singleton: true
exec:
  command: python
  args: main.py
---

Screen manager is Ghost's visual body — a window-semantics compositor served as one
web page. The model drives it through a channel; the human watches the same page and
steers where to look.

Windows are **items** in a materialization pool, each in exactly one **group**; the
**active group** lays its items out by **grid** (equal split) or **stack** (one master +
a strip). The model owns layout and ordering; the human owns which group is shown and
what is fullscreen. **veil** is the top gesture layer (mark / arrow / text); the
**background** shows MOSS presence and audio visuals when the screen is empty.

```
<screen_manager:open id="term" url="http://127.0.0.1:8768" group="code" label="terminal"/>
<screen_manager:arrange ids="term,edit,chat" family="grid" dir="lr"/>
<screen_manager:mark item="term" region="right" duration="2"/>
<screen_manager:text content="HELLO MOSS" duration="2.5"/>
```

Two faces share one store: this channel (the ghost's side) and a web surface at
`http://127.0.0.1:8766` (the human's side).
