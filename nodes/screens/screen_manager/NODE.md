---
name: 'webview_screen'  # 档位必须进名字: `screen` 是三档躯体(webview/qt/os window)共用的抽象, 裸 screen 会互相占位
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
<screen_manager:open id="term" url="http://127.0.0.1:PORT" group="code" label="terminal"/>
<screen_manager:arrange ids="term,edit,chat" family="grid" dir="lr"/>
<screen_manager:mark item="term" region="right" duration="2"/>
<screen_manager:text content="HELLO MOSS" duration="2.5"/>
```

Two faces share one store: this channel (the ghost's side) and a web surface (the
human's side). The surface opens in the system browser on start (`--no-open` for a
headless run) and binds a **fixed port: 20381** — a screen is a *place*, so the human's
window has to be findable at a known address. 20381 sits in the 2038 series, next to the
zenoh mesh endpoint (20380). The live URL is still on this channel's `url` notice; pass
`--port N` (or set `MOSS_SCREEN_MANAGER_PORT`) to run a second screen beside the first.
Ports are documented per-node, the way `ghost-in-bilibili` pins 23880.

The **background** is two layers over one audio bus: the ghost's voice falls as the
character rain, the human's voice rises as a symmetric glyph wave from the floor.
Both read `AudioSampleTopic` (`role=ghost` / `role=user`) and share one glyph set —
who is speaking is told by direction, not by colour. See
`.ai_partners/features/workstreams/2026/07/screen-manager/screen-todo.md` (W7).
