---
name: 'screen'
description: 'screen body — multi-slot QML compositor for Ghost visual presence, macOS PySide6'
category: skins
singleton: true
exec:
  command: .venv/bin/python
  args: main.py
---

Screen is Ghost's visual body — a compositor that manages windows (any URL) across
four slot layers: background, focus, front (expanded), float (meta-only). The Ghost
sees a channel tree (`screen` / `screen.layout`) and controls layout through CTML;
the human sees a unified QML scene with animated transitions.

Windows are URL-addressable resources. Any HTTP URL can become a window. Badging
uses the web-standard `navigator.setAppBadge()` API — pages need no knowledge of
screen internals.

Layout system: each layout is a QML component + sub-channel pair. Switching layouts
is a StatesChannel swap — the active command set changes, blocked during transition.
First layout: solo (1 focus + front strip + float shelf).

CTML invocation:
  <screen:open url="http://..." label="mail" />
  <screen:layout:focus id="#mail" />
  <screen:layout:float id="#mail" />
  <screen:switch_layout name="solo" />

The channel description is the discovery mechanism — no separate resource registry.
