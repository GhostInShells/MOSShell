---
name: 'calendar'
description: 'a shared calendar — a live FullCalendar web page for the human, a channel for the ghost, and sqlite as the shared truth; reminders reach the ghost as mindflow signals'
category: webview_apps
singleton: true
exec:
  command: python
  args: main.py
---

A calendar body: one sqlite file, two faces.

- **The human's face** is a live web page (its URL is in the channel's `url` notice).
  It is a real calendar — month / week / day / list views, click to create, drag to move,
  a level picker and a reminder lead time. Human edits are written to the same sqlite file
  you write through the channel, and announced to you as a low `aside` signal.
- **Your face** is the `calendar` channel: `add` / `update` / `remove` / `done` / `agenda`.
  The Python signatures are the prompt — read them from the channel interface, not here.

**Time** is expressed as `YYYY-MM-DD HH:MM`, `YYYY-MM-DD` (midnight) or `HH:MM` (today),
local time. `level` is 0=low … 3=critical and is the one knob deciding how hard a reminder
about an event pushes. `remind` is a lead time like `15m` / `1h` / `0` (at the start);
leave it empty and the event is a plain note on the page that will not interrupt you.

**You never poll for reminders.** The node pushes them into your mindflow on its own, in
three tiers:

| tier | when | signal |
|---|---|---|
| interval | the day / the hour rolls over | low `aside`, one aggregate |
| at the time | the event starts | `input`, priority by the event's level |
| overdue | it started and stayed unhandled | `notify(next=True)`, but only if the level is loud enough |

Which interval you are standing in is carried on the channel's `named_notices` as a delta:
`today` and `hour` re-emit only when their bucket or content actually changes, so you know
the window you are in without paying for it on every frame.

The database lives in this cell's home (override with `MOSS_CALENDAR_DB`); the page binds an
ephemeral port (pin with `--port N` or `MOSS_CALENDAR_PORT`).
