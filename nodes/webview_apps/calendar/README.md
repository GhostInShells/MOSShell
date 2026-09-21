# calendar

A shared calendar for a human and a ghost. The human sees and edits a live FullCalendar
page; the ghost reads and writes the same events through a channel; the sqlite file in
between is the shared truth. When something is due, the node pushes a reminder into the
ghost's mindflow on its own — no polling on either side.

## What it does

- A web calendar (FullCalendar 6, served from a jsdelivr CDN) with month / week / day /
  list views, click-to-create, drag/resize, a per-event `level` and a reminder lead time.
- A `calendar` channel the ghost drives: `add` / `update` / `remove` / `done` / `agenda`.
- Three-tier reminders delivered as mindflow signals:
  - the day / hour rolls over → one low `aside` aggregate,
  - the event starts → an `input` weighted by its level,
  - it goes overdue → a `notify(next=True)`, but only for events loud enough to deserve it.

## Setup

No install needed — the node uses only the stdlib (`sqlite3`, `asyncio`) and `websockets`
(already a `[host]` dependency). Delete `INSTALL.md` if one ever appears.

## Usage

```bash
moss nodes run nodes/webview_apps/calendar
```

Read the page URL from the channel's `url` notice and open it in a browser. The database
defaults to the cell's home directory as `calendar.db`; override with `MOSS_CALENDAR_DB`.

## Development

- `NODE.md` — the node manifest and the instruction body the ghost reads at runtime.
- `main.py` — thin entry point: builds the store, surface and channel, then provides them
  to the Matrix. `--port N` / `MOSS_CALENDAR_PORT` pin the page port.
- `src/ghoshell_calendar/`
  - `store.py` — the sqlite store (WAL + busy_timeout) and the due/overdue queries.
  - `reminders.py` — the three-tier reminder loop; `send_signal` and the clock are injected.
  - `surface.py` — one port serving `index.html` and `/ws`, plus the human-edit uplink.
  - `channel.py` — the ghost's command surface and the interval `named_notices`.
- `tests/` — store, reminder-loop and channel behaviour. Run `pytest nodes/webview_apps/calendar/tests`.

Run the node standalone for debugging: `python main.py` from this directory.
