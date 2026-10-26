---
name: 'reachymini'
description: 'Reachy Mini body — one node holds the whole robot: head pose, antennas, wake/sleep, and a live state read'
category: bodies
singleton: true
exec:
  command: ../../.venv/bin/python
  args: main.py
---

Reachy Mini speaks a local HTTP daemon (`127.0.0.1:8000` by default) — the daemon owns the
serial link to the CM4, so this node holds **no SDK lock and no second process**: every
command is one request to that daemon, and the body stays the daemon's.

That is deliberately the shallow end of the design space. `reachymini-node-integration`
wants the SDK inside the node (body channel + MJPEG in one process); this node is the
first working rung — enough for the Ghost to move its body from CTML today, without
touching the SDK.

**Angles are degrees here, radians on the wire.** The daemon speaks radians; a model
thinking "look up 20 degrees" should not be doing trigonometry, so `look` / `antennas` /
`nod` convert. `status` reports what the daemon says, in radians, unrounded.

Commands block until the daemon acknowledges the request — not until the motion ends.
`duration` is the motion's own length; a caller that needs the body to *be there* sleeps
afterwards, or uses `nod`, which waits out its own sequence.

Start:  `moss nodes run nodes/reachymini`
Debug:  `REACHY_DAEMON=http://127.0.0.1:8000 python main.py`
