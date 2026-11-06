---
name: ghoshell-dsh
description: Multi-version dsh compatibility layer — launch and manage several dsh versions from Python
---

# ghoshell-dsh

A compatibility package for dsh. Upstream dsh iterates destructively — breaking down
to the patch digit — so this package keeps **one distribution with one directory per
generation** (`vX_Y_Z`): several dsh versions can be launched from Python, coexist,
and keep their historical data readable.

Dependency direction is one-way: this package does not depend on `ghoshell-moss`,
only on mechanism layers such as `ghoshell-processes`. Iteration conventions live in
[AGENTS.md](AGENTS.md); vendored sources in `.vendors/`.
