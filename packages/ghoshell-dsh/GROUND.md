---
name: ghoshell-dsh
description: Multi-version dsh compatibility layer — launch and manage several dsh versions from Python
pins:
- label: skills
  verb: frontmatter
  arguments:
    path: "src/ghoshell_dsh/skills/*/SKILL.md"
    keys: [name, description]
  description: distilled experience this package carries — usage samples and design method
- label: blueprint
  verb: glob
  arguments:
    path: "src/ghoshell_dsh/blueprint/*.py"
    limit: 20
  description: the contracts every generation answers to
---

# ghoshell-dsh

A compatibility package for dsh. Upstream dsh iterates destructively — breaking down
to the patch digit — so this package keeps **one distribution with one directory per
generation** (`vX_Y_Z`): several dsh versions can be launched from Python, coexist,
and keep their historical data readable.

Dependency direction is one-way: this package does not depend on `ghoshell-moss`,
only on mechanism layers such as `ghoshell-processes`. Iteration conventions live in
[AGENTS.md](AGENTS.md); vendored sources in `.vendors/`.

## Where to look

- **What this package promises** → `.features/SURFACE.md`
- **What period we are in, and what was decided** → `.features/` (`moss features` from here)
- **The contracts** → `src/ghoshell_dsh/blueprint/`; the layer map is its `__init__.py`
- **Experience, including how the contract layer was cut** → `src/ghoshell_dsh/skills/`, pinned
  above; read a skill before re-inventing what it already answers.
