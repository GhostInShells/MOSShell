---
description: >-
  MOSS's L3 deliverables checklist — the project's outward face. Pointers only; no duplicated content.
version: 1
updated: 2026-09-28
---

# SURFACE

> For the mechanism and template, see `README.md` (the features convention) and the SURFACE template.
> This document only points.

MOSS (Model-oriented Operating System Shell) is the **Shell layer** of the Ghost in Shells
architecture — it lets a persistent intelligence (a Ghost) descend into the real world: perceive,
think, and act, concurrently and in real time. The project has iterated from the kernel rewrite to
v0.1.0 (stage2), and is now closing the **ghost-iterates-on-itself** loop. For the full philosophy
and positioning, see the root `README.md` / `README.zh.md`.

## Checklist

- **Positioning / philosophy** → root `README.md` (architecture philosophy and positioning)
- **Entry points (model developers / explorers)** → root `CLAUDE.md` + the `moss` CLI system; `moss start` is responsible for self-explanation
- **Self-explanation tools for the architecture promises** → `moss codex` (concepts / blueprint / contracts / architecture / channeltypes)
- **Developer knowledge to maintain** → `moss skills`
- **Docs** → `moss docs` (under maintenance, possibly to be removed; regressed at each stage close)
- **Out-of-box ghosts** → `.moss/ghosts/` — must include a MOSS ghost that iterates on itself
- **Out-of-box capabilities** → `nodes/` (openbox capabilities) — carries the Ghost in Shells idea, the
  shared-context human–model collaboration idea, and the few shots of capability development
- **Human–model collaboration architecture** → `.ai_partners/` (the second of the three open-source
  pillars: the human–model collaboration system) — open-source sharing
- **Iteration-idea carrier** → `.ai_partners/features/` (the features system) — the open-source carrier
  of all iteration technical ideas
- **Engineering declaration** → `pyproject.toml` / `LICENSE` / `Makefile` / companion CLI

## Regression on the Surface

At each stage close, regress every entry: does each pointer still resolve, does each declared
capability (especially out-of-box ghosts / nodes / docs) still exist. This is one of the closing
deliverables of the L1→L2 iteration.
