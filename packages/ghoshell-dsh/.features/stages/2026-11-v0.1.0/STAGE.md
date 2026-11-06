---
status: active
period: 2026-11 ~ 2026-12
delivery: 0.1.0.dev0 — built and verified locally, not published until MOSShell verifies the integration
---

# Stage: v0.1.0 — from a copy to a package

## Background

The package began on 2026-11-05 as a shell: `GROUND.md`, `AGENTS.md`, its own `features/`
tree, and a vendored research corpus with one version tree (`v0.2.2-alpha.2+ce7073d`).

The first code landed on 2026-11-06 by copying MOSShell's `deepseek_harness` into
`src/ghoshell_dsh/v0_2_0/` (17 modules, 7 test modules) and rewriting the intra-package
imports. It runs against dsh 0.2.0-rc.2, but it is a copy of another project's code, not a
package: it still imports `ghoshell_moss.contracts.*`, `core.blueprint.*`, `core.codex.*`
and `message`; it has no `pyproject.toml`; and nothing about it is installable.

Two capabilities were proven before this stage, in MOSShell's playground rather than here: a
plugin bundle that installs into a dsh profile without `dsh plugin add`, and an anchor call
that returns text and usage without creating a turn.

## Motivation

Upstream dsh iterates destructively — down to the patch digit — and MOSShell is about to
depend on it as a real driver. A copy that happens to import is not something a consumer can
install, version, or trust; and the next model arriving here has no way to tell what is
decided, what is proven, and what is still another project's code.

This stage turns the copy into a package, and makes the package legible to whoever arrives
next — human or model — without this conversation.

## Goals

- [ ] It stands on its own: an installable, versioned distribution that launches dsh and
      never imports `ghoshell-moss`.
- [ ] A running dsh is extensible by us: the plugin patch bundle ships inside the generation,
      and the surface names every patch we add to dsh.
- [ ] It is governable cold: a model with no context finds the field, the agent rules and the
      research corpus, and can continue without reading this conversation.
- [ ] Its code face belongs to users: English, organised around what it does for a caller,
      with development intent living in features rather than in the code.

## Plan

Not a schedule — an order that keeps each step checkable on its own.

1. **Agent-facing governance** — reshape `AGENTS.md` and `GROUND.md` so MOSShell discovers
   this package as its own field, and so a cold model knows where to start.
2. **A research system that settles** — make `.vendors/dsh-research/` a real architecture:
   one tree per `v<ver>+<sha>`, conclusions self-sufficient, a future model able to build on
   them without the source.
3. **v0_2_0 formalised** — clean up the copy into code we own, and cut the deep MOSShell
   dependencies (`contracts.*`, `core.blueprint.*`, `core.codex.*`, `message`).
4. **The plugin patch bundle** — deliver `plugin/` inside the generation, and locate on the
   surface every patch we add to dsh.
5. **Distributable version config** — `0.1.0.dev0`, buildable and verified locally; published
   only after MOSShell has verified the integration.
6. **Code governance** — English throughout, a usage-facing surface instead of a
   development-facing one, and development intent moved out of code into features.
7. **Abstraction split** — separate `launcher` from `connection`, define the blueprint
   (including the higher-level pieces), so later iterations inherit the thinking rather than
   rediscover it.

## Acceptance

- `import ghoshell_dsh` works with `ghoshell-moss` absent, and `grep -rn ghoshell_moss src/`
  returns nothing.
- The package builds (`wheel` + `sdist`) and reports `0.1.0.dev0`; it is **not** published.
- A dsh profile can be launched from this package; the plugin bundle installs through the
  documented path; anchor call and session-ref read return real values, with the evidence
  recorded under the generation.
- `launcher` and `connection` are separate modules, and a blueprint exists that explains the
  generation's pieces and the higher-level ones above them.
- No Chinese remains in code, docstrings or runtime messages; the surface reads as usage, not
  as intent.
- A model with zero context can read `GROUND.md` + `AGENTS.md`, find the research corpus, run
  the tests, and say what to do next.

## Associated Workstreams

None yet — the seven plan items are small enough, and ordered clearly enough, to execute
without workstream files. A workstream is created when an item grows past one sitting or
needs a decision record of its own.

## Retrospective

(not yet closed)
