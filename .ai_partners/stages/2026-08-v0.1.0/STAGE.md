---
status: planning
period: 2026-07 ~ 2026-08
delivery: v0.1.0
---

# Stage: v0.1.0 — Applications, Dogfooding, Dolores

## Background

Beta1 closes with every core concept in concrete form: CTML, channels,
matrix cell governance, mindflow, ghost runtime, desktop, ground, memento.
The infrastructure phase is over. What the system has not yet proven is
itself in sustained application use — the first dogfooding round (May–June)
tested concepts; this stage tests the whole.

## Motivation

A framework whose only user is its own development is not yet a project.
This stage turns MOSS from validated infrastructure into a working system
with real applications and a full-featured resident Ghost. It is
deliberately an application sprint, not a hardening pass: the goal is to
discover what the infrastructure is actually like to live in, and let
that discovery drive v0.1.0's final shape.

This intent is worth stating explicitly because it is not inferable from
the code: a model surveying the repo mid-stage should read this as a
sprint context, not as a codebase asking for defensive reinforcement.

## Goals

- [ ] Applications built on MOSS through sustained dogfooding
- [ ] Full-featured Dolores ghost — the first complete resident Ghost
- [ ] Existing features (plus a few new ones) closed out into a coherent
      deliverable
- [ ] Release v0.1.0 — the project formalizes, leaving beta

## Plan

1. Application development on top of the beta1 infrastructure, driving
   fixes and ergonomics from real use
2. Dolores ghost: identity, memory (memento), cognition (ground),
   perception (mindflow) assembled into one full-featured Ghost
3. Close in-flight workstreams (desktop-gui, memento-cli-and-agent,
   mindflow-channel, matrix-resources, interleaved-ctml-thinking, ...)
4. Version formalization: v0.1.0 tag, release notes

## Acceptance

- Dolores runs as a full-featured Ghost in daily use
- The features roster for this stage is completed or explicitly deferred
- v0.1.0 tagged; project status changes from beta to formal

## Associated Workstreams

- feature: ghost-prototype-dolores
- feature: desktop-gui
- feature: memento-cli-and-agent
- feature: mindflow-channel
- feature: matrix-resources
- feature: interleaved-ctml-thinking
- feature: moss-project-ground

## Retrospective

(not yet closed)
