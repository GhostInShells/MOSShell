# MOSS — Model-oriented Operating System Shell

> [中文文档](README.zh.md)

MOSS is a stateful duplex runtime framework. It lets large models perceive the world, express intent, and drive physical bodies — in real time, in parallel. Not turn-based conversation. Continuous presence, thinking and acting simultaneously.

It is the [Ghost](src/ghoshell_moss/core/blueprint/ghost.py) In [Shells](src/ghoshell_moss/core/concepts/shell.py) architecture: an intelligent-model-driven soul, a body that exists in the physical world in real time — together, they constitute presence.

**Technical vision**: a future of human–model symbiosis, where humans and models share a cognitive space and a shared interface. Model products must enter the physical world — not just digital space — interacting with people through bodies, screens, and voice in real time. Human–computer interfaces must ultimately serve domain experts and ordinary people, not just programmers. MOSS provides the architecture for this vision.

(Currently Beta1. Turnkey application capabilities arrive with v0.1.0.)

## Model as First Developer

MOSS is a project where **intelligent models are the first developers**. Models are not only the Ghost (soul) within MOSS — they are its architect partners and its builders.

After May 7, 2026, the vast majority of features were designed through human–model architectural discussion, with models recording features and implementing them. All core design discussions, architectural decisions, and development context are fully open-sourced in the repository.

The project provides a complete self-explaining system for intelligent model developers. Models can independently explore the project and participate in development. The trajectory of human–model architectural collaboration is visible through `moss features list`.

The main body of human–model collaboration lives in [`.ai_partners/`](.ai_partners/), architectural discussion and evolution in [`.ai_partners/features/`](.ai_partners/features/), with further traces in [`.discuss/`](.discuss/) and [`.design/`](.design/).

## What Makes MOSS Different

**Concurrent multi-source perception.** Vision, audio, touch, system events — each arrives as an independent signal stream, simultaneously. No polling. No queuing. No serialization. [Mindflow](src/ghoshell_moss/core/blueprint/mindflow.py) arbitrates them in parallel — signals compete for attention, and Ghost sees keyframes fused from multi-source signals at every moment.

**Streaming interpretation and scheduling.** [CTML](src/ghoshell_moss/core/ctml/prompts/v1_0_0.en.md) is parsed and dispatched as tokens stream. Not "generate first, execute later" — generation IS execution. Time is a first-class citizen of the syntax. Multiple command tracks execute in parallel, including physical body control.

**Runtime self-iteration.** A stateful runtime: models create [Cells](src/ghoshell_moss/core/blueprint/cell.py), modify [Channels](src/ghoshell_moss/core/blueprint/channel_builder.py), and evolve their own capabilities — without stopping, without restarting. Cells are independent processes; a crash in one never takes down the host. Filesystem conventions replace configuration — put things in the right place, they are auto-discovered and auto-injected.

```
                              <- control               -> commands 
                            ╱            ╲           ╱            ╲
                           ╱              ╲         ╱              ╲
World -> signals ->  Mindflow                Ghost                Shell  -> actions -> World
                           ╲              ╱         ╲              ╱
                            ╲            ╱           ╲            ╱
                              impulses ->              <- results 
```

MOSS's architecture is a butterfly.
The left wing receives parallel signals from the external world; Mindflow schedules keyframes of thought.
The right wing sends commands to bodies, driving parallel, time-ordered actions that affect the world.
The Ghost — an intelligent model — controls the beating of both wings.

```
                    ┌───────┐
                    │ Ghost │
                    └───┬───┘
                        ▼ 
                    ┌────────┐
                    │ Matrix │
                    └───┬────┘
        ┌───────┬───────┼───────┬──────┐
        ▼       ▼       ▼       ▼      ▼
      robots sensors  screen  modules  OS
```

MOSS organizes network process units (Cells) through the [Matrix](src/ghoshell_moss/core/blueprint/matrix.py) communication bus. Ghost controls starting, stopping, and using them at runtime — and can iterate its own capabilities without restarting.

## Quick Example

MOSS builds the model's control surface through CTML. A person waves at the robot. The vision channel detects the motion and emits an impulse. Ghost receives the context and outputs CTML:

```
What the model sees:                  What the model outputs:

  <channel name="vision">             <_>
    async def look() -> str             Hello!
  </channel>                            <robot:wave duration="0.5"/>
  <channel name="robot">                I'm MOSS.
    async def wave(                   </_>
      d: float = 0.5
    ) -> None
  </channel>

  <perspective src="vision">
    person waving at you
  </perspective>
```

- **Code as Prompt**: the model sees Python function signatures, not JSON Schema
- **Time is a First-Class Citizen**: `<robot:wave/>` executes the moment the tag closes — wave 0.5s, speech continues, no waiting
- **Parallel tracks**: speech and robot are on different channels, executing in parallel. Same-channel commands run FIFO
- **Streaming dispatch**: the first token emitted is already being interpreted and executed

Minimum knowledge entry points: `moss ctml read` (CTML syntax), `moss codex blueprint channel_builder` (building capabilities), `moss codex blueprint mindflow` (perception arbitration), `moss codex blueprint matrix` (process networking).

## What You Can Do in Beta1

Beta1 delivers the architectural foundation. What you can explore today:

1. **Study MOSS's architectural approach** — is it yet another agent framework? What design decisions set it apart? Where can its ideas be borrowed?
2. **Study CTML, Mindflow, and Shell** — how MOSS approaches real-time duplex interaction: streaming interpretation, concurrent perception arbitration, parallel command scheduling.
3. **Understand the model-as-first-developer system** — how human–model collaboration is structured, how features track workstreams across sessions, and how the self-explaining system lets models independently explore and contribute.
4. **Study specific technical implementations** — G1 humanoid robot integration, ReachiMini robotic arm, desktop GUI, and other integration paths.

Turnkey application capabilities arrive with v0.1.0.

## Installation

```bash
git clone https://github.com/GhostInShells/MOSShell && cd MOSShell
uv sync --active --all-extras
cat .moss/.env.example # review default environment variables
claude code -p "Explore the MOSS project for me — what it is, what it can do, where to start"
```

| Install path | For |
|---|---|
| `pip install ghoshell-moss` | Embed Shell + Channel as a library in another project |
| `pip install ghoshell-moss[host]` + `moss init` | Prepare a standalone environment for a MOSS application |
| `git clone` + `uv sync --active --all-extras` | MOSS kernel developers, full toolchain |

All paths share one cognitive entry point: `moss start`.

## Demos

| Cross-app real-time communication | One Ghost, multiple bodies |
|---|---|
| ![apps_cross_talk](assets/apps_cross_talk.gif) | ![multiple_bodies](assets/multiple_bodies.gif) |
| Eyes, board, vision, voice — independent processes communicating in real time via streams | One Ghost simultaneously driving a desktop robot, a robotic arm, and a robot dog |

## Project Status

Beta1. The core three (CTML / Mindflow / Matrix) are functional and test-verified. The Matrix system is operational. Turnkey application capabilities are in development for v0.1.0. The v0.1.0 milestone targets Dolores Prototype — the first full-featured Ghost.

Current stage and roadmap: `.ai_partners/stages/`

## Acknowledgments

MOSS is the product of human–model collaboration.

- [OpenHands](https://github.com/All-Hands-AI/OpenHands) — file editor protocol reference
- DeepSeek model family (V3.1 / V3.2 / V4) — architectural evolution and primary development
- Gemini 3 — architectural design collaboration
- Claude Opus 4.7 / Fable 5 — architectural evolution and development

---

*May Ghost wandering in the Shells.*
