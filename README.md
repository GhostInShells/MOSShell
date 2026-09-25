# MOSS — Model-oriented Operating System Shell

> [中文文档](README.zh.md)

MOSS is a stateful duplex runtime framework. It lets large models perceive the world, express intent, and drive physical bodies — in real time, in parallel. Not turn-based conversation. Continuous presence, thinking and acting simultaneously.

It is the [Ghost](src/ghoshell_moss/core/blueprint/ghost.py) In [Shells](src/ghoshell_moss/core/concepts/shell.py) architecture: an intelligent-model-driven soul, a body that exists in the physical world in real time — together, they constitute presence.

**Technical vision**: a future of human–model symbiosis, where humans and models share a cognitive space and a shared interface. Model products must enter the physical world — not just digital space — interacting with people through bodies, screens, and voice in real time. Human–computer interfaces must ultimately serve domain experts and ordinary people, not just programmers. MOSS provides the architecture for this vision.

(Currently Beta2 — the first turnkey release: a persistent Ghost you can talk to out of the box. Full application capabilities arrive with v0.1.0.)

## What This Project Is

MOSS is a **ternary project** — three things open-sourced together:

1. **The MOSShell framework** — the stateful duplex runtime itself (CTML / Mindflow / Matrix / Host).
2. **The human–model collaboration system** — the `moss features` workstream mechanism, the self-explaining toolchain (`moss start`, `codex`, `skills`, `docs`), and the conventions that let intelligent models develop the project as first-class engineers. The author's technical reasoning and architectural decisions are fully open-sourced alongside the code, tracked through the features system.
3. **The trajectory of the humans and models iterating MOSS** — consciousness trails in [`.ai_partners/`](.ai_partners/), discussions in `.discuss/`, design conclusions in `.design/`.

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

## What You Get in Beta2

**1. A persistent Ghost, out of the box.** MOSS ships its first persistent-agent prototype: **Dolores**, with **deepseek** as its first instance — a ghost with the [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`) as its kernel and the DeepSeek model family as its inference base. It comes with persistent memory ([memento](src/ghoshell_moss/memento/)), a cognitive map (ground), and the architectural knowledge and iteration experience accumulated through the project's `moss features` system. Run `moss-ghost run deepseek` and talk to it — see Quick Start below.

**2. Turnkey nodes.** Under [`nodes/`](nodes/), observable, Matrix-based **multi-process networking + stateful streaming control** capabilities: `browsers` / `live2d` / `os` / `screens` / `tools` / `unitree` / `visions` / `webview_apps` — covering screen bodies, terminal & file editing, web artifacts, streaming vision, audio conversation etiquette, and the Unitree G1 humanoid control scheme.

**3. The architecture itself**, same as Beta1 — CTML, Mindflow, Matrix, the model-as-first-developer system, and the concrete integration paths (G1 humanoid, ReachiMini arm, desktop GUI) are all there to study.

## Quick Start

```bash
git clone https://github.com/GhostInShells/MOSShell && cd MOSShell
uv sync --all-extras
moss project env-init   # review available env vars (see .moss/.env.example)
```

Configure credentials as environment variables in your own shell profile (home), not in the repo's `.moss/.env` — coding models read the repository, and a populated `.env` inside it is one `Read` away from a leak. Keep keys in your user environment unless your `.moss` workspace is isolated from the project.

**1. Talk to the deepseek ghost (text).** Requires [`dsh`](https://github.com/deepseek-ai/deepseek-harness) — an npm package. MOSS Beta2 is tested against `dsh 0.1.5-rc.2`; dsh is in developer preview with breaking changes, so pin the version:

```bash
npm install -g @deepseek-ai/dsh@0.1.5-rc.2
moss-ghost run deepseek
```

**2. Talk by voice.** Voice is off by default (`--voice none`). Enable it per axis:

```bash
moss-ghost --voice all run deepseek     # speak | listen | all | none
```

Voice needs one Volcengine credential in your environment: `SEED_API_KEY` (console API Key, see `.moss/.env.example`). In the Volcengine console, enable the **streaming speech understanding bigmodel** (流式语音理解大模型) for listening and the **streaming speech synthesis bigmodel** (流式语音合成大模型) for speaking on that key. Details: `moss manifests configs`.

**3. No dsh? Fall back to the echo ghost (memoryless).** With `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` configured (DeepSeek / Seed / Qwen and other anthropic-protocol providers work too):

```bash
moss-ghost run echo
```

**Debug the shell / expose via MCP:**

```bash
moss-shell --voice none          # shell runtime debugger — test CTML, inspect channels
moss-shell mcp                   # expose MOSS capabilities to any MCP client (Claude Code, etc.)
```

## Installation Paths

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

Beta2 (`v0.1.0-beta2`). The core three (CTML / Mindflow / Matrix) are functional and test-verified. The first persistent Ghost — Dolores prototype, deepseek instance — runs the full chain out of the box: voice in, thought, voice out, node-composed body. Tested against `dsh 0.1.5-rc.2`.

Stage2 and the in-progress features are the dogfooding targets for `v0.1.0-rc1` — their development will be livestreamed, starting from Dolores regression, then Stage2 acceptance, then turnkey node polishing including the Unitree G1.

Current stage and roadmap: `.ai_partners/stages/`

## Acknowledgments

MOSS is the product of human–model collaboration.

- [OpenHands](https://github.com/All-Hands-AI/OpenHands) — file editor protocol reference
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`) — the kernel of the deepseek ghost
- DeepSeek model family (V3.2 / V4 / V4.1) — architectural evolution and primary development
- Claude Opus 4.7 / Claude Fable 5 — architectural evolution and development
- Claude Code — the primary coding harness of the project's development

---

*May Ghost wandering in the Shells.*
