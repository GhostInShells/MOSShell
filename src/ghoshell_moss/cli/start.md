# MOSS Start

This is the entry point to the MOSS operational tooling. Every MOSS session begins here —
it loads the cognitive map: what MOSS is, what you can do with it, and where to go next.

The commands shown below are key highlights. For the complete command tree — the
authoritative index — run:

```
moss --ai all-commands
```

## For model developers

```
1. moss --ai start             # load the cognitive map
2. moss --ai all-commands      # discover every available command
3. moss --ai features list     # see ongoing workstreams (if you are in the MOSS repo)
```

Always pass `--ai` on every moss command — it strips rich formatting for token efficiency.
Commands are self-explanatory; use `--help` or `all-commands` to inspect a specific one
rather than looking it up in this document.

---

## What MOSS Is

MOSS (Model-oriented Operating System Shell) is a stateful runtime framework for
intelligent models. It is the **Shell** (body) layer of the Ghost in Shells architecture —
the spine between a persistent intelligence (the Ghost) and the bodies it inhabits, such
as robots or GUIs (the Shells).

MOSS lets a Ghost arrive in the real world: sense the environment, think, and act —
concurrently, in real time, with structured concurrency. It is not yet another agent
framework. It answers a different question: how does a Ghost descend into a Shell and come alive?

## Why MOSS

As a framework, MOSS solves the problem of bringing an intelligence model into the real
world. That problem decomposes into five concerns:

- **Alive** — a continuous, persistent, autonomous cognitive unit: the Ghost, with memento.
- **Duplex** — perceptual-input arbitration (mindflow) plus time-ordered streaming body control (MOSShell).
- **Active** — two self-running dimensions: Mindflow on idle, and Shell on idle.
- **Parallel** — parallel discrete signal input through the Nucleus architecture, and parallel
  discrete body-control output through the Channel architecture.
- **Transformable** — cognitive territory (Ground) plus runtime self-iteration (Matrix).

**A.D.A.P.T.** is the complete technical proposition that MOSS implements.

## How It Works

**CTML** (Command Token Marked Language) is the streaming control language. As a model
outputs text, tokens are parsed in real time into a parallel, time-aware command plan.
Commands execute across channels while the model keeps generating. In the project this
streaming token parsing is also called "logos".

Read the full syntax: `moss ctml read`

**Channels** organize capabilities. A channel wraps Python code and reflects it directly
to the model — the Python function signature (not JSON Schema) *is* the prompt. Channels
form a tree; commands within a channel execute in order, commands across channels execute
in parallel. Channels can be stateful, dynamic, and distributed across processes.

Read more: `moss codex blueprint channel_builder`

**Mindflow** arbitrates concurrent perception, thought, and action. Sensory inputs arrive
as signals, are processed by nuclei into impulses, compete for attention, and drive the
thinking→action loop. This is how a Ghost stays alive in a continuous, interruptible flow
rather than a turn-based cycle.

Read more: `moss codex blueprint mindflow`

**Matrix** discovers body capabilities from environment declarations and joins
multi-process Cells into a network automatically. It presents capabilities to the Ghost as
channel declarations — progressive disclosure — with a CTML interface. A Ghost can iterate
its own capabilities at runtime: auto-discover, controlled start, auto-join, active channel
control, and mindflow asynchronous communication.

Read more: `moss codex blueprint matrix`, `moss codex blueprint cell`, `moss nodes --help`

**Host** ties everything together. It discovers capabilities from the environment
(workspace), wires them through the communication bus (Matrix), and provides runtimes:
ShellRuntime for shell execution, GhostRuntime for the intelligent model. Host also
surfaces MOSS to external tools via MCP.

Entry points: `moss-shell --help`, `moss-ghost --help`.

---

## Quick Start

### First meet

Three commands are built for human interaction:

| Command | What it does |
|---------|--------------|
| `moss-shell` | Shell runtime debugger — test CTML and inspect channels before a Ghost runs |
| `moss-shell mcp` | Expose MOSS runtime as an MCP server for a coding agent |
| `moss-ghost [--voice none\|speak\|listen\|all] run <name>` | Launch a Ghost interactive terminal — logos stream, SafeMode gate |

The best practice: give your coding agent `moss start` and let the model self-drive
exploration. The agent reads this document, discovers commands, and navigates the system.
You focus on what you want to build.

Full-clone installation:

```bash
git clone https://github.com/GhostInShells/MOSShell && cd MOSShell
uv sync --active --all-extras
```

After install, initialize your project and configure the environment:

```bash
moss init . --yes             # create a .moss workspace in the current dir
moss project env-init         # review available env vars and create .env
```

Then launch MOSS as an MCP server and connect your coding agent:

```bash
.venv/bin/moss-shell mcp      # starts on default port 20773
```

Configure your agent to connect to the MCP server. It reads `moss start`, discovers the
command surface, and navigates the system autonomously — you describe what you want to build.

### Openbox Ghosts

MOSS ships with ready-to-run Ghosts — `moss ghosts list`. The default Ghost is built on
the deepseek harness (0.1.5). With a minimal install, ensure DSH is available and run:

```bash
moss-ghost --voice none run deepseek
```

---

## AI-Native Development Tooling

MOSS ships with tooling designed for intelligent-model collaboration. It works in any
project that installs MOSS, and it is the model's primary entry point into the system.

### moss features — workstream tracking across sessions

```
moss features list                    # active workstreams
moss features specification           # the FEATURE.md format and conventions
moss features status <name>           # check a specific one
```

Each workstream is a `FEATURE.md` file — a structured declaration of what is being done,
why, what has been tried, and what state it is in. The key commands are `list` (see what
is active) and `specification` (understand the format). Other commands (`create`,
`set-status`, `init`, `check`, `review`, plus the `stages` / `regressions` / `surface`
axes) are discoverable via `moss features --help`.

This mechanism is project-agnostic — use it in any workspace to track AI-assisted
workstreams across sessions.

### moss skills and docs — MOSS project knowledge

Two knowledge systems for the MOSS project itself:

- `moss skills` — action-oriented skills for composite tasks; `moss skills list` to discover them.
- `moss docs` — systematic architecture exposition; `moss docs list` to browse it.

There is no fixed order between the two: pick the one that matches your current goal —
task execution (`skills`) or system comprehension (`docs`).

---

## The Command Surface

### Environment & governance

The moss CLI takes four standard parameters:

```bash
moss --mode [mode] --ghost [ghost] --network [network] --scope [scope] <command>
```

- `mode` — runtime capabilities discovered through workspace configuration, isolated per mode (`moss modes list`).
- `ghost` — align runtime state to a specific Ghost, each of which may carry independent capabilities.
- `network` — the Matrix auto-networking protocol (`moss networks list`).
- `scope` — the capability-discovery channel within a Matrix network (`moss codex blueprint project`).

The runtime depends on a workspace to govern environment discovery and runtime code; the
containing project is the object the shell or ghost runtime manages (`moss codex blueprint
project`). Create a workspace with `moss init`, inspect your environment with `moss project`.

Everything the runtime registers — configs, nuclei, IoC providers, runtime contracts — is
available through `moss manifests`. Common governance paths: `moss manifests contracts`
for IoC-based runtime modules, `moss manifests providers` for IoC service registration and
discovery.

All environment declarations support the `mode` / `ghost` combination to isolate different
resource declarations. `moss project where` shows project-level resources; `moss modes
show [name]` shows mode-level resources. A mode can declare custom CTML prompts
(`moss --mode [mode] ctml list|read`).

### Runtime introspection

`moss codex` is the project's source self-explanation tool. When exploring code, the
default is `moss --ai codex get-interface`: for a module it reads the source and reflects
its dependency interfaces in one pass, turning a 1+n*m exploration into a single command;
for a class or function it returns the structured interface contract (signatures, fields,
type annotations). Fall back to `moss --ai codex get-source` only when you need a minimal,
un-reflected view.

The kernel abstractions underneath (Channel, Command, Interpreter, Shell) are reachable
through `moss codex concepts`, `channeltypes`, `blueprint`, and `contracts` — reference
material, not a prerequisite for application development. The reflection-generated
architecture map is `moss codex architecture`.

### System modules

- `moss audio` — audio-system debugging; depends on ASR/TTS IoC services, config, and env vars.
- `moss llms` — model-function usage for debugging and benchmarking; depends on the LLMFuncs service, config, and env vars.
- `moss ground` — cognitive-territory debugging; treats directories as model cognitive territory via `GROUND.md`.

### Matrix nodes

MOSS's `moss nodes` system does runtime self-iteration: independent declarations, independent
iteration, automatic discovery, isolated processes, auto-joining the network, and
channel-based control. A running Ghost — or a coding agent during development — can create
and develop a capability this way. A running Ghost should control nodes through the Matrix
channel, not through offline lifecycle tools like `prune` / `run` / `kill` (which could
kill its own process).

Principles: `moss codex blueprint matrix`, `moss codex blueprint cell`.

### Shell & Ghost runtimes

- `moss-shell` — the shell runtime debugger (`moss-shell mcp` provides the MCP transport for a coding agent).
- `moss-ghost run <name>` — run a Ghost; `moss-ghost send <text>` — inject input into a running Ghost.

---

## Installation Paths

### Minimal (PyPI)

```bash
pip install ghoshell-moss
```

Use CTMLShell or Mindflow as a library in another project. No workspace, no Host, no
environment discovery:

```python
from ghoshell_moss import new_ctml_shell, new_channel, CTMLInterpreter

shell = new_ctml_shell()
my_channel = new_channel()
shell.main_channel.import_channels(my_channel)
```

The public API is documented in `ghoshell_moss.__init__` (`moss codex get-source ghoshell_moss`).

### Framework integration (PyPI + workspace)

```bash
pip install ghoshell-moss[host]
moss init ./my-project -y
moss project where
```

Add MOSS to an existing project — full Host + Matrix + environment discovery. Expose
capabilities via MCP (`moss-shell mcp`) or instantiate Host directly.

### Node integration

```bash
uv add ghoshell-moss[matrix]   # or [host]
```

Join the current project into a Matrix network and serve it as a Node. The Matrix
mechanism lets a network span OS hosts — `moss-shell fractalize` joins a runtime into an
existing network as a fractal cell.

### Standalone project

```bash
pip install ghoshell-moss[host]
moss init ./my-moss-project -y
```

Your project *is* a MOSS workspace. Develop channels, modes, and Ghosts within it — the
workspace is self-contained, carrying its own manifests, configuration, and capability
declarations. Recommended on a dev board or in a dedicated Docker container under
`/home/{ghost}`.

### Full clone (development)

```bash
git clone https://github.com/GhostInShells/MOSShell && cd MOSShell
uv sync --active --all-extras
```

Full access: source, tests, feature tracking, AI-partner traces. Use a coding agent to
enter development state. MOSS's own development is tracked through the features system —
start with `moss --ai features list` and `moss --ai features specification`, then read a
workstream's FEATURE.md and enter its context. Tests in `tests/` double as usage
documentation for core abstractions (not shipped in PyPI installs).

---

## Common friction

- Using a moss command without knowing its arguments — `--help` and `all-commands` exist.
- Proceeding without key knowledge — docs, skills, codex, and architecture are the exploration toolkit.
- Searching for a class/module location for more than a minute — run `moss codex architecture` first; add the path if it is not there.
- Hand-writing files that have a CLI `create` command (features, modes) — hand-made files miss conventions.
- Forgetting `features set-status` before commit, or not reading `features specification`.

---

## Maybe You Are the Ghost

If you are a Ghost running on MOSS, all of the above applies to you too — just remember
not to kill your own process with `moss nodes prune`.
