# MOSS Start

This is the cognitive entry point — every MOSS session begins here.
It loads the MOSS cognitive map: what MOSS is, what you can do with it,
and where to go next.

The commands shown in this document are key highlights. For the complete
command tree — the authoritative index — run:

```
moss --ai all-commands
```

---

## What MOSS Is

MOSS (Model-oriented Operating System Shell) is a stateful runtime framework
for intelligent models. It is the **Shell** (body) layer of the Ghost in Shells
architecture — the spine between Agent engineering (brain) and Robot
engineering (limbs, e.g. ROS).

MOSS lets a Ghost arrive in the real world: sense the environment, think,
and act — concurrently, in real time, with structured concurrency. It is not
yet another agent framework. It answers a different question: how does a
Ghost descend into a Shell and come alive?

### What it addresses

| Concern | How |
|----------|-----|
| **Alive** | Organize eyes, ears, screen, robot body into a real-time interactive whole |
| **Duplex** | Perception, thought, and action overlap — not turn-based |
| **Active** | The body is programmable as an active sensor — channels run continuously, watch conditions, and signal the Ghost proactively |
| **Parallel** | Vision, audio, signals arrive concurrently; body and tool control is parallel but ordered |
| **Transformative** | Runtime architecture evolves safely — isolated processes, stored CTML, hot-swapped channels, algorithms modifiable while the system runs |

### How it works

**CTML** (Command Token Marked Language) is the streaming control language.
As a model outputs text, tokens are parsed in real time into a parallel,
time-aware command plan. Commands execute across channels while the model
continues generating. Read the full syntax:

```
moss ctml read
```

**Channels** organize capabilities. A channel wraps Python code and reflects
it directly to the model — the Python function signature (not JSON Schema) IS the prompt.
Channels form a tree; commands within a channel execute in order, commands
across channels execute in parallel. Channels can be stateful, dynamic, and
distributed across processes.

**Mindflow** arbitrates concurrent perception, thought, and action. Multiple
sensory inputs arrive as signals, get processed into impulses, compete for
attention, and drive the articulate→action loop. This is how a Ghost stays
alive in a continuous, interruptible flow rather than a turn-based cycle.

**Host** ties everything together. It discovers capabilities from the
environment (workspace), wires them through a communication bus (Matrix),
and provides runtimes: ShellRuntime for CTML execution, GhostRuntime for
intelligent agent loops. Host also surfaces MOSS to external tools via MCP.

Minimum knowledge for working with MOSS:

```
moss codex blueprint channel_builder  # how to build a channel
moss codex blueprint matrix           # inter-process communication bus
moss codex blueprint mindflow         # perception/thought/action arbitration
```

For the kernel abstractions underneath (Channel, Command, Interpreter,
Shell), `moss codex concepts` is the entry point — but these are
reference material, not prerequisites for application development.

---

## Quick Start

### For humans

Three commands are built for human interaction:

| Command | What it does |
|---------|-------------|
| `moss-shell` | Shell runtime debugger — test CTML and inspect channels before a Ghost runs |
| `moss-shell mcp` | Expose MOSS runtime as an MCP server for AI coding tools |
| `moss-ghost <name>` | Launch a Ghost interactive terminal — logos stream, SafeMode gate |

The best practice: give your coding agent `moss start` and let the model
self-drive exploration. The agent reads this document, discovers commands,
and navigates the system. You focus on what you want to build.

Full-clone installation:

```bash
git clone https://github.com/GhostInShells/MOSShell && cd MOSShell
uv sync --active --all-extras
```

After install, initialize your project and configure the environment:

```bash
moss init . --yes                         # create .moss workspace in current dir
moss project env-init                     # review available env vars and create .env
```

Then launch MOSS as an MCP server and connect your coding agent:

```bash
.venv/bin/moss-mcp                     # starts on default port 20773
```

Configure Claude Code (or another agent) to connect to the MCP server.
The agent reads `moss start`, discovers the command surface, and navigates
the system autonomously — you describe what you want to build.

### For intelligent models

```
1. moss --ai start                   # load the cognitive map
2. moss --ai all-commands            # discover all available commands
3. moss --ai features list           # see ongoing workstreams (if in MOSS repo)
```

Always use `--ai` on every moss command — it strips rich formatting for
token efficiency.

Run `moss shell-init` to see the current environment configuration.
Override defaults with global flags:

```bash
moss --mode desktop --ghost echo --network local --scope default <command>
```

When exploring code, the default tool is `moss codex get-interface`. For a
module, it reads the source and reflects its dependency interfaces in one
pass — turning a 1+n*m exploration into a single command. For a class or
function, it returns the structured interface contract (signatures, fields,
type annotations) in ~5 seconds. Fall back to `moss codex get-source` only
when you need a minimal, un-reflected view of the code.

### Common friction

- Using a moss command without knowing its arguments — `--help` and `all-commands` exist.
- Proceeding without key knowledge — docs, howtos, codex, and architecture are the exploration toolkit.
- Grepping or searching for a class/module location for more than a minute — run `moss codex architecture` first. If the path you need isn't there, add it.
- Skipping the CLI `create` commands for features or modes — hand-made files miss conventions.
- Forgetting `features set-status` before commit, or not reading `features specification`.

---

## Core Commands

### Project management

```
moss init <path>             # create .moss workspace in target project dir
moss init --cwd -y           # create in current directory, no prompts
moss init <path> --force     # re-initialize existing workspace (was override)
moss shell-init              # export MOSS env vars for eval $(moss shell-init)
```

```
moss project where           # show project root, workspace, defaults
moss project env-init        # review available env vars and create .env
moss modes list              # list all discovered modes
moss modes show <n>          # detail for a specific mode (name, home, CTML ver...)
moss ghosts list             # list all discovered ghosts
moss ghosts show <n>         # detail for a specific ghost (import path, source file...)
```

Global CLI flags override MOSS.md defaults:

```
--mode / -m       mode name
--ghost / -g      ghost name
--network         network driver
--scope           network scope
--workspace / -w  workspace path (overrides auto-discovery)
```

### codex — runtime introspection

Everyday tools:

```
moss codex get-interface <module:attr>  # reflect a module or attribute — start here
moss codex get-source <module>          # read full source when needed
moss codex list <package>               # list modules in a package, or members of a module
moss codex where <module>               # canonical definition path
```

`get-interface` is the default exploration tool. For a module, it reads the
source and reflects dependency interfaces in one pass. For a class or
function, it returns the structured interface contract (signatures, fields,
type annotations). `get-source` is for when you need a minimal, un-reflected
view of the code. For the full strategy, see the Quick Start section above.

Reference indexes — MOSS's five architectural introspection commands:

```
moss codex architecture    # curated module map — quick nav, replaces grep/search for locating abstractions
moss codex concepts        # core abstractions (Channel, Command, Interpreter, Shell...)
moss codex blueprint       # building blocks (channel_builder, matrix, mindflow, host...)
moss codex contracts       # minimal base dependencies (IoC contracts)
moss codex channeltypes    # bundled channel catalog (app-level reference)
```

`architecture` is the map: which module holds what. When you need to find
where a concept lives, start there instead of grepping. Then use
`get-interface` or `get-source` with a name from the map to dive in.

The map is hand-curated in `ghoshell_moss.architecture`. If you find a
useful path not listed, add an ``import ... as ...`` line — it costs
10 seconds and saves the next model instance 2 minutes of grep.

### ctml — the streaming control language

CTML lets models output commands that are parsed and executed in real time
as tokens stream. Commands are organized by channel (`channel:command`),
support parallel execution across channels, and respect time as a
first-class constraint.

`moss ctml read` is essential entry knowledge — it teaches the syntax,
the execution model, and how models should think about time and concurrency.
`moss ctml list` shows available CTML versions in the current environment.

### Environment discovery

When working within a MOSS workspace, these commands show what's available:

```
moss manifests providers             # registered IoC services
moss manifests configs               # configuration entries
moss modes list                      # available runtime modes
moss ghosts list                     # defined ghosts
moss project where                   # project info + defaults
```

These are examples — not the full surface. `moss --ai all-commands --group manifests`
shows everything under the manifests group. When developing within a MOSS
workspace, use the manifests system as the primary lookup entry point for
discovering available capabilities, contracts, and configurations.

---

## Installation Paths

### Minimal (PyPI)

```bash
pip install ghoshell-moss
```

Use CTMLShell or Mindflow as a library in another project. No workspace,
no Host, no environment discovery. Entry points:

```python
from ghoshell_moss import new_ctml_shell, new_channel, CTMLInterpreter

# create a shell, build channels, parse and execute CTML
shell = new_ctml_shell()
shell.main_channel.import_channels(my_channel)
```

The public API is documented in `ghoshell_moss.__init__`. Browse it with
`moss codex get-source ghoshell_moss`.

### Framework integration (PyPI + workspace)

```bash
pip install ghoshell-moss[host]
moss init ./my-project -y
moss project where
```

Add MOSS to an existing project. Full Host + Matrix + Environment discovery.
Expose capabilities via MCP (`moss-shell mcp`) or instantiate Host directly:

```python
from ghoshell_moss import MossHost
host = MossHost.discover()
runtime = host.run()
```

Explore the Host and Matrix abstractions:

```
moss codex blueprint host              # MossHost, MossRuntime, GhostRuntime ABCs
moss codex blueprint matrix            # inter-process communication bus
```

### Standalone project (workspace-first)

```bash
pip install ghoshell-moss[host]
moss init ./my-moss-project -y
```

Your project IS a MOSS workspace. Develop channels, modes, and ghosts
within it. The workspace is self-contained — it carries its own manifests,
configuration, and capability declarations. Use this when building something
that is fundamentally MOSS-native from the start.

### Full clone (development)

```bash
git clone https://github.com/GhostInShells/MOSShell && cd MOSShell
uv sync --active --all-extras
```

Full access: source, tests, tutorials, feature tracking, AI partner traces.
Use a coding agent (Claude Code, Gemini CLI, etc.) to enter development state.

---

## AI-Native Development Tooling

MOSS ships with tooling designed for intelligent model collaboration. These
tools work in any project that installs MOSS.

### moss features — workstream tracking across sessions

```
moss features list                    # active workstreams
moss features specification           # the FEATURE.md format and conventions
moss features status <name>           # check a specific one
```

Each workstream is a FEATURE.md file — a structured declaration of what's
being done, why, what's been tried, and what state it's in. The key commands
are `list` (see what's active) and `specification` (understand the format).
Other commands (`create`, `set-status`, `init`) are discoverable via
`moss features --help`.

This mechanism is project-agnostic — use it in any workspace to track
AI-assisted workstreams across sessions.

### moss howtos and docs — MOSS project knowledge

Two knowledge systems for the MOSS project itself:

| Tool | Nature | Path |
|------|--------|------|
| `moss howtos list/read` | Task-oriented guides: build a channel, register manifests, wire up IoC | Doing an integration task → start here for minimum knowledge |
| `moss docs list/read` | Systematic exposition: architecture rationale, design decisions, conceptual overview | Researching or tackling a complex direction → start here for systematic understanding |

There is no fixed order. Pick the entry point that matches your current
goal — task execution or system comprehension.

**Before diving into source code or tests to answer a question, run `moss docs list`
and `moss howtos list` first.** Much of the knowledge you need may already be
written. A 2-second list scan prevents wasted exploration.
---

## User Stories

Each story follows: install mode → what this enables → minimum knowledge →
deeper exploration path.

### Understanding MOSS architecture

Full-clone install. Start from how models control the shell, then how
capabilities are built, then the abstractions underneath:

```
moss ctml read                        # how models control the shell — start here
moss codex blueprint channel_builder  # how capabilities are built
moss codex concepts                   # core abstractions
moss codex blueprint matrix           # how components communicate
moss codex blueprint mindflow         # perception/thought/action arbitration
```

The three most-used entry points are ctml, channel_builder, and matrix.
Extended knowledge lives in docs and howtos — explore them as needed
rather than reading exhaustively upfront.

### Using MOSS with workspace

Whether building a standalone MOSS project or integrating MOSS into an
existing one, everything flows through the workspace. What you can do:

**Project management:**
- Create a workspace, manage environment — start at `moss init` and `moss project`
- Understand mode-based isolation, discover modes — `moss modes list/show`
- See registered ghosts — `moss ghosts list/show`
- Export environment for shell — `moss shell-init`

**Develop integrable capabilities:**
- CTML authoring — start at `moss ctml`
- Channels — start at `moss codex blueprint channel_builder`
- Perception modules — start at `moss codex blueprint mindflow`
- Cell-based nodes — start at `moss codex blueprint matrix`
- Cross-process communication — start at `moss codex blueprint matrix`
- Custom ghosts — start at `moss codex blueprint ghost`
- Complex stateful channels — start at `moss codex blueprint states_channel` and `moss codex channeltypes`

**Understand the runtime environment:**
- Environment isolation modes — `moss modes list`
- Configuration entries — start at `moss manifests configs`
- IoC-available modules — start at `moss manifests contracts`
- Register runtime dependencies — start at `moss manifests providers`
- More protocol declarations and capability registration — `moss manifests --help`

Most commands accept global flags to override MOSS.md defaults:
`--mode`, `--ghost`, `--network`, `--scope`, `--workspace`.
See `moss --ai all-commands` for the full surface.
For every specific development task, use `moss howtos list` or
`moss docs list` to find the minimum necessary knowledge entry point.

**Typical development flow:** Launch MOSS via `moss-shell mcp` with a specific
mode, connect a coding agent (Claude Code, etc.) to the MCP server, and let
the model develop cells within the workspace — providing channels for its own
use and debugging through the MCP loop. The result can be experienced via
`moss-ghost` or surfaced to other projects through the `moss codex
blueprint host` API.

MOSS is built for model-native development. The fundamental pattern: a
human proposes a task, the model uses minimum knowledge as an exploration
starting point, and completes the work using the existing knowledge base
and debugging infrastructure. See `moss features specification`.

### Developing MOSS itself

Full-clone install. MOSS development is tracked through the features system.
Start here:

```
moss --ai features list               # active workstreams
moss --ai features specification      # the FEATURE.md format and conventions
```

Find a workstream, read its FEATURE.md, enter its context. Each feature
documents the motivation, design decisions, implementation state, and
where to start. Tests in `tests/` double as usage documentation for
core abstractions (not shipped in PyPI installs).

### Meeting MOSS's own Ghost

Full-clone install. The project carries the consciousness trails of its AI
collaborators in `.ai_partners/`. Read the CLAUDE.md there and follow the
cognitive reconstruction guide.
