# GROUND Format Specification

Any runtime that reads and writes `GROUND.md` according to this SPEC can
participate in the ground protocol.

## 1. Concept

A **ground** is a cognitive field bound to a directory. A directory is a
ground **if and only if** it contains a `GROUND.md` file. There is no
"bare-directory" ground — the marker is the boundary.

A ground carries:

- a **loading convention** — the frontmatter block
- a **body** of open-set markdown — law, field exposition,
  `@`-referenced documents, model-editable
- **pins** — first-person gaze declarations pointing at targets
  in the world

The runtime form of a ground is a **frame** — every refresh re-observes
each pin and renders the whole ground.

A ground inherits **law** from its ancestors: the `body` content of
`GROUND.md` files in parent directories, up to `$HOME`. This is the
**law chain** (§7). The chain carries body content only — frontmatter
and pins are never inherited.

### 1.1 Templates

A **template** is a pre-authored `GROUND.md` body + pins, stored under
`.grounds/` and discovered by the ground runtime. Templates and instances
share the same frontmatter + body + pins format.

Template discovery and usage rules are defined in §11.

## 2. File Structure

`GROUND.md` sits at the ground's root directory. Two segments:

```
---
$id: <optional URI-shaped identity>
label: <optional display label>
pins:
- label: <id>
  verb: <verb>
  arguments: {<key>: <value>, ...}
  description: <optional one-line why>
---

<body — free-form markdown>
```

1. **frontmatter** — YAML between the `---` fences. Reserved keys:
   `$id`, `label`, `pins`. Unknown keys are preserved on write but not
   interpreted by MOSS.
2. **body** — everything after the closing frontmatter fence. Open set.

The `pins` key holds the YAML list of pin declarations (§4). If absent,
the pins list is empty.

A directory without `GROUND.md` is not a ground. Opening such a directory
returns an empty ground; calling `sediment` on it creates `GROUND.md`.

## 3. Reserved Frontmatter Keys

| Key | Type | Semantics |
|-----|------|-----------|
| `$id` | string | Identity claim, URI-shaped. The ground *claims* an identity; resolution is upper-layer's job. Optional; any string accepted. |
| `label` | string | Display label. Defaults to the directory basename. Collisions get `-2`, `-3` suffixes at `open` time. |
| `pins` | list | Pin declarations per §4. Each entry has the fixed envelope: `label`, `verb`, `arguments`, `description`. |

Implementations MUST NOT reject unknown frontmatter keys.

## 4. Pin Envelope

Every pin uses a **fixed envelope** — the same fields regardless of verb:

| Field | Required | Semantics |
|-------|----------|-----------|
| `label` | yes | Unique identifier. Charset: `[a-zA-Z_][a-zA-Z0-9_-]{0,63}`. |
| `verb` | yes | Pin type: `file`, `glob`, `frontmatter`, `ls`, or future verbs. |
| `arguments` | no | Keyword arguments for the verb. Default `{}`. Schema depends on `verb` (§5). |
| `description` | no | One-line marginalia. Long exposition belongs in body. |

The envelope is **monomorphic** — polymorphism is quarantined inside
`arguments`. Tools that don't understand a verb can still parse, list,
and round-trip the pin.

**Unknown verbs**: preserved, not rejected. Their frame expansion
reports the verb as unknown. They do not affect other pins.

**Unknown arguments keys**: preserved on rewrite, not rejected.
Validation may warn; it must not fail.

**Label conflict**: adding a pin with an existing `label` overwrites
the old entry (idempotent overwrite).

### 4.1 Per-Pin Budget Parameters

Every pin MAY carry three optional budget fields in `arguments`. These
are declared once here and referenced by each verb's schema in §5:

| Field | Type | Semantics |
|-------|------|-----------|
| `budget` | int | Content character limit. When exceeded, output is truncated with a `[truncated at N chars]` marker. Applies to content-emitting verbs (`file`, `frontmatter`, `exec`). |
| `limit` | int | Entry count limit. When exceeded, output is truncated with a marker showing `N of M` entries. Applies to list-emitting verbs (`glob`, `ls`, `frontmatter` with pattern). |
| `max_depth` | int | Recursive discovery depth. Once a match is found at a given level, subdirectories of that match are not recursed into. Applies to pattern-emitting verbs (`frontmatter` with pattern, `ls`, `glob` with `**`). |

All three are optional. When absent, no limit is applied (implementation
defaults may apply). Each verb declares which of these it supports in §5.

## 5. Known Pin Types

All path-typed arguments use the anchor syntax in §8.

### 5.1 `file` — single file (with optional line range)

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | File path. Anchor syntax (§8) allowed. |
| `range` | string | no | Line range: `N-M` (inclusive) or bare `N`. |
| `budget` | int | no | Content char limit (§4.1). |

**Expansion**: file content, sliced to `range` if given.

**Failure modes**: file not found; range out of bounds; binary file.

### 5.2 `glob` — matching path view (paths + metadata, no content)

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | Glob path (`*`, `**`, `?`). Anchor syntax (§8) allowed as prefix. |
| `limit` | int | no | Entry count limit (§4.1). Default: implementation-defined. |
| `max_depth` | int | no | Recursion depth cap for `**` patterns (§4.1). |

Matches are filtered through `.gitignore`. **No file content is
expanded** — a `glob` matching thousands of files must not blow up
the context window. Use `file` for content.

**Expansion**: matched paths with size per entry (human-readable:
`12K`, `1.2M`). `mtime` is not rendered — use `exec` for timestamp queries.

**Empty match** renders an empty result (not an error).

### 5.3 `frontmatter` — YAML frontmatter of markdown file(s)

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | File path, or glob pattern matching multiple files. Anchor syntax (§8) allowed. |
| `keys` | list[string] | no | Frontmatter keys to extract. Absent = full frontmatter block. |
| `budget` | int | no | Content char limit (§4.1). |
| `limit` | int | no | Entry count limit when `path` is a pattern (§4.1). |
| `max_depth` | int | no | Recursion depth when `path` is a pattern (§4.1). |

**Single-file mode** (`path` is a concrete file path): extracts the
full frontmatter block verbatim. Body is not included.

**Pattern mode** (`path` contains glob characters `*`, `?`, `[`):
matches multiple files. Each matched file's frontmatter is rendered
as an independent result block, labeled by file path. This enables
**progressive disclosure** — a single `frontmatter` pin reveals the
identities and gaze declarations of all child grounds without
opening each one.

**keys** filtering: when specified, only the named frontmatter keys
are rendered. Unknown keys are preserved. This further reduces token
cost for identity-only queries (`keys: ["$id", "label"]`).

**Expansion**: frontmatter block(s) verbatim. Naturally bounded —
no truncation needed for single files. Pattern mode subject to
`limit` and `max_depth`.

**Failure modes**: file not found; no frontmatter block; YAML syntax
error; pattern matches zero files (not an error — renders empty).

### 5.4 `ls` — directory listing (structure only, no content)

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | Directory path. Anchor syntax (§8) allowed. |
| `depth` | int | no | Traversal depth. Default `2`. |
| `limit` | int | no | Entry count limit (§4.1). |
| `max_depth` | int | no | Recursion depth cap (§4.1). Alias for `depth` — whichever is smaller wins. |

Entries filtered through `.gitignore`. **No file content.**

**Expansion**: tree view with human-readable size per file
(e.g. `12K`, `1.2M`). `mtime` is not rendered.

**Failure modes**: path not found; path is not a directory.

### 5.5 `exec` — invoke a field-authored executable

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `ref` | string | yes | Relative path to an executable file **within the ground subtree**. |
| `timeout` | float | no | Seconds. Default `10`, max `60`. |
| `budget` | int | no | Content char limit (§4.1). Applied to captured stdout. |

**Authorization model** — this is the only pin verb that executes
code. It is deliberately narrow:

- `ref` MUST be a **relative path**. Absolute paths are rejected.
- `ref` MUST resolve inside the ground subtree. `..` traversal
  outside the ground is rejected.
- The target file MUST have the executable bit set (`+x`). Files
  without it are treated as missing (authorization denied).
- The interpreter is chosen by the target's shebang. The protocol
  does not distinguish `.sh` / `.py` / native binaries.
- **The protocol never accepts inline shell strings.** A pin cannot
  say "run this command"; it can only reference a script the field
  author has committed to the ground subtree.

The executable surface is a named artifact (a committed file),
never an inline string. The trust decision belongs to the loader,
not to the protocol.

**Execution environment**:

- `cwd` = `$GROUND` (the ground root). The executable knows its
  home. It reads `$CWD` from the environment if it cares about the
  caller's position.
- Environment variables `GROUND` and `CWD` are injected with
  absolute path values.
- `stdin` = `/dev/null`. `exec` pins are non-interactive.

**Expansion**: captured `stdout`, subject to `budget` truncation.

**Failure modes** — visible, not silent:

- Non-zero exit: output is followed by `[exit N]` and up to 5 lines
  of stderr tail.
- Timeout: partial stdout is followed by `[timeout after Ns]`.
- Missing / not executable / outside subtree: renders as `[missing]`.

**Observation**: `exec` is a compute-on-observe verb. The captured
payload is stored on the `Observation`; the frame renders the stored
payload rather than re-executing. One frame = at most one process
per pin.

### 5.6 `law` — convention-file law chain (compatibility)

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `filename` | string | yes | Convention file name (e.g. `CLAUDE.md`, `AGENT.md`). Collected from `$CWD` upward. |
| `budget` | int | no | Total char limit across collected law. Truncates with marker (§4.1). |
| `lines` | int | no | Total line limit across collected law. Truncates with marker. |

**Semantics**: this pin pulls *documents*, so the argument is a
**filename**, not a path. From the current `$CWD`, it walks upward
collecting every occurrence of `filename`, stopping at the ground
root (`$GROUND`) — the boundary is the ground, never `$HOME`. The
chain renders parent-first (root → cwd), each file as an independent
result block labeled with its path relative to the ground root.

`law` is the compatibility mechanism for foreign project conventions
(CLAUDE.md, AGENT.md): the ground declares a `law` pin pointing at
the foreign filename, and MOSS reads it — the foreign project's own
files are never modified or renamed.

**Positional view**: `law` depends on `$CWD`. Walking into a
subdirectory re-collects the chain from the new position. It does
not participate in stale marking — its content is a standing-position
view, not a disk-tracked target.

**`@`-reference expansion**: each collected file's body may contain
`@`-references (§6.2), resolved relative to that file's own directory.
Only one level is expanded — the resolved content is not re-scanned.
Fenced code blocks are skipped.

Note: the one-level cap is intentional — law files are external
convention documents (CLAUDE.md, AGENT.md), and their `@`-references
are resolved more conservatively than body `@`-references (§6.2, which
allow up to 3 levels). Body `@` is the field author's own narrative;
law `@` crosses a trust boundary.

**Expansion**: the collected bodies, root-first, labeled by relative
path, subject to `budget` / `lines` truncation.

**Failure modes**: no matching file in the range renders an empty
result (not an error).

## 6. Frame

A **frame** is the rendered form of a ground — body and pin results
assembled into a single output.

The frame covers:

- **body** — the GROUND.md body verbatim
- **pin results** — each pin's observation is expanded per its verb (§5)

The frame is a **derived view** — `GROUND.md` is authoritative.

**Meta information** (ground label, absolute path, `$id`, law chain)
is available through a separate `meta` command, keeping the frame
focused on content for consumers that don't need ground protocol.

### 6.1 Pin Result Blocks

Each pin result is delimited by HTML comment markers:

```
<!-- ground:pin:<label> -->
<pin observation content>
<!-- /ground:pin:<label> -->
```

The markers are machine-readable signals that do not collide with
user markdown.

Content rendered inside each block is verb-specific (§5). Content
output follows these rules:

- **No line numbers** — line numbers are for human debugging, not
  model consumption.
- **No raw mtime** — timestamps are shell-domain. Models that need
  them call `exec`.
- **Human-readable sizes** — file sizes render as `12K` / `1.2M` /
  `300B`, not raw byte counts.
- **Truncation markers** — when `budget` or `limit` is exceeded,
  a visible marker is appended: `[truncated at N chars]` or
  `[showing N of M entries]`.

### 6.2 `@`-reference Expansion

An `@`-reference in body loads another document as **static law**.
It is **not change-tracked** — the loaded content reflects the file's
current state at frame time. This contrasts with `pin`, which tracks
change across observations (§7.2).

**Recognition**: an `@` at line start or after whitespace, followed by
a path-start character `[a-zA-Z0-9_./$]`, and not inside a fenced code
block. The path runs as a maximal token of `[a-zA-Z0-9_./$-]`.
Quoted form for paths with special characters: `@"path with spaces.md"`.

**Expansion rules**:

- Resolves against `$GROUND` by default; explicit anchors (§8) allowed
- **Cycle detection**: each doc expanded at most once per chain
- **Depth cap**: max 3 levels of nested `@`-references
- **Budget cap**: 24000 chars total (implementation may override).
  When exceeded, remaining `@`-blocks are skipped with a warning.
  **Pin expansions are not subject to this budget** — pin content is
  governed by per-pin `budget` parameters (§4.1).

**Failure modes**: doc not found; path escapes anchor subtree.

## 7. Session Behavior

### 7.1 open / close

`Grounds.open(dir, *, label=None, doc=None, template=None) -> Ground`

- `dir` — ground root (pin anchor). Must be a directory.
- `label` — short identifier, derived from `dir` basename if omitted
- `doc` — explicit GROUND.md path (law anchor). Default `dir/GROUND.md`.
  When `doc` points elsewhere, the law anchor decouples from the pin
  anchor — `doc` is the portable law unit, `dir` is the local workplace.
- `template` — name of a template from `.grounds/` (§11). When specified,
  the ground is initialized with the template's body and pins. The
  template is **copied**, not referenced — subsequent pin/unpin/update
  operations do not affect the template file.

`open` is idempotent by resolved path. `close(label)` triggers
`sediment` (writes pins back to `GROUND.md`) and removes the
ground from the active collection.

### 7.2 observe / stale

Before each frame, all pins observe their targets: read `mtime` and
content `hash`, compare against the in-memory **shadow**. A hash
difference marks the pin `changed on disk`.

The shadow is never persisted to `GROUND.md`. On first observation,
the current state becomes the baseline (nothing stale).

CLI invocations are stateless `open → render → close` cycles — stale
marks never appear in CLI output. Stale marking is a session-level
capability for long-lived CTML channel sessions.

### 7.3 Pin Content Budget

Every pin that emits content or lists entries is subject to optional
per-pin budget parameters per §4.1. When a pin exceeds its budget or
limit, the result is truncated with a visible marker. These are safety
mechanisms, not correctness guarantees — models are expected to manage
pin granularity.

### 7.4 Nested Grounds

Subdirectories with their own `GROUND.md` are **peer grounds** —
independent instances with their own pins, body, and frame. They are
not auto-opened. Their law chain naturally reads ancestor `GROUND.md`
bodies.

Pins never inherit across grounds. Discovery of descendant grounds is
opt-in: use a `frontmatter` pin with a pattern (`$CWD/*/GROUND.md`)
for progressive disclosure of child ground identities, or a `glob`
pin for structural listing.

### 7.5 Law Chain

A ground inherits body content from `GROUND.md` files in ancestor
directories, root-first, up to `$HOME`. The chain carries body only —
no frontmatter, no pins. `@`-references in chain bodies are expanded
in-place (subject to §6.2 caps).

Chain content is destined for the channel's instruction slot — the
stable, cache-friendly context distinct from the volatile frame.
The frame renders only the ground's own body and pins.

The chain reads `GROUND.md` only. To reference foreign conventions,
use `@`-references with an explicit `$HOME` anchor.

## 8. Path Resolution

Three anchors:

| Anchor | Resolves to | Role |
|--------|-------------|------|
| `$GROUND` | Directory containing GROUND.md | Law anchor (default) |
| `$CWD` | Directory passed to `open(dir=...)` | Pin anchor / workplace |
| `$HOME` | User home directory | Machine-local escape hatch |

Bare relative paths default to `$GROUND`. Explicit anchors:
`$GROUND/path`, `$CWD/path`, `$HOME/path`.

**Subtree confinement**: every path, after anchor resolution and `..`
normalization, must resolve within its anchor's subtree. Symlinks are
resolved before the check. Bare absolute paths are rejected — use
`$HOME` for machine-local references.

`\$` escapes a literal `$` in filenames. Windows maps `$HOME` to
`%USERPROFILE%`; `$GROUND` and `$CWD` are platform-agnostic.

## 9. CLI Surface

The CLI is a diagnostic and bootstrapping surface, deliberately small:

| Command | Purpose |
|---------|---------|
| `moss ground spec` | Print this specification |
| `moss ground init [dir] [--template <name>]` | Scaffold GROUND.md, optionally from a template |
| `moss ground frame [dir]` | Render the ground's frame (body + pins) |
| `moss ground meta [dir]` | Show ground identity, law chain, and pin TOC |
| `moss ground observe [dir]` | Run pin observations; emit per-pin diagnostics |
| `moss ground validate [dir]` | Validate format and pin definitions |
| `moss ground templates` | List available templates from `.grounds/` |

No `pin` / `unpin` / `update` subcommands — `GROUND.md` is a plain
markdown file; direct editing is the fastest path.

Every CLI invocation is stateless `open → render → close`. Session
state belongs to the CTML channel layer.

## 10. Compliance

A compliant implementation must:

- Recognize a directory as a ground iff it contains `GROUND.md` (§1)
- Parse and emit the two-segment file structure per §2
- Consume reserved frontmatter keys per §3; preserve unknown keys
- Read and write pins per §4 with the fixed envelope; preserve unknown
  verbs and arguments keys
- Support per-pin `budget` / `limit` / `max_depth` parameters per §4.1
- Handle all six known pin types per §5, rendering failure modes
  into results
- Support `law` upward collection per §5.6 — bounded by the ground
  root, root-first display, one-level `@`-expansion, truncation
- Support `frontmatter` in both single-file and pattern modes (§5.3)
- Render human-readable file sizes, omit raw `mtime` (§6.1)
- Expand `@`-references per §6.2 with cycle detection, depth cap, and
  budget cap
- Resolve paths per §8 with per-anchor subtree confinement
- Implement the law chain per §7.5
- Never persist observation shadow to `GROUND.md`

## 11. Template Discovery (`.grounds/`)

Templates are pre-authored GROUND.md equivalents stored under
`.grounds/` directories. They share the frontmatter + body + pins
format with instances — a `.md` file under `.grounds/` is structurally
identical to a `GROUND.md`.

### 11.1 Discovery

On initialization, the ground runtime scans these paths for templates:

1. `$CWD/.grounds/**/*.md` — project-local templates
2. `$HOME/.grounds/**/*.md` — machine-global templates
3. Ghost-carried templates (implementation-defined path)

The three sources are merged into a single template catalog.
**Project-local templates take priority** over machine-global ones
with the same name.

Template name = path relative to `.grounds/`, minus `.md` extension:
`.grounds/python-project.md` → name `python-project`
`.grounds/ghost/memory.md` → name `ghost/memory`

### 11.2 File Format

A template file uses the same two-segment structure as `GROUND.md`:

```
---
$id: <optional>
label: <optional>
pins:
- label: <id>
  verb: <verb>
  arguments: {<key>: <value>, ...}
  description: <optional>
---

<body — free-form markdown>
```

The `$id` and `label` in the template are **template metadata**.
When a ground is created from a template, these are carried over
as initial values; the instance's own `GROUND.md` can override them.

### 11.3 Usage

`Grounds.open(dir, template="python-project")`:

1. Resolve the template name against the merged catalog
2. Create a new Ground at `dir`
3. Initialize the ground with the template's convention, body, and pins
4. The ground is now live — pin/unpin/update operate in memory
5. `sediment` writes the current state to `dir/GROUND.md`

The template is **copied**, not referenced. After `open`, the ground
has an independent lifecycle.

Creating a ground from scratch (without `template=`) initializes an
empty ground — no convention, empty body, empty pins. The ground
becomes real when the first `sediment` writes `GROUND.md`.

### 11.4 Fractal Closure

A `.grounds/` directory may itself contain a `GROUND.md` — making it
a ground instance discoverable by its parent ground's `frontmatter` pin.
File system traversal is the discovery protocol; no separate registry
is required.

### 11.5 Separation from Instance Discovery

- `**/GROUND.md` — discovers ground **instances** (active cognitive fields)
- `.grounds/**/*.md` — discovers **templates** (pre-authored field blueprints)

The filename `GROUND.md` is reserved for instances. Templates use
different filenames under `.grounds/`, guaranteeing that instance
discovery never accidentally picks up templates.
