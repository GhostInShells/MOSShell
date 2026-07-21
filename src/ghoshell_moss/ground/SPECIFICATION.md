# GROUND Format Specification

Status: v1.0.0 (2026-07-21)
Location: `src/ghoshell_moss/ground/SPECIFICATION.md`

Any runtime that reads and writes `GROUND.md` according to this SPEC can
participate in the ground protocol.

## 1. Concept

A **ground** is a cognitive field bound to a directory. It carries:

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

Absence of `GROUND.md` in the directory means the ground has no
convention, empty body, and no pins.

## 3. Reserved Frontmatter Keys

| Key | Type | Semantics |
|-----|------|-----------|
| `$id` | string | Identity claim, URI-shaped. The ground *claims* an identity; resolution is upper-layer's job. Optional; any string accepted. |
| `label` | string | Display label. Defaults to the directory basename. Collisions get `-2`, `-3` suffixes at `open` time. |
| `pins` | list | Pin declarations per §4. Each entry has the fixed envelope: `label`, `verb`, `arguments`, `description`. |

Implementations MUST NOT reject unknown frontmatter keys.

## 4. Pin Envelope

Every pin uses a **fixed envelope** — the same four fields regardless
of verb:

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

## 5. Known Pin Types

All path-typed arguments use the anchor syntax in §8.

### 5.1 `file` — single file (with optional line range)

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | File path. Anchor syntax (§8) allowed. |
| `range` | string | no | Line range: `N-M` (inclusive) or bare `N`. |

**Expansion**: file content, sliced to `range` if given.

**Failure modes**: file not found; range out of bounds; binary file.

### 5.2 `glob` — matching path view (paths + metadata, no content)

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `pattern` | string | yes | Glob pattern (`*`, `**`, `?`). Anchor syntax (§8) allowed as prefix. |

Matches are filtered through `.gitignore`. **No file content is
expanded** — a `glob` matching thousands of files must not blow up
the context window. Use `file` for content.

**Expansion**: matched paths with `mtime` and `size` per entry.

**Empty match** renders an empty result (not an error).

### 5.3 `frontmatter` — YAML frontmatter of a single file

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | Path to a markdown file with YAML frontmatter, or a bare YAML file. |

**Expansion**: the frontmatter block verbatim. Body is not included.
Naturally bounded — no truncation needed.

**Failure modes**: file not found; no frontmatter block; YAML syntax error.

### 5.4 `ls` — directory listing (structure only, no content)

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | Directory path. Anchor syntax (§8) allowed. |
| `depth` | int | no | Traversal depth. Default `2`. |

Entries filtered through `.gitignore`. **No file content.**

**Expansion**: tree view with `mtime` and `size` per file.

**Failure modes**: path not found; path is not a directory.

## 6. Frame

A **frame** is the rendered form of a ground — body, expanded
`@`-references, and pin results assembled into a single output.

The frame covers:

- **body** — the GROUND.md body verbatim
- **@-expansions** — each `@path` reference in body is expanded to the
  referenced document's content (§6.1)
- **pin results** — each pin's observation is expanded per its verb (§5)

The frame is a **derived view** — `GROUND.md` is authoritative.

**Meta information** (ground label, absolute path, `$id`, law chain)
is available through a separate `meta` command, keeping the frame
focused on content for consumers that don't need ground protocol.

### 6.1 `@`-reference Expansion

An `@`-reference in body loads another document as **static law**.
It is **not change-tracked** — law follows the doc's current state
silently. This is the dividing line between `@` and `pin`:
**对账 (accounting)**. `@` = load as law, no accounting.
`pin` = watch as gaze, with change accounting (§7.2).

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
  **Pin expansions are not subject to this budget** — each pin is a
  declarative gaze commitment; the model is expected to manage pin
  count and verb choice.

**Failure modes**: doc not found; path escapes anchor subtree.

## 7. Session Behavior

### 7.1 open / close

`Grounds.open(dir, *, label=None, doc=None) -> Ground`

- `dir` — ground root (pin anchor)
- `label` — short identifier, derived from `dir` basename if omitted
- `doc` — explicit GROUND.md path (law anchor). Default `dir/GROUND.md`.
  When `doc` points elsewhere, the law anchor decouples from the pin
  anchor — `doc` is the portable law unit, `dir` is the local workplace.

`open` is idempotent by resolved path. `close(label)` removes from the
collection without touching disk.

### 7.2 observe / stale

Before each frame, all pins observe their targets: read `mtime` and
content `hash`, compare against the in-memory **shadow**. A hash
difference marks the pin `changed on disk`.

The shadow is never persisted to `GROUND.md`. On first observation,
the current state becomes the baseline (nothing stale).

CLI invocations are stateless `open → render → close` cycles — stale
marks never appear in CLI output. Stale marking is a session-level
capability for long-lived CTML channel sessions.

### 7.3 Pin budget

`file` pins expand full content. `glob` and `ls` pins can match
arbitrarily many entries. Implementations SHOULD apply a per-pin
expansion budget. When a pin exceeds its budget, the result is
truncated with a visible marker. The budget is a safety mechanism,
not a correctness guarantee — models are expected to manage pin
granularity.

### 7.4 Nested Grounds

Subdirectories with their own `GROUND.md` are **peer grounds** —
independent instances with their own pins, body, and frame. They are
not auto-opened. Their law chain naturally reads ancestor `GROUND.md`
bodies.

Pins never inherit across grounds. Discovery of descendant grounds is
opt-in (e.g., via a `glob` pin on `*/GROUND.md`).

### 7.5 Law Chain

A ground inherits body content from `GROUND.md` files in ancestor
directories, root-first, up to `$HOME`. The chain carries body only —
no frontmatter, no pins. `@`-references in chain bodies are expanded
in-place (subject to §6.1 caps).

Chain content is destined for the channel's instruction slot — the
stable, cache-friendly context distinct from the volatile frame.
The frame renders only the ground's own body, `@`-expansions, and pins.

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
| `moss ground init [dir]` | Scaffold an empty `GROUND.md` |
| `moss ground frame [dir]` | Render the ground's frame (body + pins) |
| `moss ground meta [dir]` | Show ground identity, law chain, and pin TOC |
| `moss ground observe [dir]` | Run pin observations; emit per-pin diagnostics |
| `moss ground validate [dir]` | Validate format and pin definitions |

No `pin` / `unpin` / `update` subcommands — `GROUND.md` is a plain
markdown file; direct editing is the fastest path.

Every CLI invocation is stateless `open → render → close`. Session
state belongs to the CTML channel layer.

## 10. Compliance

A compliant implementation must:

- Parse and emit the two-segment file structure per §2
- Consume reserved frontmatter keys per §3; preserve unknown keys
- Read and write pins per §4 with the fixed envelope; preserve unknown
  verbs and arguments keys
- Handle all four known pin types per §5, rendering failure modes
  into results
- Expand `@`-references per §6.1 with cycle detection, depth cap, and
  budget cap
- Render frames with body, `@`-expansions, and pin results per §6
- Resolve paths per §8 with per-anchor subtree confinement
- Implement the law chain per §7.5
- Never persist observation shadow to `GROUND.md`
