# GROUND Format Specification

Any runtime that reads and writes `GROUND.md` according to this SPEC can
participate in the ground protocol.

## 1. Concept

A **ground** is a cognitive field bound to a directory. A directory is a
ground **if and only if** it contains a `GROUND.md` file. There is no
"bare-directory" ground — the marker is the boundary.

A ground carries:

- a **loading convention** — the frontmatter block
- a **body** of open-set markdown — law, ground exposition,
  `@`-referenced documents, model-editable
- **pins** — first-person gaze declarations pointing at targets
  in the world

A ground inherits **law** from its ancestors: the `body` content of
`GROUND.md` files in parent directories, up to `$HOME`. This is the
**law chain** (§7.3). The chain carries body content only — frontmatter
and pins are never inherited.

## 2. File Structure

`GROUND.md` sits at the ground's root directory. Two segments:

```
---
$id: <optional URI-shaped identity>
name: <optional display name>
description: <optional one-line about this ground>
pins:
- label: <id>
  verb: <verb>
  arguments: {<key>: <value>, ...}
  description: <optional one-line why>
---

<body — free-form markdown>
```

1. **frontmatter** — YAML between the `---` fences. Reserved keys:
   `$id`, `name`, `description`, `pins`. Unknown keys are preserved on write.
2. **body** — everything after the closing frontmatter fence. Open set.

The `pins` key holds the YAML list of pin declarations (§4). If absent,
the pins list is empty.

A directory without `GROUND.md` is not a ground.

## 3. Reserved Frontmatter Keys

| Key | Type | Semantics |
|-----|------|-----------|
| `$id` | string | Identity claim, URI-shaped. The ground *claims* an identity; resolution is upper-layer's job. Optional; any string accepted. |
| `name` | string | Display name. Defaults to the directory basename. |
| `description` | string | One-line description of this ground. Optional. |
| `pins` | list | Pin declarations per §4. Each entry has the fixed envelope: `label`, `verb`, `arguments`, `description`. |
| `ignore` | list of strings | Ground-level ignore patterns — `.gitignore` semantics, relative to ground root. All discovery pins (glob, frontmatter pattern, ls) automatically respect these rules. Optional. |
| `ignore_file` | string | Path to a file (relative to ground root) containing additional ignore patterns, one per line. Merged with `ignore` inline list. `.gitignore` or `.groundignore` are expected names. Optional. |

Implementations MUST NOT reject unknown frontmatter keys.

### 3.1 Ground-Level Ignore

`ignore` and `ignore_file` declare exclusion rules at the **ground level** —
one set of rules for the entire ground. All discovery pins (`glob`, `frontmatter`
pattern, `ls`) automatically respect them; `file`, `exec`, and `law` pins are
unaffected. No per-pin opt-in or opt-out is needed.

Rules use `.gitignore` syntax (pathspec `gitignore`):
- `dir/` — exclude directory `dir` and everything under it
- `*.log` — exclude files matching the glob
- `/root-only` — anchored to ground root
- `!important.log` — negate a previous rule

How rules are assembled:
1. `ignore` list items become individual lines
2. `ignore_file` is read (if it exists) and non-comment, non-blank lines are appended
3. Lines are fed to a single `PathSpec` instance
4. A path matches if `PathSpec.match_file(relpath)` returns true, where
   `relpath` is the path relative to the pattern base (for glob) or ground
   root (for ls walk)

The hardcoded `GLOB_IGNORE` set (`.git`, `.venv`, `__pycache__`, etc.) is
always active as a basename filter. Ground-level ignore is applied as an
additional layer on top — both must pass for a path to be visible.

## 4. Pin Envelope

Every pin uses a **fixed envelope** — the same fields regardless of verb:

| Field | Required | Semantics |
|-------|----------|-----------|
| `label` | yes | Unique identifier. Charset: `[a-zA-Z_][a-zA-Z0-9_-]{0,63}`. |
| `verb` | yes | Pin type: `file`, `glob`, `frontmatter`, `ls`, or future verbs. |
| `arguments` | no | Keyword arguments for the verb. Default `{}`. Schema depends on `verb` (§5). |
| `description` | no | One-line marginalia. Long exposition belongs in body. |
| `always_show` | no | Boolean, default `false`. In walk mode, pins whose paths are `$CWD`-anchored expand; others fold to a compact view. A pin with `always_show: true` always expands full content — relevant for `law` pins where compact mode shows path list only.

The envelope is **monomorphic** — polymorphism is quarantined inside
`arguments`. Tools that don't understand a verb can still parse, list,
and round-trip the pin.

**Unknown verbs**: preserved, not rejected. Their rendered expansion
reports the verb as unknown. They do not affect other pins.

**Unknown arguments keys**: preserved on rewrite, not rejected.
Validation may warn; it must not fail.

**Unknown envelope keys**: dropped on rewrite — the envelope is fixed.
Only `label`, `verb`, `arguments`, `description`, and `always_show`
survive a round-trip through the runtime.

**Label conflict**: adding a pin with an existing `label` overwrites
the old entry (idempotent overwrite).

### 4.1 Per-Pin Budget Parameters

Every pin MAY carry three optional budget fields in `arguments`. These
are declared once here and referenced by each verb's schema in §5:

| Field | Type | Semantics |
|-------|------|-----------|
| `budget` | int | Content character limit. When exceeded, output is truncated with a `[truncated at N chars]` marker. Applies to content-emitting verbs (`file`, `frontmatter`, `exec`). |
| `limit` | int | Entry count limit. When exceeded, output is truncated with a marker showing `N of M` entries. Applies to list-emitting verbs (`glob`, `ls`, `frontmatter` with pattern). |
| `max_depth` | int | Recursion depth in directory levels below the pattern's static base. `1` = one layer of sub-fields (the filename is not counted). When absent, recursion is unbounded. |

All three are optional. When absent, no limit is applied. Each verb
declares which of these it supports in §5.

`max_depth` is a pure depth cap. Field-boundary stop (防穿透 — do not
recurse into a directory that directly contains a match) is a separate,
per-verb semantic: `frontmatter` pattern mode applies it (ground
discovery does not penetrate child grounds), `glob` does not.

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

Matches are filtered against a built-in set of known noise
directory basenames (`.git`, `.venv`, `__pycache__`, `node_modules`,
etc.). Ground-level `ignore` rules (§3.1) apply as an additional
filter with `.gitignore` semantics. **No file content is expanded** —
a `glob` matching thousands of files must not blow up the context
window. Use `file` for content.

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
| `max_depth` | int | no | Recursion depth in directory levels when `path` is a pattern (§4.1). |

**Single-file mode** (`path` is a concrete file path): extracts the
full frontmatter block verbatim. Body is not included.

**Pattern mode** (`path` contains glob characters `*`, `?`, `[`):
matches multiple files. Each matched file's frontmatter is rendered
as an independent result block, labeled by file path. Subject to
`max_depth` depth cap, ground-boundary stop (防穿透 — a directory that
directly contains a match is not recursed into), and ground-level
`ignore` rules (§3.1). This enables **progressive disclosure** — a
single `frontmatter` pin reveals the identities and gaze declarations
of all child grounds without opening each one.

**keys** filtering: when specified, only the named frontmatter keys
are rendered. Unknown keys are preserved. This further reduces token
cost for identity-only queries (`keys: ["$id", "name"]`).

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

Entries filtered against a built-in set of known noise basenames.
Ground-level `ignore` rules (§3.1) apply as an additional filter:
ignored directories are not recursed into. **No file content.**

**Expansion**: tree view with human-readable size per file
(e.g. `12K`, `1.2M`). `mtime` is not rendered.

**Failure modes**: path not found; path is not a directory.

### 5.5 `exec` — invoke a ground-authored executable

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
  say "run this command"; it can only reference a script the ground
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

- File does not exist: renders `[missing]`.
- File exists but lacks executable bit (+x): renders `[not executable]`.
- Absolute path or `..` escaping the ground subtree: renders `[outside ground]`.
- Non-zero exit: output is followed by `[exit N]` and up to 5 lines
  of stderr tail.
- Timeout: partial stdout is followed by `[timeout after Ns]`.

**Observation**: `exec` is a compute-on-observe verb. The captured
payload is stored on the `Observation`; the view renders the stored
payload rather than re-executing. One render = at most one process
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
the foreign filename, and the implementation reads it — the foreign
project's own files are never modified or renamed.

**Positional view**: `law` depends on `$CWD`. Walking into a
subdirectory re-collects the chain from the new position.

**`@`-reference expansion**: each collected file's body may contain
`@`-references (§6.1), resolved relative to that file's own directory.
Only one level is expanded — the resolved content is not re-scanned.
Fenced code blocks are skipped. Both body and law `@`-references share
this one-level semantics (§6.1).

**Expansion**: the collected bodies, root-first, labeled by relative
path, subject to `budget` / `lines` truncation.

**Failure modes**: no matching file in the range renders an empty
result (not an error).

## 6. Rendering Strategy

The protocol does not mandate a specific rendering format.
A renderer produces an ordered view of a ground:

1. **header** — ground identity: name, root path (`$GROUND`), and
   optionally `$CWD` when rendering from a subdirectory (walk mode)
2. **body** — the GROUND.md body verbatim
3. **pin results** — each pin's observation expanded per its verb (§5),
   in declaration order. Walk mode folds `$GROUND`-anchored pins
   into a compact index.

The pin `description` field (§4) is the **progressive disclosure
carrier** — renderers should surface it alongside pin results so a
reader understands *why* a pin exists without opening GROUND.md.

Derived views (header, meta information) are separated from content
renderings — consumers that don't need ground protocol can receive
body + pin results only.

Content output rules:
- No line numbers, no raw `mtime`
- Human-readable file sizes (`12K` / `1.2M`)
- Truncation markers when `budget` / `limit` are exceeded

### 6.1 `@`-reference Expansion

An `@`-reference in body loads another document as **static law**.
It is **not change-tracked** — the loaded content reflects the file's
current state at render time. This contrasts with `pin`, which tracks
change across observations (§7.1).

**Recognition**: an `@` at line start or after whitespace, followed by
a path-start character `[a-zA-Z0-9_./-]`, and not inside a fenced code
block. The path runs as a maximal token of `[a-zA-Z0-9_./-]`.
Quoted form for paths with special characters: `@"path with spaces.md"`.

**Expansion rules**:

- Resolves against `$GROUND` by default; no anchor syntax — the
  reference is a plain relative path (`$GROUND`/`$CWD`/`$HOME` are not
  recognized)
- **Single-level**: one level of `@`-expansion is applied; resolved
  content is not re-scanned. This applies to both body and `law`-pin
  `@`-references — there is no recursive chain.

**Failure modes**: doc not found; path escapes anchor subtree.

## 7. Session Behavior

### 7.1 Pin Content Budget

Every pin that emits content or lists entries is subject to optional
per-pin budget parameters per §4.1. When a pin exceeds its budget or
limit, the result is truncated with a visible marker. These are safety
mechanisms, not correctness guarantees — models are expected to manage
pin granularity.

### 7.2 Nested Grounds

Subdirectories with their own `GROUND.md` are **peer grounds** —
independent instances with their own pins, body, and view. They are
not auto-opened. Their law chain naturally reads ancestor `GROUND.md`
bodies.

Pins never inherit across grounds. Discovery of descendant grounds is
opt-in: use a `frontmatter` pin with a pattern (`$CWD/*/GROUND.md`)
for progressive disclosure of child ground identities, or a `glob`
pin for structural listing.

### 7.3 Law Chain

A ground inherits body content from `GROUND.md` files in ancestor
directories, root-first, up to `$HOME`. The chain carries body only —
no frontmatter, no pins. `@`-references in chain bodies are expanded
in-place, one level deep (§6.1).

The chain is a stable, cache-friendly context distinct from the
volatile rendered view. The view renders only the ground's own body
and pins.

The chain reads `GROUND.md` only. To reference foreign conventions,
use a `law` pin (§5.6); `@`-references are plain relative paths.

### 7.4 Walk Mode — Ground Interior Navigation

When the view is rendered with `$CWD` set to a directory inside the
ground but not at the ground root, the view enters **walk mode**.

In walk mode, the nearest ancestor `GROUND.md` provides the ground root
and pins. The `$CWD`-anchored pins expand full content in the current
context, while `$GROUND`-anchored pins fold to a table of contents.
This gives a per-directory view — pins declared at the ground root
with `$CWD` anchor paths follow the viewer through the ground.

Walk mode is initiated when rendering from a subdirectory that has no
`GROUND.md` but an ancestor does. No new ground is opened — the law
anchor remains at the discovered `GROUND.md`.

## 8. Path Resolution

Three anchors:

| Anchor | Resolves to | Role |
|--------|-------------|------|
| `$GROUND` | Directory containing GROUND.md | Law anchor (default) |
| `$CWD` | Current viewing position (directory) | Pin anchor / workplace |
| `$HOME` | User home directory | Machine-local escape hatch |

Bare relative paths default to `$GROUND`. Explicit anchors:
`$GROUND/path`, `$CWD/path`, `$HOME/path`.

**Subtree confinement**: every path, after anchor resolution and `..`
normalization, must resolve within its anchor's subtree. Symlinks are
resolved before the check. Bare absolute paths are rejected — use
`$HOME` for machine-local references.

`\$` escapes a literal `$` in filenames. Windows maps `$HOME` to
`%USERPROFILE%`; `$GROUND` and `$CWD` are platform-agnostic.

## 9. Compliance

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
- Render human-readable file sizes, omit raw `mtime` (§6)
- Expand `@`-references per §6.1 (single-level, no recursion)
- Resolve paths per §8 with per-anchor subtree confinement
- Implement the law chain per §7.3

