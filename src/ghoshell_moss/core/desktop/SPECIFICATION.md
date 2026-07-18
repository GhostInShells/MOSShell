# GROUND Format Specification

Status: draft 0.1.0 (2026-07-19)
Location: this file lives with the reference implementation. The workstream
plans to rename `desktop` → `ground` across the codebase; this file will
travel with the directory.

The contract layer (`contracts/`) holds the ABC bindings. The concrete layer
(`core/`) holds one reference implementation of this SPEC. Any other
runtime that reads and writes `GROUND.md` according to this SPEC can
participate.

## 1. Concept

A **ground** is a cognitive field bound to a directory. It carries three
things:

- a **loading convention** — the frontmatter block; MOSS's only legitimate
  invention domain
- a **body** of open-set markdown — 法 / 胶囊 / 场解说, model-editable
- a **pins section** — first-person declarations pointing at targets in
  the world

The runtime form of a ground is a **frame** — every refresh re-observes
each pin and re-renders the whole ground into `context_messages`.

## 2. File Structure

`GROUND.md` sits at the root of each ground directory:

```
---
$id: <optional URI-shaped identity>
label: <optional display label>
---

<body — free-form markdown, open set>

## ground:pins

<YAML list of pin declarations>
```

Three segments, boundary rules explicit:

1. **frontmatter** — YAML between the two `---` fences at the top of the
   file. MOSS reads only the keys listed in §3; other keys are model /
   upper-layer territory and are ignored by MOSS but preserved on write.
2. **body** — everything between the closing frontmatter fence and the
   `## ground:pins` heading. Open set. Carries the ground's exposition,
   inherited 法, promoted 胶囊, and 先例 trails.
3. **pins section** — a YAML list under the `## ground:pins` heading. If
   the heading is absent, the pins list is empty.

The `## ground:pins` heading is the boundary token. Any text between the
frontmatter and this heading is body; any text after this heading is
parsed as YAML.

Absence of `GROUND.md` in the directory means the ground has no
convention, empty body, and no pins — a bare-directory ground.

## 3. Frontmatter Reserved Keys

Only the keys in this table are consumed by MOSS. All other keys are
preserved on write but not interpreted.

| Key | Type | Semantics |
|-----|------|-----------|
| `$id` | string | Identity claim, URI-shaped. Follows JSON Schema `$id` prior — the ground *claims* an identity, it does not *point* somewhere. Any string accepted; MOSS does not validate scheme or shape (`moss:features`, `urn:moss:memory`, `https://…` all pass). Resolution — mapping `$id` to a concrete meaning — is upper-layer's job (typically the Ghost's `$id` registry). Optional. |
| `label` | string | Display label used in the enclosing Grounds collection. Optional; defaults to the directory basename. Label collisions inside one Grounds get `-2`, `-3` suffixes assigned at `open` time. |

Future extension keys (loading strategy hints, tree-ignore additions,
convention lineage, etc.) will extend this table. Implementations MUST
NOT reject unknown frontmatter keys.

## 4. Pins Section YAML Contract

The pins section is a YAML list. Each item is a map:

| Field | Required | Semantics |
|-------|----------|-----------|
| `label` | yes | Unique identifier within this ground. Charset: `[a-zA-Z_][a-zA-Z0-9_-]{0,63}`. Used by unpin for lookup and by the frame renderer as the fenced-block language tag. |
| `pin` | yes | Pin command argv, `list[str]`. Position 0 is the pin type verb (`file`, `glob`, `frontmatter`, `ls`, …). Positions 1+ are that verb's arguments. Every element is a string. |
| `description` | no | Short marginalia — a one-line "why this pin". Long exposition belongs in body. |

### 4.1 Example

```yaml
## ground:pins
- label: entry
  pin: [file, "src/entry.py"]
  description: "start reading here"
- label: hot
  pin: [file, "src/hot.py", "80-140"]
- label: py
  pin: [glob, "src/**/*.py"]
- label: status
  pin: [frontmatter, "FEATURE.md"]
  description: "check status field"
- label: layout
  pin: [ls, ".", "2"]
```

### 4.2 Unknown Pin Types

If `pin[0]` is not one of the known verbs in §5, the declaration is
preserved and its fenced result block in the frame renders:

```
error: unknown pin type: <verb>
```

Unknown types do not affect other pins, do not fail the frame, and are
preserved on rewrite. This is the forward-compatibility skeleton.

### 4.3 Label Conflicts On Pin

Adding a pin whose `label` already exists overwrites the existing entry.
This is the idempotent-overwrite ruling: `pin_file(l1, "a")` followed by
`pin_file(l1, "b")` leaves one pin with label `l1` pointing at `b`.

## 5. Known Pin Types

Every pin type documents: argv shape, expansion result, failure modes.
Failure modes always render `error: <reason>` inside the fenced result
block; the declaration in `## ground:pins` is preserved.

### 5.1 `file` — single file (with optional line range)

```
pin: [file, <path>]
pin: [file, <path>, <range>]
```

- `path` — ground-root-relative. Absolute paths and `..` traversal are
  rejected with `PathOutsideRootError` — the pin declaration is preserved
  but the fenced result renders the error.
- `range` — optional line range string, `N-M` (1-indexed, inclusive) or
  bare `N` (single line).

**Expansion**: file content, sliced to `range` if given.

**Failure modes**: file not found; range out of bounds; binary file.

### 5.2 `glob` — matching path view (paths + metadata, no content)

```
pin: [glob, <pattern>]
```

- `pattern` — ground-root-relative glob (`*`, `**`, `?` per standard glob
  semantics).
- **Filter**: matches are filtered through the project root's
  `.gitignore` via the `pathspec` dependency.

**Expansion**: matched paths with `mtime` and `size` per entry. **No file
content is expanded.** This is a strong SPEC guarantee — a `glob`
matching thousands of files must not blow up the context window. Callers
who want file content use `file`.

**Empty match** renders `no matches` (not an error).

**Failure modes**: pattern rejected as unsafe (absolute, `..` traversal).

### 5.3 `frontmatter` — YAML frontmatter of a single file

```
pin: [frontmatter, <path>]
```

- `path` — ground-root-relative, pointing at a markdown file with YAML
  frontmatter (or a bare YAML file).

**Expansion**: the frontmatter block verbatim (YAML). Body is not
included. The frontmatter of a single file is a bounded, small payload
by construction — no truncation needed.

**Failure modes**: file not found; file has no frontmatter block; YAML
syntax error inside the frontmatter; file cannot be parsed as markdown
or YAML.

### 5.4 `ls` — directory listing view (structure only, no content)

```
pin: [ls, <path>]
pin: [ls, <path>, <depth>]
```

- `path` — ground-root-relative, must be a directory.
- `depth` — optional traversal depth. `1` = current level only.
  Default: `2`.
- **Filter**: entries are filtered through the project root's
  `.gitignore` via `pathspec`.

**Expansion**: tree view, one line per entry. Files carry `mtime` and
`size`; directories are marked `<dir>`. **No file content.**

**Failure modes**: path not found; path is not a directory.

## 6. Frame Layout

Every frame renders the ground into `context_messages` with this
structure:

```
ground: <label> @ <absolute path>
$id: <id>                                     # only if declared

<GROUND.md body, verbatim>

<label>:<verb>(<args>) # <description>
<label>:<verb>(<args>) # <description>
...

```<label1>
<pin1 expansion>
```

```<label2>
<pin2 expansion>
```
...
```

Sections:

- **head** — ground label and absolute path, plus `$id` if the
  frontmatter declares one. One or two lines.
- **body** — the GROUND.md body verbatim. This is where 场解说, inherited
  法, promoted 胶囊, and 先例 live.
- **declaration block** — one line per pin, with the shape
  `label:verb(args) # description`. Free text is not permitted in this
  block; free exposition belongs to body. Empty description omits the
  ` # description` suffix. `args` are rendered comma-separated, quoted
  where they contain whitespace or shell-significant characters.
- **result block** — one fenced code block per pin. The pin's `label` is
  the fence language tag (e.g. ` ```entry `). Content is the pin's
  expansion per §5.

The pins YAML in `GROUND.md` and the declaration + result blocks in the
frame are two views of the same data. `GROUND.md` is authoritative;
frame rendering is derived.

## 7. Session Behavior

### 7.1 open / close

`Grounds.open(dir, label=None) -> Ground`

- Idempotent by `dir.resolve()`. Opening the same directory twice
  returns the same `Ground` instance; the second call's `label` (if
  provided) is ignored — the existing label wins.
- `label` defaults to `dir.name`. Collisions inside one `Grounds` are
  resolved with `-2`, `-3`, … suffixes at `open` time.

`Grounds.close(label)`

- Removes the `Ground` from the collection. Files on disk are not
  touched. A subsequent `open` on the same directory rebuilds session
  state from `GROUND.md`.

### 7.2 observe / stale

Before each frame, all pins observe their targets in parallel:

- read the target's `mtime` and content `hash`
- compare against the ground's **runtime shadow** (in-memory; never
  persisted to `GROUND.md`)
- mark `changed on disk` when the hash differs from the shadow

On first open, no prior frame exists, so nothing is marked `stale`. The
shadow is populated by the first observation and adopted as the initial
baseline.

The observation shadow (`seen_mtime`, `seen_hash`) MUST NOT be written
back to `GROUND.md`. The on-disk pins section only carries `label`,
`pin`, and optional `description`.

### 7.3 pin / unpin / update semantics

These are CTML-channel-layer verbs. The CLI does not implement them.
For a model or a human editing `GROUND.md` directly:

- **pin** — append a new item to the pins YAML list. If `label` conflicts,
  overwrite (see §4.3).
- **unpin** — remove the item whose `label` matches.
- **update** — force a fresh observation of the target, then adopt its
  current `mtime` and `hash` as the new shadow baseline, clearing the
  `changed on disk` mark.

## 8. CLI Surface

The `moss ground` command family is deliberately small — a diagnostic
and bootstrapping surface, not a replay of the CTML verb set. The design
rationale is that a well-specified file format plus a good editor is
faster than a wide CLI verb set for humans and models both:

| Command | Purpose |
|---------|---------|
| `moss ground spec` | Print this SPECIFICATION.md |
| `moss ground init [dir]` | Scaffold an empty `GROUND.md` in the directory (optional; hand-writing works too) |
| `moss ground frame [dir]` | Render the full frame for this ground (dogfood tool, matches §6) |
| `moss ground observe [dir]` | Run pin observations only; emit per-pin hit, mtime, and hash (diagnostics) |

No `pin` / `unpin` / `update` / `status` / `pins` / `instruction`
subcommands — `GROUND.md` is a plain markdown file, and editing it
directly is the fastest path.

Every CLI invocation is a fresh `open → render → close` cycle. There is
no cross-invocation `opened` state at the CLI layer — session state
belongs to the CTML channel layer.

## 9. Language Neutrality

This SPEC defines the on-disk format for `GROUND.md` and the wire
semantics of frame rendering. Any runtime (Python, Rust, Go, Java, …)
that reads and writes `GROUND.md` according to this SPEC can participate
in the ground protocol.

Requirements for a compliant implementation:

- Parse and emit the three-segment file structure per §2
- Consume the reserved frontmatter keys per §3; preserve unknown keys
- Read and write the pins YAML per §4; preserve unknown pin types
- Handle all four known pin types per §5, with the specified failure
  modes rendered into the fenced result blocks
- Emit frame layout per §6 (whitespace variations acceptable; block
  structure not)
- Never write observation shadow (`seen_mtime`, `seen_hash`) back to
  `GROUND.md`

## 10. Non-Goals

This SPEC does not cover:

- **Executable pin types** (`pin_bash` and family). Reserved for a
  future extension gated by T1 (cognitive interface carries no
  open-ended semantics). The argv contract in §4 accommodates them when
  the time comes without a format break.
- **L1 template instantiation** (`moss ground init` from a template
  library). Deferred; the current `init` command produces an empty
  scaffold only.
- **Cross-ground composition** (carried convention overlays, K35). Under
  discussion.
- **Ground discovery from a well-known sidecar directory** (K43,
  `.grands/`). Recorded as a fallback route; not adopted here.
