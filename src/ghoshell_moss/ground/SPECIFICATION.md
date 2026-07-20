# GROUND Format Specification

Status: draft 0.2.0 (2026-07-20)
Location: `src/ghoshell_moss/ground/SPECIFICATION.md`. K49/K50 rename
complete — file is `GROUND.md`, package is `ghoshell_moss.ground`.

The contract layer (`contract.py`) holds the ABC bindings. The concrete
layer (`_*.py`) holds one reference implementation of this SPEC. Any
other runtime that reads and writes `GROUND.md` according to this SPEC
can participate.

## 1. Concept

A **ground** is a cognitive field bound to a directory. It carries three
things:

- a **loading convention** — the frontmatter block; MOSS's only legitimate
  invention domain
- a **body** of open-set markdown — 法 / 场解说 / `@`-referenced law,
  model-editable
- a **pins section** — first-person gaze declarations pointing at targets
  in the world

The runtime form of a ground is a **frame** — every refresh re-observes
each pin and re-renders the whole ground into `context_messages`.

A ground also inherits **law** from its ancestors: the `body` content of
`GROUND.md` files in parent directories of the ground's law anchor, up
to and including `$HOME`. This is the **law chain** (§7.5). The chain
carries body content only — pins and frontmatter are never inherited.

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
   `@`-referenced law docs (§6.1), promoted 胶囊, and 先例 trails.
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

Future extension keys will extend this table. Implementations MUST
NOT reject unknown frontmatter keys. (K42: `$id` replaces the older
`$template` notion; blood-lineage is not tracked in frontmatter.)

## 4. Pins Section YAML Contract

The pins section is a YAML list. Each item is a map with a **fixed
envelope** — the same four keys for every pin, regardless of verb:

| Field | Required | Semantics |
|-------|----------|-----------|
| `label` | yes | Unique identifier within this ground. Charset: `[a-zA-Z_][a-zA-Z0-9_-]{0,63}` (`PIN_LABEL_MAX_LEN = 63`). Used by unpin for lookup and by the frame renderer as the fenced-block language tag. |
| `verb` | yes | Pin type verb (`file`, `glob`, `frontmatter`, `ls`, …). A plain string. |
| `arguments` | no | A map of keyword arguments for the verb. Default `{}`. Unknown keys preserved on rewrite, not rejected (§4.2). |
| `description` | no | Short marginalia — a one-line "why this pin". Long exposition belongs in body. |

The envelope is **monomorphic**. The polymorphism across verbs is
quarantined inside `arguments` — a single field whose schema depends on
`verb`. This separation lets tools that don't understand any verb still
parse the envelope, list pins, edit descriptions, and round-trip
unknown verbs without touching `arguments`. (K55: supersedes K44's argv
contract. Rationale: schema evolution, YAML native typing, function
calling wire-format prior `{name, arguments}`.)

### 4.1 Example

```yaml
## ground:pins
- label: entry
  verb: file
  arguments: {path: "src/entry.py"}
  description: "start reading here"
- label: hot
  verb: file
  arguments: {path: "src/hot.py", range: "80-140"}
- label: py
  verb: glob
  arguments: {pattern: "src/**/*.py"}
- label: status
  verb: frontmatter
  arguments: {path: "FEATURE.md"}
  description: "check status field"
- label: layout
  verb: ls
  arguments: {path: ".", depth: 2}
```

### 4.2 Unknown Verbs and Unknown Arguments

If `verb` is not one of the known verbs in §5, the declaration is
preserved and its fenced result block in the frame renders:

```
error: unknown pin type: <verb>
```

Unknown verbs do not affect other pins, do not fail the frame, and are
preserved on rewrite.

If `arguments` contains keys not declared by the verb's schema (for
known verbs) or any keys at all (for unknown verbs), the unknown keys
are **preserved on rewrite, not rejected**. This is the same discipline
as unknown frontmatter keys (§3) — it lets verb schemas evolve without
breaking older readers. Validation may report unknown keys as warnings;
it must not fail the frame.

### 4.3 Label Conflicts On Pin

Adding a pin whose `label` already exists overwrites the existing entry.
This is the idempotent-overwrite ruling: pinning `label=l1` with file `a`
followed by pinning `l1` with file `b` leaves one pin with label `l1`
pointing at `b`.

## 5. Known Pin Types

Every pin type documents: `arguments` schema, expansion result, failure
modes. Failure modes always render `error: <reason>` inside the fenced
result block; the declaration in `## ground:pins` is preserved.

All path-typed argument values use the anchor syntax in §8.

### 5.1 `file` — single file (with optional line range)

**Arguments**:

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | Path to the file. Anchor syntax (§8) allowed. Bare absolute paths and `..` traversal escaping the anchor's subtree are rejected. |
| `range` | string | no | Line range, `N-M` (1-indexed, inclusive) or bare `N` (single line). |

**Expansion**: file content, sliced to `range` if given.

**Failure modes**: file not found; range out of bounds; binary file.

### 5.2 `glob` — matching path view (paths + metadata, no content)

**Arguments**:

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `pattern` | string | yes | Glob pattern (`*`, `**`, `?` per standard glob semantics). Anchor syntax (§8) allowed as prefix. |

**Filter**: matches are filtered through the project root's
`.gitignore` via the `pathspec` dependency.

**Expansion**: matched paths with `mtime` and `size` per entry. **No
file content is expanded.** This is a strong SPEC guarantee — a `glob`
matching thousands of files must not blow up the context window. Callers
who want file content use `file`.

**Empty match** renders `no matches` (not an error).

**Failure modes**: pattern rejected as unsafe (absolute, `..` traversal
escaping anchor subtree).

### 5.3 `frontmatter` — YAML frontmatter of a single file

**Arguments**:

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | Path to a markdown file with YAML frontmatter (or a bare YAML file). Anchor syntax (§8) allowed. |

**Expansion**: the frontmatter block verbatim (YAML). Body is not
included. The frontmatter of a single file is a bounded, small payload
by construction — no truncation needed.

**Failure modes**: file not found; file has no frontmatter block; YAML
syntax error inside the frontmatter; file cannot be parsed as markdown
or YAML.

### 5.4 `ls` — directory listing view (structure only, no content)

**Arguments**:

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | Path to a directory. Anchor syntax (§8) allowed. |
| `depth` | int | no | Traversal depth. `1` = current level only. Default: `LS_DEFAULT_DEPTH = 2`. |

**Filter**: entries are filtered through the project root's
`.gitignore` via `pathspec`.

**Expansion**: tree view, one line per entry. Files carry `mtime` and
`size`; directories are marked `<dir>`. **No file content.**

**Failure modes**: path not found; path is not a directory.

## 6. Frame Layout

Every frame renders the ground into `context_messages` with this
structure:

```
ground: <label> @ <absolute path>[ (doc: <doc path>)]
[chain: <ancestor GROUND.md paths, root-first>]
[$id: <id>]                                     # only if declared

<GROUND.md body, verbatim — @path lines visible as text>

```@<referenced-path-1>
<referenced doc 1 content>
```

```@<referenced-path-2>
<referenced doc 2 content>
```

<label1>:<verb>(<kwargs>) # <description>
<label2>:<verb>(<kwargs>) # <description>
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

- **head** — ground label, absolute path, optional `(doc: <path>)`
  annotation when the ground was opened with a non-default doc (§7.1).
  Optional `chain:` line listing ancestor `GROUND.md` paths root-first
  (dogfooding visibility only — chain body content is NOT inlined in the
  frame, it goes to the channel's instruction slot, §7.5). Optional
  `$id:` line if the frontmatter declares one.
- **body** — the GROUND.md body verbatim. `@path` references appear as
  text in this section (the references are expanded in the next section,
  not inlined here). This is where 场解说, promoted 胶囊, and 先例 live.
- **@-expansion blocks** — one fenced code block per `@path` reference
  found in body, in order of first appearance. The fence language tag is
  the `@`-prefixed referenced path. Content is the referenced doc's
  content (§6.1). Cycle detection prevents infinite expansion; depth and
  budget caps prevent context blowup.
- **declaration block** — one line per pin, with the shape
  `label:verb(key="val", key2="val2") # description`. Free text is not
  permitted in this block; free exposition belongs to body. Empty
  description omits the ` # description` suffix. Kwargs render as
  `key="value"` pairs (values always quoted); this matches the stored
  YAML `arguments` map verbatim, making display↔storage translation
  mechanical.
- **result block** — one fenced code block per pin. The pin's `label`
  is the fence language tag. Content is the pin's expansion per §5.

The pins YAML in `GROUND.md` and the declaration + result blocks in the
frame are two views of the same data. `GROUND.md` is authoritative;
frame rendering is derived.

### 6.1 `@`-reference Expansion

An `@`-reference in body loads another document as **static law**. The
referenced doc's content appears in the @-expansion blocks (§6), but it
is **not change-tracked**. Law follows the doc's current state silently;
changes are not announced. This is the sole dividing line between `@`
and `pin`: **对账 (accounting)**. `@` = load as law, no accounting.
`pin` = watch as gaze, with change accounting (§7.2). Guidance: stable
references → `@`; volatile targets → `pin`.

**Recognition rule**. An `@` is treated as a reference when:

- the `@` character is at the start of a line or preceded by whitespace,
- immediately followed by a path-start character (`[a-zA-Z0-9_./$]`),
- and not inside a fenced code block (``` ... ```).

The path token extends as a maximal run of `[a-zA-Z0-9_./$-]`, with
trailing `.` or `-` stripped. For paths containing whitespace or other
special characters, use the quoted form: `@"path with spaces.md"` or
`@'path with spaces.md'`.

An `@` not meeting these conditions (e.g., `@` in an email address, a
Python decorator inside a code block, or `@` mid-token) is treated as
literal text.

**Expansion rules**:

- **Default anchor**: `@`-references resolve against `$GROUND` (the
  GROUND.md's own directory) by default. Explicit anchors (§8) allowed:
  `@$HOME/...`, `@$CWD/...`.
- **Cycle detection**: a visited-set per expansion; if a doc has already
  been expanded in this chain, render `(@<path> already expanded above)`
  instead of re-inlining.
- **Depth cap**: `AT_MAX_DEPTH = 3` levels of nested `@`-references
  (a doc referenced by a referenced doc, etc.). Beyond the cap, render
  `(@<path> exceeds depth cap)`.
- **Budget cap**: `AT_BUDGET = 24000` chars (implementation may
  override). When exceeded, render remaining `@`-blocks as
  `(@<path> skipped: budget exceeded)` and emit a budget warning at the
  frame head (K20 discipline: report, don't silently truncate).
- **Failure modes**: referenced doc not found; referenced path escapes
  anchor subtree (§8). Render `error: <reason>` in the `@`-block.

## 7. Session Behavior

### 7.1 open / close

`Grounds.open(dir, *, label=None, doc=None) -> Ground`

- `dir` — the ground's root directory (pin anchor). Relative paths
  resolve against the owner's workspace root.
- `label` — short identifier, globally unique within the Grounds. `None`
  = derive from `dir` basename, with `-2`/`-3` suffixes on collision.
- `doc` — explicit GROUND.md path (law anchor). `None` = `dir/GROUND.md`.
  When `doc` points elsewhere, the **law anchor** (doc's directory; the
  chain starts there) decouples from the **pin anchor** (`dir`). This
  realizes K35's carry/local duality: `doc` is the portable law unit,
  `dir` is the local workplace. The frame head annotates
  `(doc: <path>)` when `doc ≠ dir/GROUND.md`.

**Idempotent by `dir.resolve()`**: opening the same directory twice
returns the same `Ground` instance; the second call's `label` and
`doc` (if provided) are ignored — the existing values win.

`Grounds.close(label)` removes the `Ground` from the collection. Files
on disk are not touched. A subsequent `open` on the same directory
rebuilds session state from `GROUND.md`.

### 7.1.1 `switch_to` (reserved)

Move = pop + enter. API: `Grounds.switch_to(dir, *, label=None, doc=None)
-> Ground` — atomic `close(active) + open(dir)`. **Does not trigger
drain** (unlike `close`); move is "shift attention", not "leave the
consciousness stream". Implementation window: after K19 drain lands and
Memento §14 `checkout(commit_id, moment_id)` completes. Not implemented
in v0.2.

### 7.2 observe / stale

Before each frame, all pins observe their targets in parallel:

- read the target's `mtime` and content `hash`
- compare against the ground's **runtime shadow** (in-memory; never
  persisted to `GROUND.md`)
- mark `changed on disk` when the hash differs from the shadow

On first observation (no prior shadow), nothing is marked `stale`. The
shadow is populated by the first observation and adopted as the initial
baseline.

The observation shadow (`seen_mtime`, `seen_hash`) MUST NOT be written
back to `GROUND.md`. The on-disk pins section only carries `label`,
`verb`, `arguments`, and optional `description`.

**CLI statelessness**: CLI invocations are one-shot
`open → render → close` cycles (§9). The shadow does not persist across
CLI calls — every CLI `frame` starts fresh, so stale marks never appear
in CLI output. Stale marking is a **session-level** feature, meaningful
only within a long-lived CTML channel session (K14 landing). CLI is a
diagnostic and bootstrap surface, not a session.

### 7.3 pin / unpin / update semantics

These are CTML-channel-layer verbs. The CLI does not implement them.
For a model or a human editing `GROUND.md` directly:

- **pin** — append a new item to the pins YAML list. If `label` conflicts,
  overwrite (§4.3).
- **unpin** — remove the item whose `label` matches.
- **update** — force a fresh observation of the target, then adopt its
  current `mtime` and `hash` as the new shadow baseline, clearing the
  `changed on disk` mark.

### 7.4 Nested Grounds

A ground's root directory may contain subdirectories that themselves
hold `GROUND.md` files. These are **peer grounds**, not children —
opening one creates an independent `Ground` instance with its own pins,
body, and frame.

- **Marks are inert**: the presence of `GROUND.md` in a subdirectory does
  not auto-open it. Grounds exist when opened, not when entered.
- **Inheritance via the filesystem**: a subdirectory ground's law chain
  (§7.5) naturally reads the ancestor `GROUND.md` bodies. No special
  composition rule is needed.
- **Pins never inherit** (K35): each ground's pins are its own. A pin in
  an ancestor ground does not appear in a descendant's frame.
- **Discovery is opt-in**: to find descendant grounds, pin a glob:
  `{verb: glob, arguments: {pattern: "*/GROUND.md"}}`. The glob returns
  paths + metadata only (no content) per §5.2. There is no
  `hint_children` convention flag — downward discovery is a pin
  (first-person gaze), upward inheritance is a rule (loading convention).
  The asymmetry is structural: pins are confined to the root subtree so
  they look down; the chain reads ancestors so it looks up.
- **Body is local**: a ground's body renders only in its own frame. Law
  that should propagate downward belongs in an ancestor `GROUND.md`
  body, which descendants will read via their chain.

Two grounds whose root directories overlap (e.g., `/repo` and
`/repo/src/foo`) are independent: each has its own pin set, its own
shadow, its own stale marks. A pin in `/repo` pointing at
`src/foo/bar.py` and a pin in `/repo/src/foo` pointing at `bar.py` gaze
at the same file through independent shadows — the model's choice, not
a conflict.

### 7.5 Law Chain

A ground inherits **law** from its ancestors: the `body` content of
`GROUND.md` files in parent directories of the ground's law anchor
(§7.1: the doc's directory, or `dir` if `doc` is default), walking up
to and including `$HOME` (`CHAIN_BOUNDARY = $HOME`). If `$HOME` is not
an ancestor, the walk continues to the filesystem root.

**Chain payload**: ancestor `GROUND.md` bodies only. The chain does
**not** carry:

- frontmatter (each ground's convention is its own)
- pins (K35: pins never carry)
- `@`-expansions are expanded in-place when the ancestor body is read,
  so transitively referenced docs are loaded (subject to §6.1 caps)

**Order**: root-first (most general law at the top, ground's own body
last). This matches the Claude Code `CLAUDE.md` chain convention.

**Destination**: chain content feeds the channel's `instruction` slot
at the K14 landing — the stable, cache-friendly context slot, distinct
from the volatile `context_messages` frame. The frame (§6) renders only
the ground's own body + `@`-expansions + pins; chain content is not
inlined in the frame.

**CLI rendering**: `moss ground frame` may emit a one-line chain
summary at the frame head (paths only, root-first) for dogfooding
visibility. Full chain content is a CTML channel concern.

**Self-containment**: the chain reads `GROUND.md` only — not
`CLAUDE.md`, `AGENTS.md`, or any other tool's convention files. Law
propagation requires explicit ground-ification of ancestor directories.
Escape hatch for referencing foreign docs: `@`-reference in body with
an explicit `$HOME` anchor (§8, §6.1).

## 8. Path Resolution

All paths in pins (`arguments` values) and `@`-references resolve
against one of three **anchors**:

| Anchor | Resolves to | Role |
|--------|-------------|------|
| `$GROUND` | The directory containing the loaded GROUND.md | Law anchor (default for bare relative paths) |
| `$CWD` | The directory passed to `open(dir=...)` | Pin anchor / local workplace |
| `$HOME` | The user's home directory | Machine-level escape hatch |

**Default**: a bare relative path (no `$`-prefix) resolves against
`$GROUND`. Rationale: document prior (markdown links resolve against the
document's location); in the default case (`doc = dir/GROUND.md`) the
two anchors coincide and the choice is invisible; the choice only bites
when `doc` is elsewhere, where `$GROUND`-relative is the distributable
default. Drift becomes opt-in via explicit `$CWD`.

**Explicit anchors**: `$GROUND/path`, `$CWD/path`, `$HOME/path`. `$CWD`
is the explicit float — for templates ("watch `$CWD/src/main.py` in
whatever project applies me"). `$HOME` is the sanctioned escape hatch
for machine-local references (`$HOME/.config/tool/config.yaml`); the
syntax makes the machine-dependence visible and grep-able.

**Per-anchor confinement (K12 restated)**: every path, after anchor
resolution and `..` normalization, must resolve **within its own
anchor's subtree**. Escaping `..` is rejected with
`PathOutsideRootError`. Symlinks are resolved (`Path.resolve()`) before
the subtree check — a symlink within the anchor pointing outside is
rejected when the target resolves outside. Bare absolute paths (no
anchor prefix) are rejected; use `$HOME` for machine-local references.

**Literal `$` escape**: `\$` in a path renders a literal `$` for the
rare case of a real filename beginning with `$`. (Impl detail; one SPEC
sentence.)

**Windows**: implementations on Windows map `$HOME` to `%USERPROFILE%`
(or the platform equivalent). `$GROUND` and `$CWD` are path-anchored
and platform-agnostic. The `$`-prefix syntax is retained on all
platforms for protocol uniformity (matches env-file / Docker / GitHub
Actions convention).

This single path grammar covers all four pin verbs (§5) and
`@`-references (§6.1). There are no verb-specific path rules.

## 9. CLI Surface

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

## 10. Language Neutrality

This SPEC defines the on-disk format for `GROUND.md` and the wire
semantics of frame rendering. Any runtime (Python, Rust, Go, Java, …)
that reads and writes `GROUND.md` according to this SPEC can participate
in the ground protocol.

Requirements for a compliant implementation:

- Parse and emit the three-segment file structure per §2
- Consume the reserved frontmatter keys per §3; preserve unknown keys
- Read and write the pins YAML per §4 (fixed envelope, `arguments` map);
  preserve unknown verbs and unknown arguments keys
- Handle all four known pin types per §5, with the specified failure
  modes rendered into the fenced result blocks
- Expand `@`-references per §6.1 with cycle detection, depth cap, and
  budget cap
- Emit frame layout per §6 (whitespace variations acceptable; block
  structure not)
- Resolve paths per §8 with per-anchor subtree confinement
- Implement the law chain per §7.5 (ancestor `GROUND.md` bodies,
  root-first, `$HOME` boundary)
- Never write observation shadow (`seen_mtime`, `seen_hash`) back to
  `GROUND.md`

## 11. Non-Goals

This SPEC does not cover:

- **Executable pin types** (`pin_bash` and family). Reserved for a
  future extension gated by T1 (cognitive interface carries no
  open-ended semantics). The `arguments` map accommodates them when
  the time comes without a format break.
- **L1 template instantiation** (`moss ground init` from a template
  library). Deferred; the current `init` command produces an empty
  scaffold only.
- **Cross-ground composition** (carried convention overlays, K35 detail
  conflicts, K40). The `doc` parameter (§7.1) realizes the carry/local
  duality at the file level; finer composition semantics remain under
  dogfooding.
- **Ground discovery from a well-known sidecar directory** (K43,
  `.grands/`). Recorded as a fallback route; not adopted here.
- **Shadow persistence across CLI invocations** (§7.2). Stale marking
  is session-scoped; CLI is stateless by design.
