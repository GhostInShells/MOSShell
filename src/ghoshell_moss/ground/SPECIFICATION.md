# GROUND Format Specification

Any runtime that reads and writes `GROUND.md` according to this SPEC can
participate in the ground protocol.

**Draft 0.3.0** — this revision removes the "law" vocabulary, makes position a
first-class input to rendering, and moves child-ground expansion into the
ground itself.

## 1. Concept

A **ground** is a cognitive field bound to a directory. A directory is a
ground **if and only if** it contains a `GROUND.md` file. There is no
"bare-directory" ground — the marker is the boundary.

A ground carries three things:

- a **convention** — the frontmatter block: identity and machine declarations
- a **body** — open-set markdown that explains this directory to whoever reads
  it. It is the ground's own self-explaining content. It is not called "law",
  and it is not inherited by anything.
- **pins** — first-person gaze declarations pointing at targets in the world

Nothing else is inherited. A directory without `GROUND.md` is not governed by
an ancestor's *content*; it is merely **rendered through** the nearest ancestor
ground (§6.1). That is a choice of which face to draw, not an inheritance of
body, frontmatter, or pins.

The protocol's job is to let a directory declare its own cognitive field. How
that field is *programmed* is the ground author's business — the protocol does
not hardcode a harness convention (unlike `AGENTS.md` / `CLAUDE.md`, which are
defined by their consumers).

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
inline:
- <relative directory path>
---

<body — free-form markdown>
```

1. **frontmatter** — YAML between the `---` fences. Reserved keys: `$id`,
   `name`, `description`, `pins`, `inline`, `ignore`, `ignore_file`.
   Unknown keys are preserved on write.
2. **body** — everything after the closing frontmatter fence. Open set.

The `pins` key holds the YAML list of pin declarations (§4). If absent, the
pins list is empty.

A directory without `GROUND.md` is not a ground.

## 3. Reserved Frontmatter Keys

| Key | Type | Semantics |
|-----|------|-----------|
| `$id` | string | Identity claim, URI-shaped. The ground *claims* an identity; resolution is upper-layer's job. Optional; any string accepted. |
| `name` | string | Display name. Defaults to the directory basename. |
| `description` | string | One-line description of this ground. Optional. |
| `pins` | list | Pin declarations per §4. Each entry has the fixed envelope: `label`, `verb`, `arguments`, `description`. |
| `inline` | list of strings | Child grounds **this ground renders inside its own face** (§6.6). Paths are relative to the ground root; each must carry its own `GROUND.md`. Consumed by the ground that declares it. Optional; absent = no children rendered. |
| `ignore` | list of strings | Ground-level ignore patterns — `.gitignore` semantics, relative to ground root. All discovery pins (glob, frontmatter pattern, ls) automatically respect these rules. Optional. |
| `ignore_file` | string | Path to a file (relative to ground root) containing additional ignore patterns, one per line. Merged with `ignore` inline list. `.gitignore` or `.groundignore` are expected names. Optional. |

Implementations MUST NOT reject unknown frontmatter keys.

### 3.1 `inline` is a render directive, not a registration

`inline` answers exactly one question: *which of my child grounds do I draw
inside my own face?*

- It is **not** "all child grounds". A subdirectory with `GROUND.md` that is
  not listed is not rendered by its parent — it is still a ground, reachable by
  rendering it directly. Presence on disk is not presence in the face.
- It is **not** a container registration. There is no separate object that
  "materializes" children; the declaring ground renders them.
- It is one level: the listed child renders *its own* face, which may in turn
  render *its* `inline`. Recursion is per-ground, not a container's sweep.

### 3.2 Ground-Level Ignore

`ignore` and `ignore_file` declare exclusion rules at the **ground level** —
one set of rules for the entire ground. All discovery pins (`glob`,
`frontmatter` pattern, `ls`) automatically respect them; `file`, `exec`, and
`content` pins are unaffected. No per-pin opt-in or opt-out is needed.

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
| `verb` | yes | Pin type: `file`, `glob`, `frontmatter`, `ls`, `exec`, `content`, or future verbs. |
| `arguments` | no | Keyword arguments for the verb. Default `{}`. Schema depends on `verb` (§5). |
| `description` | no | One-line marginalia. This is the progressive-disclosure carrier (§6.2): a reader learns *why* a pin exists without opening `GROUND.md`. Long exposition belongs in body. |
| `always_show` | no | Boolean, default `false`. In walk mode, `$CWD`-anchored pins expand; others fold to a compact table of contents. `always_show: true` keeps a pin expanded in both faces. |

The envelope is **monomorphic** — polymorphism is quarantined inside
`arguments`. Tools that don't understand a verb can still parse, list, and
round-trip the pin.

**Unknown verbs**: preserved, not rejected. Their rendered expansion reports
the verb as unknown. They do not affect other pins.

**Unknown arguments keys**: preserved on rewrite, not rejected.
Validation may warn; it must not fail.

**Unknown envelope keys**: dropped on rewrite — the envelope is fixed.

**Label conflict**: adding a pin with an existing `label` overwrites the old
entry (idempotent overwrite).

### 4.1 Per-Pin Budget Parameters

Every pin MAY carry three optional budget fields in `arguments`:

| Field | Type | Semantics |
|-------|------|-----------|
| `budget` | int | Content character limit. When exceeded, output is truncated with a visible marker. Applies to content-emitting verbs (`file`, `frontmatter`, `exec`). |
| `limit` | int | Entry count limit. When exceeded, output shows `N of M` entries. Applies to list-emitting verbs (`glob`, `ls`, `frontmatter` with pattern). |
| `max_depth` | int | Recursion depth in directory levels below the pattern's static base. `1` = one layer of sub-fields (the filename is not counted). When absent, recursion is unbounded. |

All three are optional. Each verb declares which of these it supports in §5.

`max_depth` is a pure depth cap. Field-boundary stop (防穿透 — do not recurse
into a directory that directly contains a match) is a separate, per-verb
semantic: `frontmatter` pattern mode applies it, `glob` does not.

## 5. Known Pin Types

All path-typed arguments use the anchor syntax in §8.

### 5.1 `file` — single file (with optional line range)

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | File path. Anchor syntax (§8) allowed. |
| `range` | string | no | Line range: `N-M` (inclusive) or bare `N`. |
| `budget` | int | no | Content char limit (§4.1). |

**Expansion**: file content, sliced to `range` if given.

**Failure modes** — visible, not silent:

- File does not exist: renders `[missing]`.
- File exists but is binary: renders `[binary file, not rendered]`.
- Range out of bounds or descending: renders `error: invalid range`.
- Read failure: renders `error: unreadable`.

### 5.2 `glob` — matching path view (paths + metadata, no content)

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | Glob path (`*`, `**`, `?`). Anchor syntax (§8) allowed as prefix. |
| `limit` | int | no | Entry count limit (§4.1). Default: implementation-defined. |
| `max_depth` | int | no | Recursion depth cap for `**` patterns (§4.1). |

Matches are filtered against a built-in set of known noise directory basenames
(`.git`, `.venv`, `__pycache__`, `node_modules`, etc.). Ground-level `ignore`
rules (§3.2) apply as an additional filter. **No file content is expanded** —
a `glob` matching thousands of files must not blow up the context window.

**Expansion**: matched paths with human-readable size per entry (`12K`,
`1.2M`). `mtime` is not rendered.

**Empty match** renders an empty result (not an error).

### 5.3 `frontmatter` — YAML frontmatter of markdown file(s)

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | File path, or glob pattern matching multiple files. Anchor syntax (§8) allowed. |
| `keys` | list[string] | no | Frontmatter keys to extract. Absent = full frontmatter block. |
| `budget` | int | no | Content char limit (§4.1). |
| `limit` | int | no | Entry count limit when `path` is a pattern (§4.1). |
| `max_depth` | int | no | Recursion depth in directory levels when `path` is a pattern (§4.1). |

**Single-file mode** (`path` is a concrete file path): extracts the full
frontmatter block verbatim. Body is not included.

**Pattern mode** (`path` contains glob characters `*`, `?`, `[`): matches
multiple files. Each matched file's frontmatter is rendered as an independent
result block, labeled by file path. Subject to `max_depth`, ground-boundary
stop (防穿透 — a directory that directly contains a match is not recursed
into), and ground-level `ignore` rules (§3.2).

This is the **discovery** mechanism: a single `frontmatter` pin with
`path: "*/GROUND.md"` reveals the identities of every child ground without
rendering any of them. Discovery and rendering are different acts — a
discovered child is not a rendered child (§3.1).

**keys** filtering: when specified, only the named frontmatter keys are
rendered. Unknown keys are preserved.

**Failure modes**: file not found; no frontmatter block; YAML syntax error;
pattern matches zero files (not an error — renders empty).

### 5.4 `ls` — directory listing (structure only, no content)

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `path` | string | yes | Directory path. Anchor syntax (§8) allowed. |
| `depth` | int | no | Traversal depth. Default `2`. |
| `limit` | int | no | Entry count limit (§4.1). |
| `max_depth` | int | no | Recursion depth cap (§4.1). Alias for `depth` — whichever is smaller wins. |

Entries filtered against a built-in set of known noise basenames. Ground-level
`ignore` rules (§3.2) apply as an additional filter. **No file content.**

**Expansion**: tree view with human-readable size per file (`12K`, `1.2M`).

**Failure modes**: path not found; path is not a directory.

### 5.5 `exec` — invoke a ground-authored executable

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `ref` | string | yes | Relative path to an executable file **within the ground subtree**. |
| `mode` | string | no | Interpreter mode: `shebang` (default, requires `+x`) / `python` (`sys.executable`) / `shell` (`sh`). |
| `timeout` | float | no | Seconds. Default `10`, max `60`. |
| `budget` | int | no | Content char limit (§4.1). Applied to captured stdout. |

**Authorization model** — this is the only pin verb that executes code. It is
deliberately narrow:

- `ref` MUST be a **relative path**. Absolute paths are rejected.
- `ref` MUST resolve inside the ground subtree. `..` traversal outside the
  ground is rejected.
- In `shebang` mode the target file MUST have the executable bit set (`+x`);
  files without it are treated as missing.
- **The protocol never accepts inline shell strings.** A pin cannot say "run
  this command"; it can only reference a script the ground author has committed
  to the ground subtree.

**Execution environment**: `cwd` = `$GROUND`; `GROUND` and `CWD` injected as
absolute paths; `stdin` = `/dev/null`.

**Expansion**: captured `stdout`, subject to `budget` truncation.

**Failure modes** — visible, not silent:

- File does not exist: `[missing]`.
- `shebang` mode without `+x`: `[not executable]`.
- Absolute path or `..` escaping the subtree: `[outside ground]`.
- Non-zero exit: output followed by `[exit N]` and up to 5 lines of stderr tail.
- Timeout: partial stdout followed by `[timeout after Ns]`.
- Exceeding the render's wall-clock budget (§7.2): the pin renders with
  `status="timeout"` and no content — the render as a whole still completes.

**Observation**: `exec` is a compute-on-observe verb. One render = at most one
process per pin. Renders are cold reads (§7.1): a second render runs it again.

### 5.6 `content` — foreign content-file chain (compatibility)

> Renamed from `law`; the noun is gone from the vocabulary, the mechanism is
> unchanged. The verb is *the compatibility escape*, not a concept of the ground.

| Key | Type | Required | Semantics |
|-----|------|----------|-----------|
| `filename` | string | yes | Convention file name (e.g. `CLAUDE.md`, `AGENT.md`). Collected from `$CWD` upward. |
| `budget` | int | no | Total char limit across collected files. Truncates with marker (§4.1). |
| `lines` | int | no | Total line limit across collected files. Truncates with marker. |

**Semantics**: this pin pulls *documents*, so the argument is a **filename**,
not a path. From the current `$CWD`, it walks upward collecting every
occurrence of `filename`, stopping at the ground root (`$GROUND`) — the
boundary is the ground, never `$HOME`. The chain renders parent-first
(root → cwd), each file as an independent result block labeled with its path
relative to the ground root.

This is the compatibility mechanism for foreign project conventions: the ground
declares a `content` pin pointing at the foreign filename, and the
implementation reads it — the foreign project's own files are never modified
or renamed.

**Positional view**: `content` depends on `$CWD`. Walking into a
subdirectory re-collects the chain from the new position.

**`@`-reference expansion**: each collected file's body may contain
`@`-references (§6.5), resolved relative to that file's own directory. Only one
level is expanded. Fenced code blocks are skipped.

**Failure modes**: no matching file in the range renders an empty result.

## 6. Rendering

### 6.1 Position resolves the face

Rendering always takes a **position** — a directory — and never "a ground
object". The first step is always:

1. If the directory has its own `GROUND.md`, that ground is the face
   (**field-root face**, §6.2), and `$CWD` is its root.
2. Otherwise, walk upward to the **nearest** ancestor `GROUND.md` and render it
   with `$CWD` = the given directory (**walk face**, §6.3).
3. The upward walk MUST stop at the renderer's **boundary**. A directory above
   the boundary is never consulted; if no ground is found inside it, there is
   no ground to render.

The boundary is supplied by whoever asked for the render (a channel's home, a
CLI invocation's root). It is not `$HOME` and not the filesystem root.

**No merge.** Finding an ancestor does not merge anything: the ancestor's body
is not appended to the child's, and pins are never inherited across grounds.

### 6.2 Field-root face

Rendered when the position is a ground's own root:

1. **header** — identity: `$id`, `name`, `description`, `$GROUND`
2. **body** — the `GROUND.md` body verbatim (§6.5 for `@`-references)
3. **pins** — every pin expanded per its verb, in declaration order
4. **children** — every ground listed in `inline`, rendered recursively
   (§6.6), in declaration order
5. **templates** — the discovered template list (§6.7), when non-empty

Every result is labeled by its pin's `label` and carries its `description` as
marginalia — that is what makes the pin's *reason* visible without opening the
file.

### 6.3 Walk face

Rendered when the position is inside a ground but is not its root:

1. **header** — identity plus `$CWD` (the position)
2. **no body** — walking one directory deeper must not re-deliver the whole
   body; the body belongs to the ground's own root face
3. **`$CWD`-anchored pins** — expanded full (this is the point of the walk:
   pins declared at the root follow the viewer)
4. **`$GROUND`-anchored pins** — folded into a compact table of contents
   (label, verb, arguments one-liner, description), unless `always_show`
5. **`content` pins** — compact by default: the collected paths only,
   unless `always_show`
6. **children** — `inline` is still rendered (§6.6): the children belong to
   the ground, not to the position

No new ground is opened by walking. The law anchor is untouched.

### 6.4 Reference rendering: the self-explaining container

The protocol does not mandate a rendering format, but a renderer that delivers
into a model's context should use **the reference shape** below. It is an XML
container: nesting is expressed structurally, and content is escaped
(`&` → `&amp;`, `<` → `&lt;`) so that nothing inside can forge structure.

```xml
<ground path="/abs/ground/root" name="Ghost Ground" description="…">
  <cwd path="/abs/ground/root/journal"/>      <!-- walk face only -->
  <body>
    …GROUND.md body, @-references expanded in place…
  </body>
  <pin verb="frontmatter" label="subfields" desc="子认知场存在性">
    …pin result…
  </pin>
  <ground path="existence" name="existence" description="…">
    …this child's own face, recursively…
  </ground>
  <folded>
    <pin verb="glob" label="scratch" desc="…"/>
  </folded>
  <templates>
    <template name="review" desc="…"/>
  </templates>
  <truncated reason="budget" path="works"/>
</ground>
```

Rules that make it self-explaining:

- **Same shape at every level.** A nested `<ground>` is byte-identical to
  rendering that directory standalone, apart from the path prefix. A reader who
  understands one level understands every level.
- **Different kinds look different.** A nested ground is a `<ground>`. A pin
  result is a `<pin>`. A fold is a `<folded>` list of empty `<pin/>` markers.
  A truncation is `<truncated/>`. Nothing is "a block" that could be any of
  them.
- **Every stopping point says why.** A node that did not render says so, and
  the reason is data (`reason="budget"`, `reason="timeout"`), not prose.
- **Escaping is mandatory.** Ground content is data. It never becomes structure.

### 6.5 `@`-reference Expansion

An `@`-reference in body loads another document **in place, as static
content**. It is not change-tracked; the loaded content reflects the file's
current state at render time.

**Recognition**: an `@` at line start or after whitespace, followed by a
path-start character `[a-zA-Z0-9_./-]`, and not inside a fenced code block. The
path runs as a maximal token of `[a-zA-Z0-9_./-]`. Quoted form for paths with
special characters: `@"path with spaces.md"`.

**Expansion rules**:

- Resolves against `$GROUND` by default; no anchor syntax — the reference is a
  plain relative path
- **Single-level**: one level of `@`-expansion is applied; resolved content is
  not re-scanned.

**Failure modes**: doc not found; path escapes the ground root.

### 6.6 Recursive child rendering

A ground's `inline` (§3.1) lists the child grounds it draws inside its own
face. For each entry, in declaration order:

1. Render the child's own face (field-root face — the child is a root of its
   own face, regardless of where the viewer is standing).
2. Nest that face inside the parent's container (§6.4).
3. A missing or broken child renders as a `<truncated>` node with its reason;
   it does not abort the parent's render.

Recursion is bounded by the **budget**, not by a depth constant: a render
carries a byte budget (§7.2), and when it runs out the last node is a
`<truncated reason="budget" path="…"/>`. Entries after the cut are not
rendered — the marker names where the face stopped, so the reader knows exactly
what to render next.

### 6.7 Render tables — a template, or a target `.md`

A **render table** is a table that is not written into the ground being viewed:
a markdown file whose frontmatter may carry `name` / `description` and whose
body and pins are used *instead of* the target ground's own. `GROUND.md` is
itself a render table — a template is the same thing, held outside the
directory it is used against.

- **Discovery**: templates live in `.grounds/` under the ground root
  (`<root>/.grounds/<name>.md`, relative paths allowed). Discovery is
  automatic — **no pin and no command is needed to perceive them**: a render of
  a ground with a non-empty `.grounds/` carries a `<templates>` node listing
  `name` + `description`.
- **Use — two forms, one meaning**:
  - `template=<name>` — a discovered template, looked up in `.grounds/`.
  - `template=<path>.md` — a target markdown file, addressed directly.
  Both render the position through that table. The directory's own body and
  pins are not used; `@`-references inside the table's body still expand (§6.5).
- **Why**: it is how an author tests what a change to `GROUND.md` would look
  like, and how one directory can be viewed through several tables without
  editing it.
- **Boundary**: a template path MUST resolve inside the renderer's boundary
  (§6.1). Template content is copied, never referenced.
- **Nothing is written**: rendering through a table is a lens, not a patch.

Identity comes from the target directory when it is a ground; otherwise from
the directory basename.

**Failures are results.** A render never raises out to the caller. If the table
cannot be used — file missing, YAML broken, template name unknown, path outside
the boundary — the render returns an error result that says so:

```xml
<ground path="/abs/ground/root" status="error" reason="template not found: review"/>
```

The same rule applies to a position that resolves to no ground at all:
`<ground path="…" status="empty" reason="no GROUND.md up to boundary"/>`.
An error face is a message to the reader, not an exception to the caller.

## 7. Session Behavior

### 7.1 Renders are cold

A render is a **cold read**: it re-reads `GROUND.md`, re-runs discovery pins,
and re-executes `exec` pins. There is no change detection, no cached "changed"
flag, and no background refresh — an implementation that wants a fresh face
asks for a render. Callers that need a snapshot hold one; the protocol does not
pretend a stale face is current.

### 7.2 Render budgets

Two budgets bound one render. Both are properties of the render, not of any
single pin:

- **byte budget** — total rendered characters. When exhausted, the current node
  is replaced by `<truncated reason="budget" path="…"/>` and rendering stops.
- **wall-clock budget** — total render time. When exhausted, **running pins are
  not cancelled mid-node**: each `exec` pin still obeys its own `timeout`, and
  a pin that cannot answer within the remaining render budget renders as
  `<pin … status="timeout"/>`. The render as a whole still completes and
  delivers what it has.

Both defaults are implementation-defined but MUST be finite.

### 7.3 Governance: nearest ground, no merge

A directory without `GROUND.md` is rendered through the **nearest** ancestor
ground. That is a statement about which face to draw:

- the ancestor's body is not appended to anything
- frontmatter and pins are not inherited
- a ground that is not the nearest does not participate, even if it is higher up

## 8. Path Resolution

Three anchors:

| Anchor | Resolves to | Role |
|--------|-------------|------|
| `$GROUND` | Directory containing GROUND.md | Content anchor (default) |
| `$CWD` | Current viewing position (directory) | Pin anchor / workplace |
| `$HOME` | User home directory | Machine-local escape hatch |

Bare relative paths default to `$GROUND`. Explicit anchors:
`$GROUND/path`, `$CWD/path`, `$HOME/path`.

**Subtree confinement**: every path, after anchor resolution and `..`
normalization, must resolve within its anchor's subtree. Symlinks are resolved
before the check. Bare absolute paths are rejected — use `$HOME` for
machine-local references.

**Boundary**: `$GROUND` and `$CWD` MUST both resolve inside the renderer's
boundary (§6.1). A pin whose target escapes the boundary fails visibly
(`[outside ground]`) rather than silently reading elsewhere.

`\$` escapes a literal `$` in filenames. Windows maps `$HOME` to
`%USERPROFILE%`; `$GROUND` and `$CWD` are platform-agnostic.

## 9. Compliance

A compliant implementation must:

- Recognize a directory as a ground iff it contains `GROUND.md` (§1)
- Parse and emit the two-segment file structure per §2
- Consume reserved frontmatter keys per §3; preserve unknown keys
- Read and write pins per §4 with the fixed envelope; preserve unknown verbs
  and arguments keys
- Read `inline` as a render directive and render exactly the listed children
  recursively (§3.1, §6.6); never render an unlisted child
- Resolve the face by position, bounded by the renderer's boundary (§6.1)
- Render the field-root face (§6.2) and the walk face (§6.3) with the
  differences stated there — in particular, **walk renders no body**
- Support per-pin `budget` / `limit` / `max_depth` parameters per §4.1
- Handle all six known pin types per §5, rendering failure modes into results
- Support `content` upward collection per §5.6 — bounded by the ground root,
  root-first display, one-level `@`-expansion, truncation
- Support `frontmatter` in both single-file and pattern modes (§5.3)
- Render human-readable file sizes, omit raw `mtime`
- Expand `@`-references per §6.5 (single-level, no recursion)
- Resolve paths per §8 with per-anchor subtree confinement and boundary checks
- Expose discovered templates in the render (§6.7) without requiring a pin
- Treat renders as cold reads (§7.1) and bound them by both budgets (§7.2)
