# ANCHOR Format Specification

Any runtime that reads and writes anchors according to this SPEC can
participate in the anchor protocol.

## 1. Concept

An **anchor** freezes the key frames of a model or agent's calls into
cognitive anchor points — self-explaining files that are distributable,
rebuild state, and serve as the starting point of reasoning. An anchor is
an intelligence entity's reviewable past, and the means by which its
future self reviews its present.

It is the **reference frame** of cognition: the anchor holds still, new
information flows past, judgment comes from the collision.

An anchor is **not a checkpoint**. A checkpoint restores to origin; an
anchor stays fixed and observes change. See §9 for the core propositions.

The protocol is framework-agnostic and minimal. It defines a file format
(§3) and a data structure (§2), and nothing more — what `payload` contains
is the job of `ref` (§5), the protocol's single key proposition.

## 2. Data Structure

```python
class AnchorMeta(BaseModel):
    uid: str          # primary key (ULID). In meta, not filename.
    name: str         # human-readable name. Filename stem in file storage.
    description: str  # one-line note
    ref: str          # HTTP URL defining the payload structure (§5)
    created: datetime # ISO 8601 timestamp
    metadata: dict    # escape hatch — free-form, uninterpreted

class Anchor(BaseModel):
    meta: AnchorMeta
    payload: Any      # protocol-native data, structure defined by meta.ref
```

`Anchor` has two parts: `meta` (protocol layer, readable by every consumer)
and `payload` (protocol-native data, deliberately untyped here).

## 3. File Format

A single YAML file, two sections separated by a `---` document separator:

```
uid: 01JXXXXXXX...
name: anchor-utterance-u3
description: a failing frame from the utterance-end-detection benchmark
ref: https://github.com/GhostInShells/MOSShell/blob/main/src/ghoshell_moss/llms/funcs.py
created: 2026-08-10T21:48:00+00:00
metadata:
  model_generation: deepseek-v4-flash
  anchor_type: call-result
---
<payload — YAML-serialized, structure defined by ref>
```

1. **Section 1 — meta**: `AnchorMeta` fields flattened to top-level YAML
   keys. Unknown keys preserved, not interpreted.
2. **Section 2 — payload**: everything after `---`.

**Why `---`**: reading meta must not require parsing payload. A consumer
listing or indexing anchors reads up to the first `---` and stops. Payload
can be megabytes without slowing meta access.

## 4. Meta Fields

| Field | Required | Semantics |
|-------|----------|-----------|
| `uid` | yes | Primary key, a ULID string. Global, independent of storage location. |
| `name` | yes | Human-readable name. Filename stem in file storage. Not a key — collisions resolve by `uid`. |
| `description` | no | One line — what the anchor is, or the cognitive scene. |
| `ref` | yes | HTTP URL defining the payload structure (§5). |
| `created` | yes | ISO 8601 timestamp. |
| `metadata` | no | Escape hatch — free-form dict, uninterpreted. Suggested: `model_generation`, `anchor_type`, `labels`, `schema_version`. |

Implementations MUST NOT reject unknown meta keys.

**Why `uid` in meta, not filename**: a filename full of ids has governance
cost — `ls` shows half-opaque identifiers. Filename uses `name`; `uid` is
the key that stays unique under collisions and in database storage.

## 5. `ref` — The Key Proposition

The **only** constraint: `ref` points to **an HTTP URL**.

- Which URL exactly — raw, branch, tag, commit — is unconstrained.
- The URL points to the **definition of the payload structure**: code that
  explains the shape of the payload and thus how the call is reconstructed.
- A model can `curl` the URL and reconstruct the entire call, independent
  of any runtime interpreter.

This is code-as-prompt at the protocol layer: the description of the
payload is executable code, not prose to parse. `ref` makes an anchor
*independently reconstructible* by a model.

Two properties follow:

- **Public and language-agnostic**: a URL is publicly addressable; the code
  it returns is readable by any model.
- **Stable or versioned**: consumers needing long-term fidelity SHOULD pin
  the URL to an immutable address or record `schema_version` in `metadata`.
  Not mandated — a consumer decision.

## 6. Payload

`payload` is the protocol-native request data needed to reconstruct a call.
The protocol defines **no structure** for it — structure is determined by
`ref` and, optionally, `metadata.protocol`.

Producers MAY define their payload as a self-explaining model
(code-as-prompt) that declares its own `ref`, mirroring how typed
declarations work elsewhere in MOSS. The protocol itself stays agnostic.

## 7. Discovery

Anchors are discovered by **filename pattern**, mirroring SKILL.md:

- Suffix convention: `.anchor.yml`
- Glob: `**/*.anchor.yml`
- Filename `name` is human-readable; `uid` is the identity authority.

Discovery tools are implementation-specific. The protocol defines only the
file convention.

## 8. Read / Write

The protocol defines the **on-disk contract** (§3, §4) and nothing more.

- **Write**: serialize `meta` → section 1, `payload` → section 2.
  A reference implementation may exist; the contract class may carry one
  as a self-explaining sample.
- **Read**: parse §3/§4. Reading is protocol-agnostic — the reader does
  not need to know any payload structure to read meta. Consumers implement
  reading themselves, or assume a library exists.

The protocol does not dictate the write/read API surface.

## 9. Core Propositions

Three propositions define the anchor's evaluation criteria:

1. **Productivity, not fidelity** — judged by whether "anchor + new
   information" produces new judgment, not by replay fidelity.
2. **Preserves the doubt structure** — unresolved questions and punted
   forks, not just the decision. An anchor without doubt is invalid.
3. **Naming is semantic enforcement** — "anchor" is a reference frame
   (observe delta), not a checkpoint (restore to origin). The naming must
   not regress in code or docs.

## 10. Compliance

A compliant implementation must:

- Recognize an anchor as `*.anchor.yml` (§7)
- Parse and emit the two-section YAML per §3
- Read/write `meta` per §4; preserve unknown keys; never reject an anchor
  for an unrecognized meta key
- Treat `ref` as an opaque HTTP URL (§5) — never interpret it as a path
- Treat `payload` as opaque data defined by `ref` (§6)
- Never require payload-structure knowledge to read meta (§3)

## 11. Versioning

The format is versioned by **file convention**, not a version field:

- Breaking changes are signalled by a suffix/naming change
  (e.g. `.anchor.yml` → `.anchor.v2.yml`)
- `metadata.schema_version` MAY record the payload schema version when a
  producer must distinguish payload structures under the same `ref`

No self-version field. Versioning by filename convention keeps meta minimal
and detection cheap and glob-friendly.
