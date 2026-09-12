# Blueprint — the blueprint of MOSS architecture modules

`blueprint` is the **blueprint** of MOSS architecture modules: it defines each module's
boundary, contract, and wiring, and is the **authoritative surface** consumed by
`moss codex`, IoC assembly, and runtime code-as-prompt alike.

Most of what lives here is a **facade** (marked `Facade`: you hold it and call it,
you do not inherit it). The rest is **abstract** (marked `Abstract`: you inherit it
to implement a new type). The real implementations are not in this directory —
blueprint defines *what it is and how to use it*; implementations own *how it works*.

## Code as Prompt discipline

The docstrings on this surface are the authoritative surface consumed by both models
and model-developers. Four rules:

1. **A readable, self-explanatory authoritative surface.** The surface stands on its own;
   it is readable without any external document.
2. **Usable without reading the implementation source**, especially with IoC. A consuming
   model's first entry point is the surface, not the implementation. Representative case:
   `channel_builder.py`.
3. **An index into the related implementation.** The surface answers "what is here and
   where to look"; implementation details are pointed to from the surface, not copied into it.
4. **Wiring and control flow are self-explanatory on the surface.** Control-flow code
   exposed on the surface may be replaced inside the implementation. The surface promises
   the shape of the control flow; the implementation promises replaceability.

Corollary: **decision history does not belong on the surface.** The surface states
*what it is, how to use it, what the contract is*; *why it was decided this way, what was
rejected, what state the work is in* belongs to FEATURE.md and `git log`.

## Do not couple the surface to the feature system

The following must **not** appear in docstrings or module-level comments:

- `see xxx FEATURE.md` / `see workstream xxx`
- decision numbers (`Decision N` / `KD N`), whether orphaned or with a citation
- progress language ("design is locked", "already landed", "not implemented this cycle",
  "hook not yet provided")
- historical narrative ("back when engineering complexity was low…", "kept from the vN
  refactor because…")

**Why:** FEATURE.md is not authoritative (features specification: "Not authoritative over
code"), and it goes stale, gets compacted, and is not shipped (`.ai_partners/` is outside
the packaging scope in `pyproject.toml`). Reverse lookup goes through `git log -- <file>` —
the spec already declares that path; forwarding it again from the source is redundant and
rots.

Where the content goes:

| Content | Destination |
|---|---|
| Contracts and invariants needed to use the abstraction | keep in the docstring (drop the citation) |
| Rationale a reader on-site genuinely needs | `#` comment |
| Decision trail, rejected alternatives, progress | drop it (FEATURE.md + `git log` are already the index) |

**Pointers to live code are fine** (`see channel_builder`, `see core/concepts/qa.py`) —
the test is whether the target rots, not whether a pointer exists.

## docstring vs comments

**docstring** (reflected by `get-interface`; visible to models and developers):

- what this is and how to use it
- contracts required for use: invariants, parameter semantics, negative constraints ("must not X")
- relationships to other abstractions, IoC wiring

**`#` comments** (not in `get-interface` output; visible when reading the source):

- implementation details and rationale
- temporary workarounds, known traps
- design rationale that genuinely must sit on-site
- TODOs still in force

**Test:** if you delete the text, can a consumer still use this class/function correctly?
No → docstring. Yes → `#` comment, or delete it.

## Language

The primary language for docstrings is English. Simple descriptions, parameter docs and
one-line contracts are written directly in English. Prose that carries heavy architectural
philosophy may stay in Chinese — but every module's top-level docstring carries an English
intro, so the surface stays usable at its entry point.

Architecture-critical terms carry a Chinese gloss on first use; ordinary technical terms
do not. Do not stack one piece of content in two languages.

When in doubt about what to translate, or how, align with the human first.
