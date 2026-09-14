# FRAME Format Specification

A runtime that reads and writes `*.frame.md` files according to this SPEC
participates in the frame protocol.

## 1. Concept

A **frame** is a question set that a model resolves from its own context, to hold a
stable situational understanding across context loss (compaction, a new window, a
different model instance).

The questions are the asset; the answers are working state. Asking the right
question makes an answer that already exists in the context explicit and
ready to reason on. A frame is an **extraction** device, not a decision device:
every question must be answerable from context.

## 2. File Identity

- A file is a frame **if and only if** its name ends with `.frame.md`.
- Frames are discovered by scanning for that suffix under a **root path** chosen by
  the embedder. There is no registry and no explicit link between frames — the
  filesystem is the index. A model finds frames by listing the root.
- The **label** of a frame is its path relative to the root, with the suffix
  stripped (posix). `<root>/debug/concurrency.frame.md` has label
  `debug/concurrency`.

## 3. File Structure

Two segments: an optional YAML frontmatter block, then a body.

```
---
description: <one line, self-explanatory>
nexts:
  - <path to another frame, resolved against this file's directory>
---

<question paragraph>

<question paragraph>
```

### 3.1 Frontmatter

- `description` — one line. What the frame orients. The filename is expected to be
  self-explanatory as well.
- `nexts` — optional. Paths to other frames, resolved against this file's directory.
  A **hint** for where to go once this frame is complete. It may be a single branch
  or a multi-choice menu; the protocol does **not** choose — the hint is surfaced
  when the frame completes, and the model decides.

### 3.2 Body — questions

The body is a sequence of paragraphs separated by blank lines. **Each paragraph is
one question.** Order is the index: the first paragraph is question `0`.

There is no other structure in the body. Every paragraph is a question.

## 4. Resolution

- A question is **resolved** when the runtime has recorded an answer for it — any
  text, including an explicit `unknown`.
- A frame is **complete** when every question is resolved.
- `unknown` is a first-class answer, not a failure. The unresolved set is the frame's
  most valuable output: it names what the model does not yet know about its situation.

## 5. Authoring

- Questions must be extractable from context. "What should I do next?" is not a frame
  question — it is a reasoning question, and it belongs to reasoning, not to the frame.
- Keep questions short. A frame's content is re-presented across context loss, so its
  length is a recurring token cost.
- A frame is a schema; keep it stable. Answers may change; questions should not.
