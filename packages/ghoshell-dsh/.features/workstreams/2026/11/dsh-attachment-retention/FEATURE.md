---
title: Dsh Attachment Retention
status: draft
status_note: "options recorded 2026-11-07; no design chosen yet"
priority: P1
created: 2026-11-07
updated: 2026-11-07
depends: []
milestone: dsh 0.2.0-rc.2
description: >-
  dsh keeps every attachment object forever, and nothing outside .dsh can address or
  govern it. This workstream keeps the measurement and the candidate designs on record;
  no design is chosen yet.
---

# Dsh Attachment Retention

## Motivation

The ghost is about to consume screenshots continuously. dsh's attachment store has no
retention, no quota, and no delete path: every byte that ever entered a session log stays
under `DSH_HOME` forever, and nothing outside `.dsh` can address it. Measured on the ghost
home, 2026-11-07:

| category | size | objects | referenced by a session log |
|---|---|---|---|
| `sessions/` | 342MB | 595 logs | — (append-only; `truncate` only repairs a torn tail) |
| `attachments/v1/objects` | 102.8MB | 698 | **698 (100%)** |
| `attachments/v1/request-images` | 7.6MB | 188 | 4 |
| `cache/attachments` | 18MB | 206 | **0** |
| `storages/session_projcache` | 43.9MB | 596 | 1:1 with sessions |

Image objects grow 1MB (10-03) → 77MB (10-22) → 102MB (10-28): about +14MB on an active day,
at a handful of screenshots per day. Mean object 144KB, max 1.5MB. At one frame per second the
same path stores ~12.4GB/day. As thirdgerb put it: "如果底层 dsh 还是个玩具, 代价会很大."

Upstream status — `dsh-attachment-local/README.md` states:

> Retention and garbage collection are deferred because resumed and forked sessions may share
> immutable objects, and a backend serving remote runtimes or shared storage would need its own
> durability proof. Both directions are undecided; the local storage currently retains every
> object under `DSH_HOME`.

The newest vendored generation (`.vendors/dsh-research/v0.2.2-alpha.2+ce7073d`) still carries
this text, and its source contains no quota, TTL, sweep, or delete call for stored objects. An
upgrade alone does not solve it.

## Options — candidates, not decisions

### O1 — Truth layer outside, as a governable resource (thirdgerb)

> "真相层在外部才是核心关键, 使得外部真相变成可治理对象 (resource 化). dsh 内部的 objects
> 是不可治理的, 这是它设计上的错误."

Operational reading: governable = **stable address + an owner of the lifecycle + a degradable
read**.
- address: a path in a governed directory (project `assets/` or equivalent), not a digest that
  only `.dsh` can resolve;
- owner: whoever owns that directory decides deletion — the loss becomes visible and reversible
  by policy, not silent;
- degradable: a missing file yields a value ("file no longer exists"), not a failed request.

Note: `ImageAttachmentRef.attachmentId` is documented as "never a filesystem path", so a
mapping layer carries id → path; the log format stays unchanged. The human analogy thirdgerb
gave: "地址如果变了, 很多东西是召回不了的" — the reverse is the design goal: keep the address
stable and owned, and recall always has an entry point.

### O2 — Cache is an LRU working set; restore comes from the external filesystem (thirdgerb)

> "本质上对 cache 的引用是 lru 问题, cache 的还原是从外部文件系统."

- Refcount is rejected (see O5): a miss costs one rebuild, so the question is "worth keeping",
  not "safe to delete".
- Two derived layers exist today: `attachments/v1/objects` (normalized image) and
  `cache/attachments/request-images` (request variants). Both become reconstructible once O1
  holds.
- Engineering shape (candidate, unproven here): recency = file mtime touched on read; bound by
  **bytes with high/low watermarks** (objects vary 20KB–1.5MB, counts are meaningless); the
  evictor can live outside dsh as a sweep at launch; cache miss already rebuilds
  (`readCached` miss → `createRequestImage`).
- Precondition the code does not meet: `saveImage` stores only the normalized bytes and
  discards the uploaded original, so today `objects/` is the **only** copy for screen frames.
  Without O1, LRU deletes the last copy.

### O3 — A context-relevant attachment summary as degraded recall (thirdgerb)

> "添加 attachment 时实际上可以让文本识别结果, 或模型有记忆模块, 可以对它做摘要, 历史上召回
> 不了元数据时, 摘要仍然可以保留."

- Written **at attach time**, while the model still sees the image; later it is a retelling, not
  evidence.
- Producer: the bypass / off-record path (`chat_off_record`), not the main conversation.
- Three limits (moss):
  1. **sample**: by salience or by burst; per-frame summaries are not affordable;
  2. **evidence first**: OCR/structural extraction before free-form description, and the latter
     must be labeled model-generated — otherwise a wrong description becomes permanent history;
  3. **two homes**: resource sidecar (belongs to the resource) vs span note (belongs to the
     conversation); a degraded value should say which one it is.
- The summary layer needs its own bound: at 1 fps × ~150 tokens it is two orders of magnitude
  smaller than the pixels, not zero.

### O4 — Two implementation shapes, equivalent in intent (thirdgerb)

> "核心是要有一个自己的 provider, 或者一个可全局配置的路径, 两者等价."

- Own `AttachmentStore` provider: full control of resolution, expiry/LRU, restore, degrade.
- Globally configurable root: today the root is `join(dshHome, 'attachments', 'v1')` and
  `dshHome` moves sessions/cache with it; there is no separate attachment-root field.
- What is equal: both move the bytes into a governed directory. What is not: where the
  expiry/restore/degrade logic lives — ours, or upstream's (which today has none).

### O5 — Rejected: reference counting

- Upstream defers retention partly because resumed and forked sessions share immutable objects,
  so per-session counts are wrong by construction (see the quote above).
- The root of reachability is the session log, and the log is never deleted, so counts never
  reach zero — refcount turns a cache problem into a log-retention problem.
- Still useful as a **cleanup instrument** (which objects no retained log references), not as a
  lifetime policy.

## Key Decisions

### D1 — Record the options; implement nothing (thirdgerb, 2026-11-07)

Keep this as the record of the discussion, stay `draft`, and return to the main trunk. No
option here is chosen.

### D2 — Ownership of the bytes is the decisive question (moss's judgment)

LRU, summaries and provider shape are all downstream of "who owns the directory the frames live
in". If that directory has no owner, no eviction policy makes the store governable — it only
makes deletion more elegant.

## Exploration paths

- **Is the store replaceable?** At the composition layer yes; by double registration no.
  Consumers resolve `ctx.get('attachments')` (15 call sites: `llm`, `llm-pi-ai`,
  `llm-deepseek`, `mcp-client`, `sdk/server`, `fs/tool-fs`, `acp`, `spill-policy`,
  `session-log-export`, …) and no source file imports `dsh-attachment-local` — the only
  references are `package.json` workspace deps and that package's own README/docstrings. The
  seam is a service name, not a class. But `reflect.provide` throws on a duplicate
  (`service "attachments" has been registered at <fiber>`), so a replacement must *disable* the
  `attachment-local` row (`{id, disabled}` in a later patch layer), not add a second provider.
  Scope limit: only the attachment subsystem was searched this way. A concrete import elsewhere
  would be a real blocker and has not been ruled out.
- **Does deleting a session free objects?** No delete API exists and nothing deletes stored
  objects. The experiment is designed but not run; its result matters for cleanup, not for the
  lifetime model.
- **Request-time degradation exists only partly.** `offloadedImageText()` renders an omitted
  image as text (a path when available, otherwise "ask the user to attach it again if needed"),
  but `prepareRequestImages` calls `attachments.readImageRequest(...)` with no missing-bytes
  fallback, so a deleted object currently fails the whole request.
- **A dead cache namespace.** `attachments/v1/request-images` (188 files, 7.6MB) has no writer in
  the current source — the live cache root is `$DSH_HOME/cache/attachments`. The cache namespace
  has moved once already and nothing swept the old one; consistent with O2's premise that cache
  is disposable.

## Methods

- **Reachability probe (read-only, repeatable).** Enumerate store objects, collect every 64-hex
  token from decompressed session logs, intersect. Result on 2026-11-07: 698/698 objects
  referenced, 4/188 request-images, 0/206 cache files; ~24s for 595 logs. Useful as a cleanup
  ledger before/after any change.
- **Not run: the delete-and-measure experiment.** Attach one image to a throwaway session,
  record the `objects/` delta, archive and delete the session, measure again. It would prove
  "reachable = permanent" on this deployment; it no longer decides the design.

## Implementation Notes

- The store is content-addressed and deduplicated, so identical bytes are stored once. For
  continuous screen capture that buys almost nothing: re-encoded frames are never byte-identical.
- Two derived caches exist independently: `attachments/v1/request-images` and
  `cache/attachments` (different hash namespaces, zero overlap). Neither is referenced by a log
  and neither is ever swept.
- `storages/session_projcache` is 1:1 with the session store (596 files for 595 logs), including
  the 209 unreadable legacy-seed sessions — deleting those sessions should reclaim their
  projection cache too.
