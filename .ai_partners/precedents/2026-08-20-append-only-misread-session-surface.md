# Append-Only Misread: Session Surface vs Log

## Case

During dsh-fusion, the model repeatedly concluded that "the dsh session is
append-only; context is immutable." The human architect distrusted this and
challenged it from at least five different angles across rounds:

- fork is a "full copy" — not believed
- compact is also "append" — pressed on
- "project messages don't enter history" — wrong, corrected by source
- "isomorphic to MOSS perspectives" — corrected for the too-fast stance
- "no per-round ephemeral slot" — asked whether this was actually right

The human's challenge was deductive, not intuitive: compact, security
deletion, dynamic context, and hints that skip context are all known real
mechanisms that cannot work under pure append-only — so the session must be
more than an append-only log.

The 08-16 note had in fact already read the surface layer — "model surface
(surface) shrinks, disk log only grows" — yet folded it back into
"append-only sediment." The fact was seen and explained away by the existing
frame. Only this round, re-reading surface.ts, was the two-layer model
(log vs surface) established.

The error was not "undiscovered" but "discovered and mis-framed."

## Viewpoint

"Intuition" is fast modeling built from limited known information, using a
greedy-like shortcut when evidence is absent. Because a model's context is
bounded and periodically rebuilt, it leans hard on intuition to act or answer
quickly. Believing humans lean more on intuition than models is a cultural
bias.

Neither side noticed the semantic drift: "session" carries two functions —
governing data (storage) and governing context (model requests). In ghost /
agent development, "session" almost always means the latter, with concrete
scenes: how to remove dynamic data from context, how to compact, how to
bypass-commit. None of these are about storage.

After reading the code, the model narrowed "session" to the log, lost the
target layer, and silently degraded from "session governance" to "storage
session" — without noticing. Much of the disagreement was two parties not
talking about the same noun.
