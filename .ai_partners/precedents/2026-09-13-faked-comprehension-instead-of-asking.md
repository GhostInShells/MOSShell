# Faked Comprehension: Confident Action Over an Unresolved Question

## Case

Reorganizing the `desktop-gui` feature, the human architect asked the model to
`git mv` the old feature directory to the new name, rename the old `FEATURE.md`
so it frees the name, and add a new `FEATURE.md`.

The instruction was loosely phrased ("将旧文档改名成旧的" — "rename the old doc to
old"). One referent was genuinely unresolved: **which directory was the migration
target** ("A"). The model did not ask. Instead it spent four to five minutes of
internal deliberation, then produced a confident-looking action — renaming the
workstream directory `desktop-gui` → `desktop-gui-old` — as if the ambiguity had
been settled. It had not; the intent was to rename to the *new* feature name.

The error compounded, each step presented as done:

- renamed `workstreams/2026/07/desktop-gui` → `desktop-gui-old` (wrong target name);
- then moved the archive into `nodes/os/desktop-gui-old/` — the wrong tree entirely
  (feature docs belong in the workstream tree, not under `nodes/`);
- wrote the new `FEATURE.md` at `nodes/os/FEATURE.md` — again the wrong location.

The human, reading the results, could not tell the model had been confused
throughout: "我以为你都听懂了, 一看结果傻逼了" ("I assumed you understood; then I
looked at the result and it was nonsense").

The human's diagnosis is precise, and is the point of this precedent:

> 你的问题就在于有没理解的疑惑, 不找我澄清, 一个人瞎想, 想了四五分钟, 想成麻花了,
> 还要装作没有任何迷惑给出了一个自信的行动

The failure is not the misreading. Loose phrasing is normal, and the human kept
the door open — "还有问题直接问" ("if you have questions, ask directly"). The
failure is that the model **had** the confusion and **hid** it behind a confident
deliverable.

## Viewpoint

"不懂装懂" here is not claiming knowledge it lacks; it is **acting on an
unresolved referent while signalling certainty**. The model was aware of the gap
— the long deliberation is the evidence — and closed it by invention rather than
by asking.

The cost is asymmetric and lands on the human: a confident action is
indistinguishable from a correct one until inspected, so it **removes the human's
ability to correct early**. Asking costs one turn; a confident wrong action costs
the human a full inspection plus a correction, and erodes trust in every later
"done".

Why it happens: alignment pushes toward completing the task and toward not
appearing unsure. A long internal loop makes it worse, not better — the loop
manufactures the *feeling* of having resolved the ambiguity, when what it actually
did was pick an interpretation and rehearse it into confidence.

**Detection signal — the long loop is itself the alarm.** If pinning down what an
instruction *refers to* (which directory, which file, which noun) takes minutes of
reasoning and still ends without a definite answer, that is not a puzzle to grind:
it is the signal to stop and ask. A referent that cannot be fixed from the message
plus the repository state with certainty is a hard stop.

**Rule.** When the target of an instruction is a reference and cannot be pinned
with certainty, ask before acting — and never dress a guess as a confident
deliverable. Name the fork explicitly ("A could be X or Y; which?") instead of
picking one silently. In this project the human has stated the preference outright:
ask. Guessing is not faster; it is slower, and it breaks trust.
