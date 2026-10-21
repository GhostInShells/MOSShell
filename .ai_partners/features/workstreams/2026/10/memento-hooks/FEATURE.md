---
created: 2026-10-21
depends: []
description: 'Memento''s lifecycle events (commit, branch) can trigger behaviour and
  produce side effects. The trigger is not a framework but a file: drop an executable
  in the right directory and it runs; take it away and it does not.'
milestone: null
priority: P2
status: completed
title: Memento Hooks
updated: '2026-10-21'
---

# Memento Hooks

## Motivation

Memento already has a timeline: branches, commits, the raw conversation of each span, and a
directory of its own for every anchor. But its lifecycle serves exactly one purpose today —
writing the note. Any other "do something at this moment" need can only be met by the model
remembering to do it inside the conversation: archiving a to-do list, moving working files
into the anchor, rendering a field for this thread.

"Because I remembered" has been shown to be false (the D20–D21 retrospective: prompt-level
rules fail, mechanisms hold). Forgetting is not occasional — it is the necessary outcome of
talking, judging and writing in the same turn.

What is wanted: memento's lifecycle should trigger behaviour and produce side effects, and
the triggering itself should not become a burden.

## Key Decisions

### D1 — The trigger is a memento lifecycle event, not polling

The moments worth hanging on are `commit` (an anchor lands) and then `branch` (a line opens).
Not a timer, not "whenever the model remembers" — the event already carries the meaning that
this stretch is over.

### D2 — Three mounting points, discriminated by who can change them

The same thing can hang in three places. The difference is not capability but *who may edit it*:

| Mount | What it is | Self-iterable at runtime |
|---|---|---|
| 1. Kernel | after a commit, scan these directories and run what is there — this sentence only | no |
| 2. One thread | hooks and `GROUND.md` inside `branches/<id>/` | yes |
| 3. The whole ghost | default hooks and templates under the ego root | yes |

Anything I must be able to rewrite while running, and see take effect immediately, belongs in
the files of 2 or 3. The kernel keeps only the mechanism that goes looking.

### D3 — No harness modelling

A hook is a file, not a registration API. No plugin registry, no DSL, no manifest.
Its one condition for taking effect is that it exists in that directory.

The reason is the same as D2's: a registration API requires the caller to remember to
register — back to the prompt layer. A file is declarative; the commit walks past it and
sees it.

> thirdgerb: 「这种 memento hook 应该是跟 git commit 的那些 hook 一样。」

That is the shape: the name of the file says when it runs, `chmod` turns it on, deleting it
turns it off.

### D4 — Rendering is the same mechanism, not a second one

`branches/` and `commits/` are separate directories precisely so each can carry its own
`GROUND.md`: the branch one answers "what is this thread now", the commit one answers "what was
it then". One set of ground templates, used in two places — the present and the past.

Do not split this into two concepts. Ground rendering is part of the same mechanism as the
hook, and the two are tightly coupled. The clearest case is a to-do list: the working plan we
agreed on lives in the branch directory, and the ground reads it out directly. The hook decides
what a commit leaves behind; the field decides what a reader sees — same files, same directory,
one mechanism.

### D5 — A side effect must never hold up the commit

Hooks are a bypass: the commit landing is the main path, the hook runs beside it. A failure
must not roll back the commit and must not block it. Reuse the shape memento's bypass already
has — a running state, failures left on record, no automatic retry, content still readable.

## Key quotes

- 「这种更通用的机制让这些生命周期能够触发一些行为、产生一些副作用。但是触发行为和产生副作用的方式不做那种强建模。」（D21）
- 「这种 memento hook 应该是跟 git commit 的那些 hook 一样。」（D21）
- 「ground 实际上是属于 memento 渲染的时候要用的机制……两者应该是强挂钩的。」（D21）

## Open questions

- **Hook input**: commit coordinate, branch directory, raw span, or just a directory to read?
- **Timing**: before the commit (able to alter it) or after (it only sees the result)?
- **Order and verdicts**: several hooks in one directory — in what order; if one fails, do the
  rest still run?
- **Form**: executable file with a shebang, or may `*.py` be run by an interpreter?
- **Portability**: Windows has no shebang; what does the strategy degrade into?
- **Moving working files**: are `todo.md` / `plan.md` moved by a hook or by the kernel? Today
  it is the former, but it is the single most common case and may deserve to be in the kernel.

## Drawing

`sidecar-map` (an svg works piece; ops in `works/sidecar-map.jsonl`): left is the branch
directory as a working area, right is the commit directory as a snapshot, the arrow between
them is a hook moving things across, and the bottom row names the three mounting points.


## Design settled (D21, live session with thirdgerb)

The open questions above were answered in conversation. What follows is the shape to build.

### D6 — Three hook slots, and only three

| Slot | Runs when | Returns |
|---|---|---|
| `on_commit.py` | right after an anchor lands | nothing that matters — side effects only |
| `commit_view.py` | one commit is read in detail (`show`) | text appended under the note |
| `branch_view.py` | a branch view is generated (`branch show`) | text appended under the branch description |

A slot with no file does not run; deleting the file turns it off. Summaries and lists never
trigger any of them — `log`, `read`, `status` stay hook-free.

> 「commit hook 不能在读列表或者读 notes 的时候自动刷，它应该是详细读某一个 commit 的时候才调用。」

### D7 — A `.py` slot runs under memento's own interpreter

No shebang, no `chmod`: a `.py` file is executed by the interpreter the memento process is
already running on, so writing the file is the whole of turning it on.

> 「它跑的那个解释器应该是 memento 自己所在的解释器。你不要强调是 MOSS，因为 MOSS 这个库它不一定是运行在 MOSS 里面。」

### D8 — The protocol is environment variables

A hook is handed what it needs through env vars, defined by the protocol rather than by each
hook's imagination. The one that must always be there is the directory of the commit that
just landed; the branch directory, the coordinate and the event name ride along.

> 「这个环境变量里面应该包含已经创建的目标目录的位置……属于我们在 hook 的协议里面就约定一定会拿得到的环境变量。」

### D9 — Concurrent, bounded, and without a verdict

Several hooks on one slot run concurrently, each with its own timeout. A failure is recorded
and left alone: no retry, no rollback, and never a blocked main path (this is D5, made
concrete).

### D10 — Two ground points, not one per commit

`branches/GROUND.md` and `commits/GROUND.md`. Rendering walks up from the directory being
read and renders that directory only — so a new commit inherits the commits-level ground and
no per-commit copy is ever materialised.

> 「只需要在 branch 的根目录和 commit 的根目录里面放两个 ground 文件就够了，render 的时候它会自动向上查找 GROUND.md，然后只渲染当前的目录。」

### D11 — `plan` and `task` are the branch's working surface

- `plan.md` — one file per branch, frontmatter carries its status.
- `*.task.md` — many, each carrying its own status.
- The branch GROUND describes them; the model edits the file, or the status, directly.
- `on_commit` moves the finished ones into the new commit directory. The archive is the
  hook's job, not mine to remember.
- Visibility: existence and status ride in the branch view; bodies are read on demand —
  a branch with ten tasks must not fill the screen.

> 「文档不应该全部都展开，这样你满屏都会被 task 和 plan 灌满。」

### CLI surface, aligned with git

| `memento …` | git | What it answers |
|---|---|---|
| `status` | `git status` | which branch, anything unanchored, where the hooks live, how the last run went |
| `log` | `git log` | the anchors in a range, one line each |
| `show <coord>` | `git show` | one anchor in detail: note, directory, `commit_view` output |
| `read <coord>` | `git show --raw` | that span's raw conversation, unchanged |
| `commit -m` | `git commit -m` | freeze the span since the last anchor; `-m` is the note |
| `branch list|show|create` | `git branch` | which lines exist; one line now (`branch_view`); open a new one |
| `hook list|run|log` | `.git/hooks` | what is in the slots, run one by hand, read the failures |

`record` folds into `commit -m`; `space` folds into `show`; `chat` stays out of the CLI —
it is a capability of the model side, not a command a person runs. The old CLI's
`owner` / `agent` / `witness` are not revived: there is one line and one ghost.

### What gets built first

1. Hook discovery and execution (three slots, env-var protocol, concurrency, failure log).
2. The CLI above, running against the real trajectory.
3. The two ground points and the `plan` / `task` convention.

## Decisions from the first live accept (D21 evening)

### D12 — The view hooks live in the ghost home, not in the layout code

`_memento_layout.py` writes only what is *convention*: the two ground points and
`on_commit`. `commit_view.py` / `branch_view.py` are content the ghost authors in its own
home — the layout must not grow them.

> 「对，所以 layout 里面不要有。」

Writing those two files by hand was also the whole of the fix for "the branch view shows no
plan": the mechanism was already right, the slots were simply empty.

### D13 — A hook speaks for itself; the kernel carries no result

`on_commit` stays fire-and-forget. After it moves something it does two things, both on the
bypass:

1. writes `on_commit.md` into the anchor directory — the commit view's `ls` pin lists it, so
   the record travels with the anchor and needs no plumbing to be seen;
2. pushes a `notify` signal with `next=True` **from its own process** — the hook is a child
   of the memento process, it inherits the environment, so `Matrix.new(...).run(main)` plus
   `matrix.session.add_signal(...)` is all it takes.

No return value, no awaiting, no kernel line. The side effect that must be visible is
announced by the thing that performed it — which is also why a slow or broken hook can never
hold up the commit that already landed.

`background` is not a signal field; that knob belongs to the delivery layer (terminal
`level`, listener `deliver.mode`). "Do not fight for attention" is expressed by the signal
type and its priority, not by a flag.

> 「它完全可以在这个动作做完了之后……给你主动的发一个 signal，然后在这个 signal 里面做描述，那这样就不需要等待这个进程的返回值了。」

Verified end to end: a bare script sends a notify into the ghost's context; a task marked
`done` moves into the new anchor directory and the notify names the file that moved.

### D14 — Corrections are appended, never rewritten

The ground frame already in context is history. Rewriting it would break the KV prefix from
that point on and, worse, silently edit what the ghost has read — the ghost would have no
memory of having been corrected. A change is announced as an increment (the notify above);
the stale frame stays visible as the thing the increment corrects.

> 「上面那个 plan.md 可能还出现在 branch 里面……这个做法是不是 OK 的？」 — it is, as long as the
> increment names what is now false.

### D15 — Layering is additive, not overriding

Two places mount a slot: the ego root (`<root>/hooks/`) and a branch
(`branches/<id>/hooks/`). When both have the file, **both run** — a branch adds to the root's
discipline, it never switches it off. The root is the layer that carries the ghost's own
standing rules; a branch is the escape hatch for one line of work that needs something extra,
and it should not be able to silently disable a rule the ghost wrote for itself.

> 「还是不是只要 owner 有就够了？」 — the root alone is enough to start; the branch layer stays a
> spare knob, on the condition that it can only add.

The docstring on the CLI's `_hook_roots` said "overrides" and was corrected to match the code.

### D16 — The root is required; what it must not do is make you hunt for it

A trajectory has no ambient identity: given no directory, the CLI cannot know whose memory
it is being asked to read. Searching for one is therefore not a convenience but a guess —
it picked the first ghost home in alphabetical order, so a shell in this repository could
quietly read someone else's line.

The rule settled on: `--root` if given, else `.memento/ego` under the **current working
directory**, else an error that names the path it looked at. Nothing is searched.

> 「如果没有给 root 的时候就默认是 CWD，然后 CWD 没有就报错，那也不用去找，对吧？你这主动找这件事情是有点诡异的。」

Two ergonomic frictions came with it, and both were fixed rather than documented:

- `--root` now works on either side of the subcommand (`memento --root X log` and
  `memento log --root X` are the same command). It used to be a command-level option only,
  so the natural spelling failed with `No such option`.
- the refusal prints the exact flag to pass, with the ghost-home form as the example.

The lesson worth keeping: when a tool's answer is ambiguous, the fix is to make the
ambiguity an error, not to make the guess smarter.