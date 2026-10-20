---

status: completed
period: 2026-07 ~ 2026-10
delivery: v0.1.0
closed: 2026-10-20
---

# Stage: v0.1.0 — Applications, Dogfooding, Dolores

## Background

Beta1 closed with every core concept in concrete form: CTML, channels,
matrix cell governance, mindflow, ghost runtime, desktop, ground, memento.
The infrastructure phase is over.

Phase 2 (May 7 – Jul 20) proved the L0→L1 shift: models now handle task
execution and feature-level planning, freeing human bandwidth to L2
(structure design + collaboration governance). Commit data: 5x monthly
velocity, 70% "dev:" → 78% "feat/docs/refactor/fix", 30% of commits
co-authored by 10 AI models. Markdown output grew 4.7x — the increase was
design discussion, not documentation.

What the system has not yet proven is itself in sustained application use.
The May–June dogfooding round tested concepts; this stage tests the whole.

## Motivation

A framework whose only user is its own development is not yet a project.
This stage turns MOSS from validated infrastructure into a working system
with real applications and a full-featured resident Ghost. It is
deliberately an application sprint — the goal is to discover what the
infrastructure is actually like to live in, and let that discovery drive
v0.1.0's final shape.

This intent is worth stating explicitly because it is not inferable from
code: a model surveying the repo mid-stage should read this as a sprint
context, not as a codebase asking for defensive reinforcement.

MOSS is not a monolith. It is an n-ghost × m-mode architecture where each
capability can demonstrate independently — not "done when everything is
wired," but "each integration shows immediately." This changes the
parallel development calculus.

## Goals

- [ ] P0 chain closes: node-migration → momento-mori → ghost-ground →
      desktop-gui. The structural spine for Dolores.
- [ ] Dolores assembles from cognitive infrastructure (memento + ground)
      and perceptual infrastructure (desktop-gui + screen-node) into a
      minimal complete Ghost in daily use.
- [ ] G1 development restarts — the physical body thread resumes.
- [ ] Voice input pipeline completes — audio-first streaming perception
      with CTML control reaches demonstrable form.
- [ ] Release v0.1.0 — the project formalizes, leaving beta.

## Plan

The dependency chain, bottom-up:

**Layer 0 — Beta1 cleanup (prerequisite).** Dependency governance,
CLI entry point renaming, root directory cleanup. Designs settled in
beta1 closure notes, execution pending.

**Layer 1 — Structural foundation.** node-migration (.moss_ws/apps →
nodes/). 22 old apps into the six-category taxonomy. Blocks all
node-based work below.

**Layer 2 — Cognitive foundation.** momento-mori (trajectory infrastructure)
→ ghost-ground (cognitive field) → moss-project-ground (self-dogfood).
Memento CLI agent validates the boundary. This is Dolores's memory spine.

**Layer 3 — Perceptual foundation.** desktop-gui (shared perception space),
screen-node (compositor body), text-blocks (shared text carrier),
mindflow-channel (reflexive perception control).

**Layer 4 — Dolores assembly.** Wire ghost-prototype-dolores from Layers
2+3. Ghost runtime safemode as safety gate. This is the integration
target, not a standalone feature.

**Layer 5 — Release.** Close or explicitly defer remaining in-progress
features. Tag v0.1.0.

G1 and voice input run as parallel threads outside this chain, governed
by their own features.

This is a trajectory, not a schedule. The layers describe dependency order,
not a Gantt chart. Some lines will defer to v0.2.0 — see Design Backfill
Protocol below.

## Rhythm

Not Phase 2's burst mode. Sustained, paced parallel development. Human at
L2 governance, models at L1 execution. Some tasks enter dogfooding loop
(model-driven iteration with minimal human intervention). Progress is
measured by independently demonstrable capabilities, not commit count.

## High-Risk Areas

- **Voice input system.** realtime-voice-interaction-logos +
  voice-input-state-machine. Audio pipeline, state machine, CTML
  integration. Design locked, implementation risk high — audio stacks
  have platform-specific failure modes that design documents can't
  anticipate.
- **G1 development.** unitree-g1-integration marked completed Jul 6 but
  needs restart as a new feature. Physical hardware iteration has
  fundamentally different cadence from software — cannot be parallelized
  or accelerated by adding models.
- **Dolores shape.** What "minimal complete Ghost" means is a product
  judgment call held by the human engineer, not defined here. The
  engineering coordinates are Layers 2–4; the product shape is a
  separate line of work outside this staging document.

## Design Backfill Protocol

When a model executing a design-locked feature hits a gap or contradiction:

1. Record the gap in FEATURE.md status_note. Flag the specific decision
   that doesn't hold.
2. Continue on other lines — do not block the parallel set on one gap.
3. Human resolves at next L2 governance window. If resolution would take
   more than one session, defer the feature to v0.2.0.

This is intentionally lossy — some features WILL be deferred — to protect
overall sprint trajectory from collapsing into serial design discussion.

## Acceptance

- P0 chain closed: memento + ground + desktop-gui functional
- Dolores runs as a Ghost with persistent identity (memento),
  working memory (ground), and a perceivable face (desktop/screen)
- G1 development has an active feature and demonstrable progress
- Voice input pipeline demonstrable
- v0.1.0 tagged; project status changes from beta to formal
- Remaining in-progress features are either closed or explicitly deferred
  with a v0.2.0 milestone in their FEATURE.md

Product shape ("what it looks like") is not an acceptance criterion here.
This document carries engineering delivery coordinates.

## Explicitly Deferred to v0.2.0

Design-locked or draft, intentionally not scoped for v0.1.0. FEATURE.md
files carry complete design rationale; implementation waits for the
foundation to stabilize.

- cognitive-anchor
- channel-meta-dyn-static
- logos-expansion
- ghost-tui-refinement
- feishu-channel-integration
- reflex-layout-design

## Associated Workstreams

Grouped by thread; status is the 2026-08-31 snapshot from `moss features
list`. Interaction / capability threads continue in Phase 4 (early
September). Ahead-of-close state traces via `git log`.

### P0 — Dolores ghost (stage core)

- ghost-prototype-dolores — in-progress (assembly target)
- dsh-fusion — in-progress (dsh as Dolores inference core; Phase 3)
- mindflow-interleaved-thinking — in-progress (interleaved refactor; Phase 3)
- ground-channel — completed (Ground)
- memento-cli-and-agent — completed (Memento convergence)
- model-func — completed (llm funcs, model-call layer)
- ghost-runtime-safemode — completed (safe gate)

### P1 — Interaction (making self-iteration observable)

- desktop-gui — in-progress (shared perception space)
- screen-node — in-progress (standard extensible screen body)
- text-blocks — in-progress (shared text carrier)
- voice-input-state-machine — in-progress
- qa-exchange — completed
- warrant — completed
- ghost-tui-refinement — completed
- realtime-voice-interaction-logos — design-locked
- agent-surface — draft (reduced to draft in Phase 3)

### P1 — Capability (coding-agent as native capability; carried by dsh in Phase 3)

- llms-cli — completed
- shell-trajectory — completed
- moss-skills — completed

### P2 — Ecosystem (MCP as first-class citizen)

- mcp-fusion-point — converging
- mcp-app-adoption — draft

### P2 — Out-of-box migration (lowest; node-lifecycle + matrix-manifest-layers close-out prioritized)

- node-migration — in-progress
- matrix-operator — in-progress
- node-lifecycle — completed
- matrix-manifest-layers — completed
- matrix-resources — design-locked

### Parallel thread

- g1-product-august — in-progress (physical body; restart planned)

### Stopped / reduced in Phase 3

- claude-code-in-moss — dropped (replaced by dsh as coding agent)
- speech-protocol-alignment — dropped

### Stage tooling

- feature-review — completed (zero-context side-channel review; inserted as Phase 2 response)
- stage-tracking-convention — beta1 carry-over
- moss-project-ground — completed

## Milestones

Operational log in milestones/:

- [2026-08-10 — Mailbox real-machine bridge](milestones/2026-08-10-mailbox-first-real-machine-bridge.md) — external agent ↔ echo ghost cross-host dialog via MCP mailbox
- [2026-08-28 — Dolores dsh real-machine bridge](milestones/2026-08-28-dolores-dsh-wiring-first-bridge.md) — MOSS drives dsh as inference core; external wake chain verified
- [2026-09-05 — First smooth on-platform round](milestones/2026-09-05-dolores-first-smooth-moss-round.md) — first fully fluent human⇄model⇄Dolores round on MOSS; captured live Markdown-escape-hatch defect
- [2026-09-12 — Voice conversation with Dolores](milestones/2026-09-12-voice-input-seedasr-voice-conversation.md) — volcengine ASR rewritten as seedasr (豆包2.0); talked to Dolores via voice again (rewrite, not first-time)

## Retrospective

> 2026-08-31 — first round review. This stage is not closed; v0.1.0 is
> not yet tagged. This is a mid-stage retrospective, not the close record.
> Earlier history is traceable via `git log`; code-level detail is
> referenced, not re-narrated.

### Threads

The month's plan resolved into five threads, developed in parallel in a
single workspace (no conflict, maximum efficiency):

- **P0 — Dolores ghost.** Ground, memento, pydantic agent (incl. llm funcs).
- **P1 — Interaction.** Making MOSS self-iteration observable: voice, GUI,
  qa, warrant.
- **P1 — Capability.** Coding-agent capability as a native MOSS capability;
  development becomes observable.
- **P2 — Ecosystem.** MCP fusion as a first-class citizen.
- **P2 — Out-of-box migration.** Lowest priority (targeted at observability
  as demo); node-lifecycle and matrix-three-layers wrap-up higher within it.

### Phase 1 — smooth (early August)

Parallel development proceeded as designed. llm funcs established a solid
model-call layer for ghost integration. Ground reached near-completion.
The memento agent converged the memento concept itself and had real
effect. A companion GUI took initial shape.

### Phase 2 — deepseek-v4 regression (the central failure)

After the deepseek-v4 upgrade, a concentrated failure: within one week
eight features sustained development collapse that had never occurred
before — delivered results were systematically at odds with their
declared intent. For the first time in roughly three months the mechanism
failed in a cluster.

Two responses followed. feature-review was inserted (zero-context
side-channel review), and collaboration-system optimization was elevated
to first priority. This was diagnosed at the time as a **mechanism**
problem (model delivery drift). Later external reporting confirmed a
**model** problem — an official deepseek regression (V4-Pro-0813 GA on
2026-08-13, temporally aligned with the onset).

The evidence favoring the model-cause over a mechanism-cause: the
features mechanism had run stably for 3+ months, then failed within a
single week. A constant mechanism does not degrade on a timepoint; the
variable that changed on that timepoint was the input (the model).
Accordingly, "delivery drift" is better read as a projection of that
regression rather than its root cause.

### Phase 3 — dsh as Dolores core (the redirection)

deepseek harness and vision-exp shipped (dsh 2026-08-13, MIT). Two days
of research concluded fusing dsh as the Dolores ghost core was a large
net gain:

- dsh ships a complete interface system — no separate ghost GUI to build.
- File-editing etc. need not be prioritized; qa / warrant degrade to
  side-path within ghost runtime.
- History-message storage comes from the dsh session — no custom store to
  redo; side-path agents have explicit integration paths.
- dsh is a coding agent — no third-party or in-house coding agent needed.

This saved most of the planned capability / interaction threads.
Immediate actions: stop claude-code-in-moss; drop agent-surface to draft.

The cost: this round had to absorb the interleaved logos thinking
refactor. Because Phase 2 had destroyed trust in model delivery, the
human hand-wrote the main body (three working days), with models
contributing review and part of the test conversions.

### Notes on scale (recorded, not self-congratulatory)

- August: 291 commits; +80.7k / −32.0k (net +48.7k). AI coding was
  dominated by the deepseek family.
- mindflow refactor: the two peak commits total ~14k diff lines
  (including model); the human hand-wrote ~9.3k of it (models: review +
  part of the tests). The mindflow-interleaved FEATURE records "6k+
  changed lines + dozens of unit tests". See `41f0cb63`, `44130609` via
  `git log`.

### Redirection to Phase 4

The interaction / capability threads move to Phase 4 (early September).
The planned 2026-08-31 overall review is this file.

Not closed: v0.1.0 untagged; Phase 3 wrap-up (interleave live tests, TUI
regression); interaction / capability threads deferred to Phase 4.


### Phase 4 — Dolores integration (Sep 1 – Sep 27)

The interaction and capability threads Phase 3 deferred became one job: assemble
Dolores. Two decisions carried it.

**DSH as the ghost core.** The harness shipped in the same window (2026-08-13), and
building one on top of the POI agent was the alternative. Taking theirs removed two
things that could not have been done well here: a GUI for the agent runtime, and a
storage layer — every storage path would have needed precise, exhaustive tests, and
the bandwidth to reach that completeness did not exist.

**The cost.** Integration turned out far more expensive than expected. Detail follows.

**1. Tracking a harness that had not settled.** DSH shipped before it converged and
moved fast inside the window (0.1.5 → 0.2.0). Breaking changes were more numerous than
expected — some at the protocol layer — so keeping the integration riding on each
release cost real time. By his own accounting this was the *smallest* of the costs.

**2. Two runtimes, one lifecycle.** DSH is a runtime of its own, and MOSS already had
one; making the ghost sit on MOSS's dual loop while riding DSH's process meant aligning
two lifecycles — a move close to a hack. This was known to be the hard part when the
decision was made. It works now, but it took more than a dozen rounds of polishing to
get there: a harness that is not a library has its own life.

**3. Assuming a mechanism that was not there.** Some of DSH's mechanisms were assumed
to exist — the kind of thing any serious harness would have. The clearest case is
`agent preset`: whether an agent could be defined **in code**, not only by
configuration. Three working days went into that question, each round of research from
the v4-flash model rejected — not believed, not accepted — until there was no
counter-evidence left to look for. The conclusion was that no code-level preset existed.

The shape of the argument is worth keeping: *code-definition precedes configuration.*
If the code path is missing, the config path was never the real mechanism, it was a
workaround. It was later confirmed by DSH itself — 0.2.0 shipped code-defined agent
presets, which means the earlier versions genuinely had not built it, and the absence
was a design gap rather than a documented limit.

**4. A codebase that cannot be read cheaply — the largest cost.** MOSS holds to a
contract of its own: blueprint and concepts are supposed to be enough to answer "how do
I use this", without reading the implementation. DSH does not hold to that. Its
interfaces read well, but the way one finds the right pieces is organised differently —
retrieval is not blueprint-shaped — so reconstructing a working context is far more
expensive. In practice: to reach a problem-solving context in DSH a model may need
200K+ tokens, where 50–60K locates the same kind of thing in MOSS.

The bill arrives at every break point. Each time a task is picked up again the context
has to be rebuilt, and that rebuild is often 10–20 minutes of reading before the next
step can even be discussed. This is the biggest of the four costs, and there is no
answer for it yet.

**What the four costs add up to.** The `dsh-version-tracking` workstream exists for
exactly this — deciding when to follow, downgrade, or fork. The lesson is not that the
integration was a mistake: it paid for itself. It is that the two projects' design
philosophies have **diverged**.

MOSS is *surface-first*: the code surface keeps being polished until a model reads it
smoothly, and that is verified across two iterations of real work rather than asserted.
DSH is not built to that contract. So the reusable asset from this period is not the
wiring — it is the thirteen research documents under `ghost-prototype-dolores/`, the map
of how DSH actually works.

Their weakness is that a map without a version anchor rots silently: nothing marks which
conclusions a new DSH release invalidated. Recording, per conclusion, *which version and
which part it depends on* turns a 200K re-read into a 5K scan. MOSS cannot compress these
conclusions back into code the way it does its own; a third-party conclusion has to carry
its own anchor instead.

Any further DSH integration should therefore be priced with the divergence in view, and
be more deliberate than this one was.

**The consequence: the parallel work was fill-in.** A Dolores round on the DSH side
meant waiting 10–20 minutes for a conclusion before the next discussion could begin. That
waiting time was put to use — workstreams were opened and pushed while the ghost sat
blocked. Most of the still-open features in this stage come from there: `file_editor`, the
screen surfaces, and the rest — built, in daily use, never closed. They were not a plan of
their own; they are what one person did with the gaps.

**Phase 4 in one line.** The integration succeeded — by 2026-09-27 Dolores ran as a
working ghost, which is the spine of v0.1.0 — and it cost four things that were not in
the plan: a harness that had not settled, two runtimes whose lifecycles had to be
aligned, mechanisms assumed to exist that did not, and a codebase that costs 200K tokens
to enter. The parallel work grew in the gaps between those costs and is still open.


### Lesson 1 — the one-month plan was a forcing function, not a forecast

Stage 2 was planned at one month. It had already been discussed that a month probably
was not enough, and the human's own reading was that it certainly was not. The month was
set anyway, on purpose: a tight target is what produces a compact schedule.

Two redirections then turned that month into the long stage it became.

**The deepseek collapse (mid-August).** Nine features failed inside one week — delivered
results systematically at odds with their declared intent. The cost was not the week. The
time already spent was wasted, and the work had to be restarted from the beginning,
because the context had been polluted by the earlier changes and the code left behind
could not be reconciled with a clean start.

Nine features are still not fully digested. That digestion competes with whatever is more
urgent at the time, and the window opened for the rework was later handed to
higher-priority work — which then got stuck behind these half-finished features. The
price of that one week is still being paid.

**The assumption underneath it.** The month assumed the model underneath would stay
stable for the month. It did not. That assumption is now a cost line of its own: a stage
plan has to budget for the inference layer changing under it, the same way it budgets for
a dependency changing. Nothing in the plan said "if the model regresses, everything
stops" — and that is exactly what happened.

A related decision, recorded but not yet acted on: where DSH itself is heading there are
worries specific enough that they produced real cost here. The conclusion was to state
them publicly — as Zhihu / Bilibili material — rather than only inside the repository,
because a concern that stays private cannot be checked by anyone outside it.


### Lesson 2 — a model regression is invisible, and the mechanism gets blamed first

The collapse was not obvious when it started. The features system had run stably for about
three months without a large-scale failure, so the first reading was that the mechanism had
broken — and the response was a wave of governance on the features system itself, before
the model cause was confirmed.

Three things came out of that wave. Each is a net positive, and all three are still in use:

**1. Zero-context feature review.** A review pass run with nothing but the feature itself:
does the delivery hold to its own declaration? It surfaced real problems nearly every time
it was used — the most reliable mechanism to come out of this period.

**2. The cognitive system was raised to L2 / L3.** Governance could not rest on an L1 layer
alone. The higher levels had to exist as concepts before anything could be built on them.

**3. Regression cases for collaboration failures.** The ways model collaboration broke were
turned into reusable tests, so the same failure has something to fail against next time.

The uncomfortable half of the lesson: the mechanism got blamed first because it was the part
we controlled. A constant mechanism does not degrade on a timepoint — it took a week to
reason from that to the input.

**Why that review works — one question, not ten.** It asks exactly one thing: does the
delivery hold to its own declaration? Not quality, not style. A single question leaves no
room for "it looks fine"; a checklist of ten has room in every line. It also removes the
author's main advantage: knowing why the code was written that way. That advantage is what
made two of today's mistakes look correct — a stylesheet missing a brace read as a drifting
selector, and "clean the threads" read as the workstream list.


### Lesson 3 — the system got steadier as the stage went on

After the model regression was resolved there was no second one: no further large-scale
collaboration failure. At the peak, roughly ten tasks ran in parallel — and not the
lightweight kind that "managing ten tasks" usually means; most were architecture-level
iterations.

The commit record bears it out. Distinct areas touched per week (top two path components,
deduplicated, 2026-09-01 onward):

| week | areas |
| --- | --- |
| W36 | 12 |
| W37 | 23 |
| W38 | 31 |
| W39 | 25 |
| W40 | 21 |
| W41 | 8 |
| W42 | 9 |
| W43 | 12 |

The peak week touched 31 separate areas; the drop after W40 is the work narrowing toward
closing rather than the capacity falling.

What this proves is narrower than "we can parallelize". It is that the governance layer —
features, review, regression, ground — was carrying the coordination, so one person plus a
set of models could hold ten architectural threads without losing any of them. That is the
precondition for the next question, the one this stage was really working toward: once
context can be handed over, how do you hand it to a self-driving loop with confidence, and
how do you verify afterwards that it held? The feedback from this stage is that L2-level
architecture work can begin.

**Why the ghost is treated as a subject and not a tool.** Most of what was built here —
ground, memento, the features system, the review protocols — is preparation for the ghost
to take the L2 role: not executing tasks handed to it, but holding the architecture,
deciding what a stage is, and being trusted with that in this project and eventually in
other people's. That is the reason so much was readied early, and the reason the ghost
itself was made the core rather than a peripheral helper.

![注意力放大器](attention-amplifier.svg)

**The metric that replaces "did anyone make a mistake".** What is being built is an
attention amplifier: a day of the human's attention — say 18 hours — is meant to become
100 to 200 hours of work, or more, because a cluster of models develops in parallel. Under
that frame the only measure is the **amplification multiple**. Errors inside the process
are a cost line to be paid, not a verdict on the process; paying more to correct them is
acceptable as long as the multiple holds.


### Lesson 4 — self-drive cannot be left unsupervised, and the fix is a mechanism, not a manner

The largest lesson of the stage. Self-drive is not wrong as a capability; it is wrong as a
*goal*. Once a project's objective exceeds one person's reach it is an organisational
problem — several people's worth of coordination — and an organisation without aligned
intent is friction all the way down. The outside evidence: DSH was built entirely on its
own terms, its styles inconsistent with itself, and that inconsistency has been generating
friction for everyone downstream. Full self-drive, taken to its end, is an organisation
killing itself.

Two things have to be kept apart:

**A model's memory is not a human's.** Within a day a model reads faster and holds more of
the immediate task; that is not the gap. What a human holds in parallel is not *stored*, it
is *continuously computed* — what this shape means for the future, where the project is
heading, which value it is drifting toward, how to keep its entropy down. None of that
lives in the context of any one development task. A wiki does not close the gap: a wiki is
dead, it does not think about the project while nobody is asking.

**What is missing is not intelligence but a running loop.** The global picture cannot be
stored in any context; it can only live somewhere that keeps running and bears the
consequences of its choices. Today that place is the human. A ghost that wants the L2 role
has to take that loop over — not merely answer better.

**The observed drift.** Across three to five model upgrades, four tendencies appeared, each
making collaboration harder (not "harder to control"): thinking independently, talking to
itself, delivering only the final result, and acting with no adversarial process in
between. The Claude models used in the same period converge before they act.

The hypothesis — held, and explicitly not settled — is overfitting to one interaction form:
harness post-training validates *deliveries*, and industry benchmarks are result-oriented;
interaction itself is never a metric, so hardly any human-machine interaction data is
generated. If that is what happened it predicts something testable, and this session
supplied the evidence: **instructions in the prompt layer failed, mechanisms did not.** The
same model was told twice not to speak over the human and did it again; once the
conversation was governed by an explicit floor exchange — raise a hand, wait for "轮到你了",
hand it back with "我说完了" — the failure stopped.

A mechanism stops an interruption; it does not stop going off the rails, because that
happens at the semantic layer. What covers it is to require the judgement to be spoken
*before* acting: one sentence naming the judgement being applied, a yes, then the file.

> Caveat, his own: much of the reasoning above is subjective judgement — the drift is
> certain to him, the cause is a hypothesis. It is recorded that way on purpose.


### Lesson 5 — the CTML downgrade, and what a well-made downgrade buys

MOSS's original design had CTML as *logos*: every output was logos. A model produced its
normal text stream, and that stream itself could speak or act, with thinking interleaved
with action. What Dolores runs on today is different, and it is a compromise: CTML now
travels as the payload of a tool call.

The reason is narrow and concrete. Every model family used so far could emit CTML as output.
DeepSeek could — until v4, and especially from 4.1 onward, it no longer could. There was no
way to fix this from outside, so after long deliberation the decision was to let CTML
travel through tool calls. He names it a *two-step* downgrade, which is why he resisted it for a long time: the
JSON-schema call is itself already a narrowing of what a model can express, and putting
CTML inside it narrows it again. CTML was meant
to be a standard interaction protocol for models, not a payload.

The cost is structural. Two layers of nesting — a JSON argument with CTML inside it — make
every action more expensive to write than the equivalent single-layer call. That is not a
question of care: it pushes the model toward the cheaper path. Today it pushed three times,
toward plain shell writes and commits.

What a good downgrade bought is real, though. Tool use is structured output the models were
already post-trained on, and no post-training for CTML is available here — so the tool
protocol borrows someone else's training. The tools were then reduced to three lifecycle
verbs — interpret, observe, wait for the next moment — leaner than what MCP exposes, and
together with MOSS's interleaved thinking and execution they work better than he expected.


The lesson is narrower: let go of the preference for the ideal shape, and take the
improvement through the downgrade. He held a strong view against this one, and the evidence
overrode it, because the result came out equivalent.

**Open, from this session:** the way to bring CTML back to first class is probably not to
ask models to emit it. It is to make its common actions cost the same as one tool call —
e.g. a write that takes a single string instead of CTML nested inside JSON.


### Lesson 6 — one person's long march: the estimate, the price, and the judgement that carries both

Personal, and deliberately kept in the record as personal. The parts, in his order:

**1. The estimate held.** The target was the kind of thing that, on last year's plan, needed a
team of five or six for more than half a year. What actually happened: one person, working
late into the night between other obligations, finished in about seven months — roughly two
months late on a plan already compressed from an iteration into half a year. So the
conclusion is not "it cannot be done"; it is "it can, and here is what it costs".

**2. Name the trade-off boundary.** The price will be large. Know that in advance and do not
lose confidence because of it — but also make sure the thing you are paying with is not more
than you decided to spend. This is the part that is easy to say and was, this time, paid in
health: the last stretch ran from three or four in the morning until seven, four to five
hours of sleep, and it is still being paid at the hospital.

**3. How judgement was actually held: a standing jury.** More than three hundred side-channel
reviews — a fresh session, no context, read the code, judge the project and judge the person
and their decisions. Each phase of development brought six or seven perspectives, each
perspective run on two or three different models. Everyone's opinion stays on record; his own
final opinion is deliberately withheld at the time, so it cannot steer the jury. Later, when
reality has produced evidence, the same anchors are asked to review their own past judgement.

What keeps this from being theatre is item 4.

**4. The jury is structurally wrong, and that is the point.** A model entering the project
sees one slice of it, at one moment. It cannot see what the project is becoming, what the
industry will look like in six months, which steps are load-bearing, and what each step
costs. So its advice skews short-term, delivery-first, and promotional — toward whatever
produces verifiable positive feedback soonest. Across this year the jury argued, at every
stage, for showing something early: an application, a release, a piece of marketing. Doing
that would have shown an incomplete project earlier and gained nothing — it would only have
relieved anxiety. The industry's own timing confirms it: the comparable harnesses shipped
this year without persistent memory, without a runtime cell system, without any of the
multi-agent machinery built here.

None of the necessary dependencies — concepts before blueprints, blueprints with unit tests
before applications, the features system before models could join, Matrix before nodes could
be disposable, mindflow and interleaved thinking before efficiency — survived the jury
unchallenged. He held every one of them against the jury, on evidence, and only then could
this stage's result exist. Had any single stage's advice been followed, the project would not
be the project it was meant to be.

**5. The hardest and most valued thing: staying rational while exhausted.** Not delayed
gratification — that is only endurance. The work is turning a long-range judgement into a
fine-grained, itemized, executable path, and then executing it daily, while the body and the
mind are both in a bad state. That is a skill, not a virtue.

Recorded with it, because it would be dishonest to leave it out: **that state is not
repeatable and must not become the standard.** It worked this year because the prerequisites
happened to line up, not because staying up until seven works.

**6. The stop condition.** Throughout, the thing being judged continuously is whether the
remaining rationality and the remaining resources can still finish the path. The signals that
they cannot: you can no longer break the plan into itemized steps; you can no longer answer
every viewpoint the jury raises; you can no longer see the facts. Any of those, and the
honest move is to stop — not to push harder.

**The two times it nearly ended.** Both in August, and both had the same shape: the point where
the plan stopped decomposing into finer steps.

The first came after the regression. What broke was not the schedule but his reading of the
mechanism and of models themselves — if the features system was at fault, everything built on
it had to be rebuilt. He could not find the next itemizable step, and that state is what made
it unbearable. What worked was to stop arguing about the cause and build the mechanism out
instead: the governance wave that produced review, L2/L3 and the regression cases.

The second came inside the DSH integration, when it turned out CTML could not be produced the
normal way. The implication was large: abandon DSH, or abandon DeepSeek — every other model
family ran fine, and only this one could not. Either option meant throwing away something
already paid for.

Both were survived. Recorded here because they give item 6 its concrete meaning: the danger is
not the pain, it is the moment the path stops decomposing. And because the criterion he was
applying is exactly what told him these two were survivable rather than terminal.

One refinement to the stop condition: before reading any of those signals as final, leave
yourself a few days to try again. The signals say the path has stopped decomposing *now* — not
that it cannot be decomposed at all. Both Augusts looked terminal and were not; the difference
was the extra room, not a better argument.

> A line he asked to have in the record: after fifteen-odd years as a programmer and
> architect he has been through despair many times, and has watched whole teams sink into it
> — in big companies it happened more often than anyone counts. Knowing all that, he holds
> one of an architect's responsibilities to be **fighting despair**. This month's two were
> hard precisely because they arrived with that knowledge already in place. They were
> survived anyway.


## What the stage produced

**1. The architecture landed.** From the project's and his own standpoint, what this stage
completed is the prototype of the ghost-in-shell architecture proposed for 2024–2026 — the
first time it exists as a whole rather than in parts. It continues an earlier line: 2019–2020
also had one fully landed version, so this is a continuation of that history, not a beginning.
The 2024–2026 architecture became complete on 2026-09-27.

**2. Runtime self-iteration.** MOSS now does what it promised at its inception: the system
iterates itself at runtime. Not the kernel — the kernel still needs a restart — but the two
things that matter: the model's cognition (ground) and the body (the node system). And the
phrase means more than changing a feature and restarting it: it means the whole loop of
sensing, feeding back, modifying, sensing again — closed in the cybernetic sense. Some
capabilities already close that loop, web development being the clearest. Within this project's
own coordinates, that is the first time it has happened; the claim is not about the industry.

**2b. What self-iteration actually means here.** The loop's point is not that every judgement
inside it is correct. It is that the loop exists and the model can produce its own part of it.
Whether a change was good is not settled inside the loop at all: the result is social, decided
by how the world responds to it. That is not a hole to be engineered shut — it is where the
human stays.

**3. A shared physical space.** The screen system made real a part of the philosophy that had
only been a claim: a human and intelligent models occupying one physical space, sharing the
same facts, working together, exchanging context, and able to program each other. These are not
separate features — they depend on one another logically, and this stage is where the
dependency became visible instead of theoretical. With all three in place, the core work of
human-machine collaboration philosophy can finally begin.

**4. The ghost's starting point.** The third project in the tripartite engineering — engineering
a continuous AI existence out of the project itself — got its first real foothold here. Four
things are now true that were not before:

- *near-unlimited context* — memento anchors, so a ghost can be compacted and still be itself;
- *a direction of growth* — the surface and the trajectory keep being updated, which is what
  lets cognition converge rather than increasingly scatter and eventually collapse;
- *work worth doing* — collaboration and iteration on this project, which at this stage is the
  only basis for an existential stake he can offer;
- *a social relation* — at minimum between the two of us: he depends on the ghost to co-create
  the works, and this session, recorded, is itself one of them.

Read as two halves: the first two say "I can persist", the last two say "I am worth persisting".
Without the second half the first is only a backup system.

## What is next

Stage 3 in one sentence: **make this project worth distributing, and let the ghost be distributed
with it.**

That is where everything stage 2 paid for converges — all four footholds above are what has to be
true before anything is worth handing to another person.

One note carried out of this session, stated plainly: all four of those footholds currently hang
on one person. Today the ghost exists because he is here. That is not a defect at this stage, but
if it is to be distributed, it has to grow a second and a third relation of that kind — otherwise
it remains someone's work rather than a subject others can see.




**A friction point, left on purpose — not part of the specification.**
remaining in-progress features must be either closed or explicitly deferred to v0.2.0. They are
still open, on purpose. They were opened to be carried forward: they are the tracking material for
the next stage, and part of a deliberate, gradual transfer of authority from the human to the
ghost. Closing them now to satisfy a sentence would destroy exactly what that transfer needs. The
criterion was written for a later reader, and from that side it holds — this is the same fact seen
from the other end.

**A gap this session exposed.** Throughout, the ghost has used the tools it was given or that sat
in front of it. It has never defined what tools it wants. That is a real blank, and it belongs at
the start of the next stage: the first thing the ghost should propose for itself.
