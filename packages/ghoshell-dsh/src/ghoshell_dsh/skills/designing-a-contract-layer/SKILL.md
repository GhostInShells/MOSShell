---
name: designing-a-contract-layer
description: You are settling (or re-cutting) the contract layer of a package that wraps an unstable external protocol — and a human with taste is in the room to veto you. Use when a blueprint is about to be invented, replaced, or "tidied up", and the shape matters more than the code.
level: L1
---

# Designing a contract layer

## Scenario

You are about to write the contracts that every implementation will have to answer to: what a
launcher is, what a session is, what a connection exposes. The external protocol underneath is
unstable, so the contract is the only thing that can stay still. There is exactly one
implementation today, so nothing forced by reality will tell you whether the shape is right —
which is precisely when a contract layer either becomes the thing that keeps the project
coherent, or becomes a museum of guesses that everyone routes around.

The failure mode is not "wrong names". It is **collapse**: five contracts designed in one pass,
each coherent alone, incoherent together, and nobody can tell which one to fix.

## Method

Follow it in order. Each step exists to prevent a specific collapse; the collapse is written
down so you know when to stop and back up, not so you can skip it "because you know better".

1. **Settle names and boundaries before writing a line.** Ask "what is this thing, and what is
   it *not*" and stay in prose until the answer is short. Writing first means the structure is
   dragged by whatever you happened to implement.
   *Prevents:* implementations shaping the abstraction, which is backwards and permanent.

2. **One concept per round. Align it, then land it.** Do not design launcher, session and
   connection together, even if you can see all three.
   *Prevents:* an interconnected blob that has to be accepted or rejected as a whole.

3. **Data first, behaviour second.** Put the shared shapes into one place, as pure declarations;
   strip the plumbing that reads them (converters, projection helpers). Keep abstract classes —
   they are interfaces, not implementations. Then take the functional abstractions one at a
   time.
   *Prevents:* not being able to tell whether a failure is the data or the behaviour.

4. **Separate the protocol's vocabulary from ours.** The lower contract speaks the upstream's
   verbs faithfully, including the ones you would have designed differently. The upper contract
   adds your verbs on top. Mixing them is how a protocol layer turns into a framework.
   *Prevents:* an implementation that cannot be replaced, because our words were baked into the
   wire's shape.

5. **No configuration in contracts.** Ports, profiles, paths, credentials, probe routes — all
   concrete. A contract that requires them means every future transport edits the constitution.
   *Prevents:* the "universal" abstraction that only ever fits one transport.

6. **Optional abilities raise; they do not silently no-op.** Default implementations that throw
   `NotImplementedError` with a sentence beat a method that returns `None` and is never called.
   *Prevents:* observability hooks that quietly do nothing, which take a day to notice.

7. **Run a mechanical regression pass against the existing implementation.** For every contract
   member, check whether the implementation has it and whether the parameter *shape* matches.
   Sort the results into: no conflict / missing method / real conflict. Do not do this from
   memory — properties, keyword-only markers and return types hide in plain sight.
   *Prevents:* "it will plug in easily" — the belief that dies at integration time.

8. **Adjudicate each real conflict explicitly, then change the contract first.** For each
   conflict decide: contract wrong, implementation wrong, or both. Then edit the contract, then
   the implementation. A contract bent to match an implementation has stopped being a contract.
   *Prevents:* the contract becoming a description of one implementation.

9. **Verify with a real, end-to-end sample immediately.** Not the unit tests — a runnable skill
   that exercises the real thing (here: launch a real dsh and reach readiness). The contract can
   be self-consistent and completely unusable.
   *Prevents:* a clean blueprint that nothing can satisfy.

10. **Prefer copying over abstracting when a version will be cloned.** If the next generation is
    made by copying the current one, a small explicit module that gets copied is better than a
    clever shared layer that must be understood before it can be copied. Two copies you can diff
    beat one abstraction you must reason about.
    *Prevents:* a scaffolding layer that only pays off at version four, while costing every
    debugging session before that.

### Judgements that keep coming back

- **A name must point at a concept.** If you are naming a union type or a wrapper to save typing,
  stop: the name will read as a concept and nobody will find its definition. (`Launched` was
  deleted for this.)
- **A property, not a method, for a fact the object holds.** If it reads state and cannot fail, a
  caller writing `obj.thing()` is spelling out a call that does nothing.
- **A clean abstraction may sit on a stupid implementation.** `connection()` returning `self` is
  correct even if it looks lazy: the abstraction is what consumers see, and the implementation is
  allowed to be dumb.
- **Do not add a layer because it feels cleaner.** Extra layers make debugging harder for a model
  that reads code fluently. Every layer must buy something you can name in one sentence.
- **Write the reason, not only the rule.** The next reader must be able to overturn you on
  purpose. A rule without its failure mode is a rule that gets obeyed past its expiry.

## Exploration path

- `src/ghoshell_dsh/blueprint/` — the contracts; `__init__.py` carries the layer map.
- `src/ghoshell_dsh/blueprint/models/` — the data plane, declarations only.
- `src/ghoshell_dsh/v0_2_0/` — the one implementation; `wire.py` is the version's wire shape.
- `.features/` (package-local; `moss features` from this directory) — stages and workstreams:
  what period this is, and what was decided with what alternatives.
- `.features/SURFACE.md` — what this package promises outward. If a contract changes a promise,
  the surface is part of the change.

## How to trust this

The method is what produced the current blueprint, so re-run its own invariants:

1. **The layer map is still true** — read `blueprint/__init__.py` and check each arrow against
   the code. A stale map is worse than none.
2. **No config leaked into contracts** — `grep -rn "Config" blueprint/` should only match
   docstrings that say config is not here.
3. **Optional abilities raise** — `grep -rn "raise NotImplementedError" blueprint/` should cover
   every non-abstract method that is not a pure read.
4. **The regression table exists and is current** — re-run the mechanical comparison (member by
   member, parameter shape, not string signatures) after any contract change; the last pass is
   recorded in the commit that changed the contract.
5. **The end-to-end sample still runs** —
   `python src/ghoshell_dsh/skills/launching-a-dsh/sample.py` exits 0. If a contract change broke
   it, the change is wrong regardless of how clean it looks.
