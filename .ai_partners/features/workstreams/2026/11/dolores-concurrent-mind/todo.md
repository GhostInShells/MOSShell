# Todo — construction tasks

Things to do while building this workstream. Split off, not urgent: record only, no
investigation — this file is an attention offload, not a plan.

| # | Task | Status |
|---|------|--------|
| 0 | One-shot readonly request that leaves **no** session log. Today `bypass/run` always calls `agents.create({sessionId: randomUUID(), seed})` and only disables tools — the log stays and the session is attached to the home workspace. Needs an ephemeral path in dsh: no log, no workspace attach. | todo |
| 1 | ~~Bypass commit must run without a session log.~~ Same root cause as #0. | merged → 0 |
| 2 | A workspace like ego's: the ego inside it should be auto-archived. | todo |
| 3 | Remove the two blocks "未分类" and "home" that currently show on the Dolores ghost prototype's dsh surface. | todo |
| 4 | Make dsh's subagent runnable — or remove it and rebuild. Without post-training the main path won't issue a bypass even when able, so the important triggers must be automatic, not manual; stand it up config-driven, observable and editable for the productization side. | todo |
| 5 | The `request` carrier for reply clones: schema-checked, retryable, billable. `bypass/run` returns one text message today. v1 shape: temporary tool definition + forced `tool_choice` (costs one extra round) rather than waiting for provider structured output. | todo |
| 6 | **Memento memory surface**: `<memento owner="…" branch="…">` replaces the bare `<branch>` container; minute-precision local `at=`; confluence / child rows rendered inside the commit that absorbed them; a reserved slot at the branch opening for the forked parent instead of the currently-dropped `BranchView.previous`; digest sample and window sizes become config (`digest_titles`, `view_limit`), not module constants. | todo |
| 7 | **Cross-branch read**: a `branches()` roster command (archived included, read from `branches.jsonl` rather than the `*.ref.json` glob), `view` / `history` accepting an index as well as a name, and an instruction line saying that any coordinate under the owner is readable while the write boundary does not move. | todo |
| 8 | **`LLMFunc` + staged search**: define the typed `LLMFunc` contract (schema in / schema out / prompt asset / effort / budget / cache family) on top of the bypass path, then `search` as its first instance — L0 grep, L1 score and cut in program, L2 rank only above a threshold. | todo |
| 9 | **Ground into the system prompt**: split `_runtime.system_prompt()` into named slots, render the ground's law as one slot, sync a manifest (order + hash + text) to the plugin before the ego session opens; replacement happens at session creation only, never mid-session. | todo |
| 10 | **Keepalive**: the prefix family dies after ~5 min idle, so a pause costs the whole context. Ping on a configurable cadence with the inherited effort and `maxTokens=1`; guard against header drift (a new static head invalidates everything anyway). | todo |
