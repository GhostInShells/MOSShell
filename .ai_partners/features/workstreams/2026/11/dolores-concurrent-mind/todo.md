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
