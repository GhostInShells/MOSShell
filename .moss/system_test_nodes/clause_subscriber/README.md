# clause_subscriber

A clause-topic drainer probe: subscribes to the `clause` topic and prints each
`ClauseTopic` as one JSON line. Verifies the speaking-side wiring — that
`moss_runtime._clause_topic_bridge` broadcasts speech's `on_clause` results as
`ClauseTopic` — and symmetrically covers the listening side (ASR clause
finalization on the same topic).

## What it does

Subscribes via `matrix.session.topics.subscribe_model(ClauseTopic)` (cross-process
over zenoh), blocks on `poll_model()`, and prints:

    clause #1: {"text":"...","speaker_id":"...","speaker_name":"...","role":"ghost","lang":"","meta":{...}}

## Test method (recorded)

1. Start this node:

       moss nodes run .moss/system_test_nodes/clause_subscriber/

2. In another process, run `moss-shell` and speak several sentences in the TUI.

3. Exit `moss-shell`.

4. Count the `clause #n:` lines this node printed — it should equal the number
   of sentences spoken (punctuation-driven clause splits). Each line carries
   `role`, `speaker_name`, `text`, and `meta.created_at`.

Run in the same network scope as the speaker (the default scope from `.moss`).
