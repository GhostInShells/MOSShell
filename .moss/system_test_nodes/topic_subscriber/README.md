# topic_subscriber

A generic topic drainer probe: subscribes to an arbitrary topic by name and prints
each raw `Topic` as one JSON line. Replaces the old `clause_subscriber` (hardcoded
to `ClauseTopic`) — one node for any topic, no per-topic node needed.

## What it does

Subscribes via `matrix.session.topics.subscribe(topic_name)` (cross-process over
zenoh, no topic model), blocks on `poll()`, and prints:

    audio/sample #1: {"meta":{...},"data":{"role":"user","rms_db":-30.1,...}}

## Test method (recorded)

1. Start this node with the topic name as an argument:

       moss nodes run .moss/system_test_nodes/topic_subscriber/ -- audio/sample

2. In another process, produce audio for that topic (e.g. `moss audio listen` for
   `role=user`, or `moss-shell` speaking for `role=ghost`).

3. Count the printed lines — each is one published topic, self-contained JSON.

Run in the same network scope as the producer (the default scope from `.moss`).
