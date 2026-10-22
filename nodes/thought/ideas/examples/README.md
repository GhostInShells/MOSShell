# examples

Each first-level directory here is a **group**, and a group is one `graph.json`:

```json
{
  "rev": 3,
  "meta": {"concepts": {"core": "#e8f5ea", "risk": "#d0715c"}},
  "nodes": {"n1": {"label": "动机", "body": "...", "concept": "core", "weight": 1.0}},
  "edges": [{"id": "e1", "from": "n1", "to": "n2", "kind": "detail"}]
}
```

`ops.jsonl` beside it is the account of how it got there — disposable, used only by
`diff`. Everything here is an example: it is copied into the data root on first run and
then belongs to whoever owns that root. Delete it; the node never assumes its data is
precious.
