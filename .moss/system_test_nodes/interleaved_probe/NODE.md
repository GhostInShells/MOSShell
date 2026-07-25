---
name: 'interleaved_probe'
description: 'Semantic instrument panel for the interleaved-ctml-thinking workstream — controlled test cases for cursor-projection semantics.'
singleton: true
exec:
  command: python
  args: main.py
---

# interleaved_probe

Test instrument for the interleaved-thinking control paradigm. Channel `probe`
offers one command per cursor-projection semantic — nothing here touches the
real world, so test tracks are fully controlled and repeatable.

| Command | Semantic under test | Expected projection |
| --- | --- | --- |
| `probe:slow duration steps` | live progress of a running task | `step k/steps` visible mid-flight |
| `probe:emit value` | observed non-empty result | result with identity |
| `probe:silent_observed` | observe=True + empty outcome (K9) | placeholder with identity, never evaporates |
| `probe:silent_plain` | unobserved empty success | folds into success tally, no identity |
| `probe:fail msg` | runtime failure | errmsg with identity |
| `probe:critical msg` | critical observation error | interrupts dispatch; fail-closed trigger |
| `probe.a:tick` / `probe.b:tick` | parallel FIFO tracks | per-channel cursor + cancel cut anchors |

Typical test tracks:

```ctml
<!-- progress + interrupt timing: interrupt mid-flight, check the cut point -->
<probe:slow duration="60.0" steps="60"/>
<probe:emit _cid="after" value="did I survive the cut?"/>

<!-- per-channel cut anchors: two parallel tracks, then interrupt -->
<probe.a:tick times="30"/>
<probe.b:tick times="30" interval="2.0"/>

<!-- K9 pair: same emptiness, different observe mark, different projection -->
<probe:silent_observed _cid="watched"/>
<probe:silent_plain/>
```
