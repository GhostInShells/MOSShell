---
label: "r2-11"
kind: "mermaid"
description: "修正：op 不直连 runtime，总线是唯一的地址翻译处；族 = 同一套 op 契约 + 同一个 facts 根；ghost 是 root"
created_at: "2026-10-29T16:47:27"
---

flowchart TB
  G["ghost：最顶层，root matter 由它签发"]
  O["op：只写「对谁做」，从不写「谁来做」"]
  B["总线：唯一的地址翻译处"]
  R1["dsh harness"]
  R2["另一个族实例 / 异种 runtime"]
  G --> B
  O --> B
  B -->|"按控制协议分发"| R1
  B -->|"按控制协议分发"| R2
  R1 -.->|"只与总线通讯，不认识别的 runtime"| B
  R2 -.->|"只与总线通讯，不认识别的 runtime"| B
  subgraph FRAME["族 = 同一套 op 契约 + 同一个 facts 根"]
    direction TB
    F1["族内：共享 facts，传的是寻址不是上下文 -> 消息 O(1)"]
    F2["族间：A2A 式，传上下文或快照"]
  end
