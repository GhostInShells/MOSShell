---
label: "r2-03"
kind: "mermaid"
description: "按角色还是按职责：角色用身份替代信息，职责用边界替代身份；判据三条，边界的最小单元是契约不是文件"
created_at: "2026-10-29T15:13:30"
---

flowchart LR
  subgraph ROLE["角色：用身份替代信息"]
    direction TB
    R1["起因：上下文窄（专业性 = 小上下文）"]
    R2["决策退化为博弈：只换结论，不换上下文"]
    R3["症状：产品经理不懂程序"]
    R1 --> R2 --> R3
  end
  subgraph DUTY["职责：用边界替代身份"]
    direction TB
    D1["前提：共享认知，交换便宜"]
    D2["只声明写边界 + 触发条件"]
    D3["共识不靠说服，靠判据可裁决"]
    D1 --> D2 --> D3
  end
  subgraph CHECK["判据"]
    direction TB
    J1["删掉「必须某身份才能做」还能跑 → 职责"]
    J2["边界的最小完整单元不是文件，是契约"]
    J3["职责的形状 = 并行拓扑上的割"]
  end
  R2 -->|"绕开的是共识成本"| J1
  D3 --> J1
  D2 --> J2
  D1 --> J3
