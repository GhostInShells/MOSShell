---
created: 2026-08-12
depends:
- session-communication-bus
description: 'Parameter 重做为 host-truth 广播: host 持真值并广播, worker 读真值上报本地写; declare 即
  require (version -1/0/>=1), version 按 epoch(host 地址)作用域、host 重启从头算; key 组 address-free;
  Memory/Zenoh 双 transport; parameter 回到 session.'
milestone: null
priority: P1
status: completed
status_note: host-truth 广播; 16 测试 (memory 13 + zenoh 3) 绿, 含跨 session host 重启 (新化身续值/版本从头算)
title: Parameter Host Truth
updated: '2026-09-25'
---

# Parameter Host Truth

> Use `moss features set-status parameter-host-truth <status> -m "note"` to update state.

## 历史

parameter 源自 session-communication-bus (2026-05, completed) 的 D10/D12, 2026-08-12
因 voice-input-state-machine 的前值需求独立成篇。一路走了三段:

1. **SessionParameterStore** (SQLite 真值 + zenoh 失效信号) — 依赖方向反了 (parameter 反向吃
   上层 Session), 且"同机读同一 tmp_storage"的真相源假设跨网络不成立, 2026-08-29 删。
2. **点对点 + cell address** (2026-09-01 收敛) — 单声明者 + declare/subscribe, 但把 cell
   address 塞进了 key。**这是装线事故**: address 是动态 uid, 不该做 parameter 身份; 真相源
   应该是 host 广播, 不是每个 cell 自说自话。且 parameter 被移出 session 到 matrix 面。
3. **host-truth 广播** (本轮) — 回到 session, host 持真值广播, worker 读真值上报。见 Key Decisions。

## Motivation

跨进程的共享状态需要单一真相源。之前的"点对点"把真值散落在每个声明它的 cell 里, 没有
仲裁; 而 cross-project 的远程节点不随 host 重启被清理, 需要识别"host 重启了、真值从头算"。

## Key Decisions

### D1. host 是唯一真值源, 广播

host 持真值并广播变更; worker 读真值、上报本地写; host 缺席则本地值即真相。
多个 worker 可上报同一 key, 由 host 仲裁 (单写者语义退化为单真值源)。

### D2. declare 即 require (version 三值)

统一规则: **declaration 进 → truth 必出**。每条进入 host 的声明, 采纳与否都引发一次真值广播;
采纳与否只决定广播的是谁的值。因此 declare 本身就是 require, 没有独立拉原语。
version 三值: `-1` 仅要求广播一次; `0` 首包 (无真值时采纳); `>=1` 真值声明。

### D3. version 按 epoch 作用域

epoch = host 地址 (每实例唯一, uid 非持久化)。跨 epoch 的版本不可比: host 重启 (新 address)
后 version 从头算。消费者 `_accept_truth` / host `_absorb_version` 都按 epoch 裁决。

### D4. key 组 address-free

address 只活在 payload 的 `ParameterMeta`, 不进 key (形如 `TopicMeta.sender`)。
key 组: `host/truth/{key}` / `worker/declaration/{key}` / `host/liveness/{address}`。

### D5. 双 transport

- Memory (`MemoryBus` + `MemoryParametersBroadcaster`) — 进程内, 收敛逻辑可离线断言。
- Zenoh (`ZenohParametersBroadcaster`) — 映射到 zenoh key expr。

### D6. parameter 回到 session

session 是聚合面 (持 zenoh session), parameter 一开始就该挂 session。session 内 is_host
判定, 分派 `TruthHostParameters` / `WorkerParameters`。matrix 面废弃。

## Implementation

- [x] `TruthHostParameters` / `WorkerParameters` / `ParametersBroadcaster` (`core/parameter/_base.py`)
- [x] Memory transport (`MemoryBus` / `MemoryParametersBroadcaster`)
- [x] Zenoh transport (`ParameterNamespace` / `ZenohParametersBroadcaster`)
- [x] session 面装线 (`session.parameters` + is_host 分派)
- [x] 16 测试 (memory 13 + zenoh 3)
- [x] 实机同步验证 (parameter_probe declarer/subscriber)
- [x] zenoh liveness 跨 session 集成测试 (host 重启) — `test_host_restart_over_zenoh_extends_value_with_new_epoch`

## 复盘

上一版"点对点 + cell address"是装线事故, 不是设计演进: 08-12 草稿就是 host-as-truth (host
唯一真相 + 广播/query), 09-01 的 converge 提交在同一次提交里静默改写成了 point-to-point
address, 没留碰撞记录。人类回忆确认: "进入 matrix"这个位置有印象, 但 address 耦合进 key
从来不是设计初衷 — address 是动态的, 不能做 parameter 身份。

教训: 机制没变, 是线接反了。host 单点真值 + 广播/query + 弱一致, 复杂度塌缩; 强状态观测需自建。
