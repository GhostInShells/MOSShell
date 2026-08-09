---
title: Warrant
status: draft
priority: P2
created: 2026-08-10
updated: 2026-08-10
depends: [qa-exchange]
milestone:
description: >-
  Matrix 级通用授权体系 — 在 QA Exchange 之上补规则/凭据层. Permission 声明
  授权场景 (何时要问/如何裁决), PermissionRegistry 做共享配置目录 + 全局授权/拒绝,
  Warrant 做执行门 (发 QA 审批, fail-open). 第一期单机版, 跨 cell 同步留扩展点.
---

# Warrant

> Use `moss features set-status warrant <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

qa-exchange (completed) 提供了广播问答交换协议: Asker 广播问题 / Watcher 应答 /
requester 持真相。它解决的是**审批的传输层**——把"问题广播出去、拿到答案、裁定终态"
这个交换过程做对。

但授权体系有三个核心命题 QA 不碰:

1. **谁有权批准** — QA 的 watcher 是"任意应答者先到先得",没有身份/权限强约束。
2. **规则从哪来** — 哪些操作需要授权、按什么策略判定,QA 完全不涉及。
3. **批准后的凭据** — approve 之后被授权方拿什么去执行,QA 只返回一个 Answer。

Warrant 在 QA 之上补这三层。**配置驱动**: 授权结果沉淀为配置更新,天然支持
"批准一次以后不再问"。

边界: SafeMode 闸 logos 级 (进程内 Future, articulate→action 链路); QA 是
action/question 粒度跨进程广播; Warrant 是 QA 之上的规则/配置层,三者通道不合并。

## Design Index

- 概念蓝图: `src/ghoshell_moss/core/blueprint/warrant.py` (当前只有骨架草图)
- 前身/依赖: `qa-exchange` FEATURE.md + `core/concepts/qa.py`
- 参照: `ResourceRegistry` (contracts/resource.py:341) — registry 形状

## Key Decisions

### KD1: 授权配置同步 = 唯一服务模型 (host 权威 + topic 广播)

配置跨 cell 共享时的同步模型:

- **host 节点**: 权威写者 — 写文件 (持久化 config) + 聆听广播 + 存储。
- **非 host 节点**: 内存持有 + 聆听广播 + 广播自己的变更。
- **topic 广播就够**: 授权配置是**广播数据 (push)**,不是问题-答案 (ask)。
  与 QA 的既有区分对齐: topic = 广播数据, QA = 广播问题 + 带回答案。

**第一期不绑 topic** — 单机版 registry 本地持有 (host 落盘到 session-scope
storage, 非 host 内存)。跨 cell 同步留 `PermissionRegistry` 抽象扩展点, 不实现。
没有真实场景在消费"跨 cell 授权配置同步"之前, 不上 topic 复杂度。

### KD2: 三层抽象 — Permission / PermissionRegistry / Warrant

| 层 | 职责 |
|---|---|
| **Permission** (场景声明) | config schema + namespace + check/replied 策略 |
| **PermissionRegistry** (共享配置) | topic 广播同步, host 权威落盘, 目录 + 全局授权/拒绝 |
| **Warrant** (执行门) | 读 registry config → 需要审批发 QA → 裁决 → 写回 registry |

### KD3: Warrant 从 IoC 取, 不绑 Session ABC — fail-open

Warrant 是**可选能力** (拿不到就当授权), 因此不绑 Session 契约 (绑 Session 意味着
每个 session 必须有 warrant)。从 IoC container 取, fetch 不到返回默认放行实现
(check → None)。

**fail-open 是默认信任模型**: 未配置授权体系的运行时默认放行。需要 fail-closed
(拿不到必须拒)的具体 Permission 自己声明, 抽象不预设。

### KD4: 存储默认 session-scope, Permission 可覆盖 (逃生门)

- **默认**: session-scope storage (常驻, 最小粒度)。授权配置是"会话内交互的产物",
  ghost 请求 → 人类批准 → 会话内生效, session 隔离不跨会话污染。
- **逃生门**: 需要跨 session 持久化 (如"永远允许"、系统级安全配置) 的 Permission
  在 `factory(container)` 里覆盖自己的 storage。
- matrix 级 scoped storage (runtime_scopes 四维) 不是现行隔离机制, 不用。
- 配置用现成 `Storage.read_yaml/write_yaml` 序列化成 `{permission_type}/{config}.yml`。

### KD5: 审批走 QA namespace 分流, question 给人类/TUI/GUI

每个 Permission 声明 `namespace()`, 审批问题发到对应 qa namespace, 由人类/
TUI/GUI watch 应答。Warrant 从 concrete Permission 身上取 namespace, 骨架零配置。

### KD6: 骨架只管发, 具体 Permission 自扩展

抽象层薄到只保证骨架能跑 (统一无聊层, 特别化有趣层)。Permission 都是 concrete
实现, 需要丰富结果 (凭据/令牌/会话) 时自己额外暴露接口, issue 处看到的永远是
concrete 类型。抽象不约定。

- `check` 返回 str|None 的 str 是**默认出口** — 自然语言提示 (通过=成功 note,
  拒绝=理由)。需要结构化结果的 Permission 自己加 result 方法。
- `replied` 返回 `AuthorizationResult` (config + granted + reason) — 最小数据载体,
  骨架写回 config / 用 granted 放行 / 给 reason 提示。

### KD7: Question 由 Permission 自建, 骨架只发

`Permission.check(config)` 返回构造好的完整 Question (带 kind/content/选项),
Warrant 只管发出去。Permission 知道自己的话术和审批面。

## Implementation Notes

### 命名沿革

最初草图叫 Item/ItemModel/ItemMeta (人类自认瞎写), 评审后重命名:

| 草图 | 定稿 | 理由 |
|---|---|---|
| Item | Permission | 授权场景, 不跟 ResourceItem 撞 |
| ItemModel / ITEM_MODEL | PermissionConfig / CONFIG | 就是持久化状态 |
| ItemMeta | PermissionMeta | 列表化投影 |
| — | AuthorizationResult | 裸 tuple (config, granted, reason) 换结构化对象 |

### 落点

- `PermissionRegistry` 挂 matrix (跨 cell 共享, 走 IoC 不走 operator)。
- `Warrant` 挂 session 语义 (审批是会话内交互, `session.qa` 已在 Session ABC),
  但**不绑 Session 契约**, 从 IoC 取 (见 KD3)。
- `Warrant` 机制与 cell 无关。

## 待做

- [ ] 概念层实现: `core/blueprint/warrant.py` 落地 KD2 三层 + PermissionConfig/
      PermissionMeta/AuthorizationResult
- [ ] 单机版 PermissionRegistry (session-scope storage 落盘)
- [ ] Warrant 骨架 (读 config → check → 发 QA → replied → 写回)
- [ ] 一个验证场景 (如 speak/shell 审批) + 测试
