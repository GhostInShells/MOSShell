---
created: 2026-08-10
depends:
- qa-exchange
description: 'Matrix 级通用授权机制 — 在 QA 之上补规则/凭据层。三层职责分离: qa 是交互协议, warrant 是存储 + 装线,
  permission 是业务逻辑。warrant 自 issuer 发起审批、自己等待、 随调用方 scope 取消。返回最小对象 (仅拒绝描述)。2026-08-11
  由人类重新正式化, 推翻 08-10 草稿记录。'
milestone: null
priority: P2
status: in-progress
status_note: design re-formalized with human; abstract rewritten (Permission/AuthorizationResult/Warrant)
title: Warrant
updated: '2026-08-11'
---

# Warrant

> Use `moss features set-status warrant <status> -m "note"` to update state.

## 重写 (2026-08-11)

前夜 (2026-08-10) 的 FEATURE.md 由记录模型压缩, 丢失关键设计: warrant 自 issuer 的
自动化机制、最小返回对象约定、存储归 warrant 的职责、save flag 两种保存模式、
`__aenter__` 生命周期对象; 类比引入的 Registry 冗余抽象一并去除。旧版见
git log `3f06c8a3`。2026-08-11 与人类重新正式化, 本版为准。

## Motivation

qa-exchange (completed) 提供广播问答交换协议: 问题广播 / 应答 / issuer 裁定终态,
解决审批的**传输层**。授权体系有三个命题 QA 不碰:

1. **谁有权批准** — QA 的 watcher 任意应答者先到先得, 无身份强约束。Warrant 的答案:
   namespace 分流 (软身份), 且机制本身自动化, 不依赖人类在场。
2. **规则从哪来** — 哪些操作需要授权、按什么策略判定。答案: Permission 的业务逻辑。
3. **批准后的凭据** — approve 之后被授权方拿什么去执行。答案: 授权状态 (state)
   持久化, "批准一次以后不再问"。

## 设计

### 三层职责 (核心, 是职责归属不是命名)

| 层 | 职责 | IO |
|---|---|---|
| **qa** | 交互协议 — 问题/答案的传输与生命周期 (issue/wait/cancel) | 通讯 |
| **warrant** | 存储 + 装线 — 读 state, 发 QA, 写回 state | 所有 IO |
| **permission** | 业务逻辑 — 决定何时要问、怎么解释应答、要不要存 | 无 IO |

question/answer 是**软件组件之间的对话协议**, 不是给人或 ghost 的。warrant 自 issuer
发起问题、自己等待、随调用方 scope 取消。人类/TUI/GUI 看 namespace 只是 watcher 的
一种可能, 机制不依赖它存在。给 ghost 的提示由 command / 接入方自己从 concrete 结果
里取, warrant 不负责。command 要走 QA 拿实参时, 直接自己调 QA, 不走 warrant。

### 抽象面

```python
class Permission(ABC, Generic[PermissionState]):
    namespace: str                          # 审批问题发往的 qa namespace
    def default(self) -> PermissionState    # 无存储时的初始状态
    def check(self, state: PermissionState) -> Question | None
                                            # None = 无需授权; Question = 构造好的完整审批问题
    def replied(self, answer: Answer) -> tuple[PermissionState, AuthorizationResult, bool]
                                            # 新 state + 结果 + 是否更新存储 (save)

class AuthorizationResult(BaseModel):
    reason: str | None = None               # None = 通过, str = 拒绝理由. 最小对象, 只有这个

class Warrant(ABC):
    async def __aenter__(self) -> Self      # 创建生命周期对象 (存储/QA 协调)
    async def __aexit__(self, exc_type, exc, tb) -> None
    async def require(self, permission: Permission) -> AuthorizationResult | None
```

### require 闭环

1. `state = storage.read(permission)` 或 `permission.default()`
2. `q = permission.check(state)` — None 则直接放行
3. 有 Question → warrant 自 issuer 发到 `permission.namespace`
4. `wait()` 等应答; 取消随调用方 scope (问题被 cancel)
5. `new_state, result, save = permission.replied(answer)`
6. `save` → 存储动作按 cell 类型选模式执行, 在生命周期对象的异步 task 里
7. 返回 `result` (None = 放行)

### save flag: 两种保存模式, 按 cell 类型选

warrant 携带 save flag, 决定**怎么存**。构建 warrant 时判断 cell 类型选定:

- **topic 模式**: 发 topic 广播 "某 state 要存", 由持有存储权 warrant 的 cell 同步落盘。
  分布式 / 跨 cell 保存。
- **写 storage 模式**: 真实写本地 storage。

**要不要存** 是 permission 的业务判断 (replied 第三值), 与 warrant 的**怎么存** (机制)
分开。

## Key Decisions

- **KD1 三层职责分离**: qa 交互协议 / warrant 存储+装线 / permission 业务逻辑。
  warrant 是唯一 IO 面, permission 是纯逻辑, qa 是底层协议。这是本设计的地基。
- **KD2 warrant 自 issuer, 自动化**: warrant 发起 question、自己等待、随调用方 scope
  cancel。与 ghost 无关; 给 ghost 的提示是旁路信息, 由接入方自理。
- **KD3 最小返回对象**: AuthorizationResult 只有 reason (None=通过, str=拒绝理由)。
  需要结构化结果的 concrete 自己扩展接口, issue 处见 concrete 类型, 抽象不约定。
- **KD4 存储归 warrant**: permission 不碰持久化。warrant 读 state → permission 只决定
  逻辑 → warrant 写回。
- **KD5 save flag 两模式**: topic 广播 / 真实写, 按 cell 类型在构建 warrant 时选定;
  permission 决定要不要存。
- **KD6 生命周期对象**: 存储动作在 `__aenter__` 创建的生命周期对象的异步 task 里执行;
  取消随调用方 scope 传播 (cancel question 同路径)。
- **KD7 fail-open**: warrant 可选能力, 从 IoC 取; 拿不到 → 放行。需要 fail-closed 的
  concrete 自己声明, 抽象不预设。
- **KD8 命名 state**: config 有歧义弃用。静态授权参数在 `__init__` 配置, 存下来的是
  动态授权状态 (PermissionState)。
- **KD9 state 粒度**: 按 permission class 存一份; 参数化 permission 的实体隔离在 state
  文档内部由 concrete 管 (check 拿整份, replied 还整份), warrant 保持哑。
- **KD10 存储作用域**: 默认 session-scope (会话内产物, 不跨会话污染); 跨 session 持久化
  的 concrete 显式覆盖 (逃生门)。

## 落点

- 概念: `core/blueprint/warrant.py`。当前骨架仍用旧名 (Item/ItemMeta/ItemModel), 待按
  本版重写。
- warrant 从 IoC 取, 挂 session 语义但不绑 Session ABC (KD7)。
- 依赖: `qa-exchange` (`core/concepts/qa.py`)。

## 待做

- [ ] 概念层实现: `core/blueprint/warrant.py` 落地 Permission/AuthorizationResult/
      Warrant 抽象 (permission check/replied 同步, warrant require + context manager async)
- [ ] warrant 存储层: session-scope state 读写, save flag (topic 模式留扩展)
- [ ] 一个验证场景 + 测试 (驱动完整闭环)
- [ ] topic 模式接收侧 (存储权 cell 监听落盘) — 出现跨 cell 场景时再确认