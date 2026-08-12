---
created: 2026-08-10
depends:
- qa-exchange
description: 'Matrix 级通用授权机制 — 在 QA 之上补规则/凭据层。三层职责分离: qa 是交互协议, warrant 是存储 + 装线,
  permission 是业务逻辑。warrant 自 issuer 发起审批、自己等待、 随调用方 scope 取消。模板方法化: require 闭环抽象层默认实现,
  语言无关 key/type, PermissionStateData 弱类型载体, 有 state 即存。2026-08-12 迭代至 v7 抽象面。'
milestone: null
priority: P2
status: in-progress
status_note: v7 abstract rewritten 2026-08-12 (template-method require, PermissionStateData carrier, no persist); pending human review
title: Warrant
updated: '2026-08-12'
---

# Warrant

> Use `moss features set-status warrant <status> -m "note"` to update state.

## 重写 (2026-08-11)

前夜 (2026-08-10) 的 FEATURE.md 由记录模型压缩, 丢失关键设计: warrant 自 issuer 的
自动化机制、最小返回对象约定、存储归 warrant 的职责、save flag 两种保存模式、
`__aenter__` 生命周期对象; 类比引入的 Registry 冗余抽象一并去除。旧版见
git log `3f06c8a3`。2026-08-11 与人类重新正式化, 本版为准。

### v6 (2026-08-12)

人类 review 迭代抽象面至 v6 (本版为准):

- AuthorizationResult 从"最小对象 (仅 reason)"扩展为完整返回对象 (allowed/reason/state/persist),
  泛型化; require 去 None, allowed 单通道表达通过/拒绝 (KD3 推翻旧"最小对象")。
- Permission.namespace 移除, 归 warrant 约定 (KD11)。
- replied 不再返回 tuple, 改返回 AuthorizationResult (KD3)。
- Permission 增加语言无关 key/type classmethod (KD9, 弃用 Python import path)。
- 新增 WarrantMeta/WarrantState {meta,data} 弱类型载体, 对齐 Topic/QA 的 Model + Meta 模式 (KD12)。
- Warrant 新增 is_running/store/list_states (运行观测 + 主动存 + 枚举)。

### v7 (2026-08-12)

人类手改样貌, 模板方法化 (本版为准):

- require 闭环上移为抽象层默认实现 (模板方法, KD13), concrete 只填
  states/ask_question/store/list_states 四个原材料。
- WarrantMeta/WarrantState 合并为 PermissionStateData 单类弱类型载体
  (key + created + data, KD12)。
- persist 标志删除: 有 state 即存, permission 通过 result.state 是否有值控制存储 (KD3/KD5)。
- store 改收 PermissionStateData, 同步入队 + 内存缓存, 落盘 IO 由生命周期 task 消费队列 (KD5)。
- 修模板方法两 bug: finally 吞 CancelledError (取消沿 scope 传播), store 缺 from_state 转换。

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
StateT = TypeVar("StateT", bound=BaseModel)

class AuthorizationResult(BaseModel, Generic[StateT]):
    allowed: bool                           # 单通道: True 通过, False 拒绝
    reason: str | None = None               # allowed=False 时的拒绝理由
    state: StateT | None = None             # 变更后的授权状态; 有值即落盘

class Permission(ABC, Generic[StateT]):
    @classmethod
    def key(cls) -> str                     # 语言无关唯一键, 寻址每份 state. 形如 a.b.c
    @classmethod
    def type(cls) -> str                    # 语言无关类型标识, 约定 permission 类型. 形如 a.b.c
    def default(self) -> StateT             # 无存储时的初始状态; 返回实例的类型是 StateT 权威来源
    def check(self, state: StateT) -> Question | None
                                            # None = 无需授权; Question = 构造好的完整审批问题
    def replied(self, answer: Answer) -> AuthorizationResult[StateT]
                                            # 解释应答, 返回结果 (含新 state)

class PermissionStateData(BaseModel):
    key: str                                # 语言无关唯一键, 对应 permission.key()
    created: AwareDatetime                  # 记录创建时间
    data: dict[str, Any]                    # StateT 序列化本体 (model_dump)

class Warrant(ABC):
    async def __aenter__(self) -> Self      # 加载存储进缓存, spawn 落盘 task + 有序队列
    async def __aexit__(self, exc_type, exc, tb) -> None
    def is_running(self) -> bool            # 生命周期内 true
    def states(self) -> dict[str, PermissionStateData]     # 读缓存全量 (key → data)
    async def ask_question(self, question: Question) -> Answer   # 经 warrant namespace 发问题等答案
    def store(self, state: PermissionStateData) -> None    # 同步入队 + 内存缓存; 落盘 IO 由生命周期 task 执行
    def list_states(self) -> list[PermissionStateData]     # 枚举. 同步, 读缓存

    # 模板方法 (抽象层默认实现):
    def get_permission_state(self, permission) -> StateT   # 读缓存 + 还原强类型; 无记录/失败 fallback default
    async def require(self, permission: Permission[StateT]) -> AuthorizationResult[StateT]
                                            # 闭环默认实现, allowed 表达放行 (无 None)
```

### require 闭环 (模板方法, 抽象层默认实现)

1. `state = get_permission_state(permission)` — 读缓存 (states), 有记录则按
   `type(permission.default())` 还原强类型, 无/失败 fallback `permission.default()`
2. `q = permission.check(state)` — None 直接返回 `AuthorizationResult(allowed=True)`
3. 有 Question → `await ask_question(q)` — warrant 自 issuer 发到约定的 namespace,
   等应答; 调用方 scope 取消时 CancelledError 沿此传播
4. `result = permission.replied(answer)`
5. 有 `result.state` → `store(PermissionStateData.from_state(permission, state))` —
   同步入队, 落盘 IO 由生命周期 task 消费队列执行 (保序)
6. 返回 `result` (allowed 表达放行, 无 None)

### 存储时序与 save 模式

**要不要存** 由 permission 通过 `result.state` 是否有值隐式决定 — 有 state 即入队落盘,
无 (如 deny) 不存. permission 想在 deny 后记状态, 塞 state 即可.

**怎么存** 由 warrant 决定, 两模式按 cell 类型构建时选定:

- **写 storage 模式**: store 同步更新内存缓存 + 推入有序队列, 落盘 IO 由 `__aenter__`
  创建的生命周期 task 消费队列执行, 保序. 第一版实现.
- **topic 模式**: 发 topic 广播 "某 state 要存", 由持有存储权 warrant 的 cell 同步落盘.
  分布式 / 跨 cell 保存. 留扩展.

## Key Decisions

- **KD1 三层职责分离**: qa 交互协议 / warrant 存储+装线 / permission 业务逻辑。
  warrant 是唯一 IO 面, permission 是纯逻辑, qa 是底层协议。这是本设计的地基。
- **KD2 warrant 自 issuer, 自动化**: warrant 发起 question、自己等待、随调用方 scope
  cancel。与 ghost 无关; 给 ghost 的提示是旁路信息, 由接入方自理。
- **KD3 完整返回对象**: AuthorizationResult 承载 allowed/reason/state, 泛型化;
  require 统一返回 (无 None), allowed 单通道表达通过/拒绝。需要结构化扩展的 concrete
  继承本类加字段。
- **KD4 存储归 warrant**: permission 不碰持久化。warrant 读 state → permission 只决定
  逻辑 → warrant 写回。
- **KD5 存储时序 + save 两模式**: store 同步更新内存缓存 + 入有序队列, 落盘 IO 由
  生命周期 task 消费队列执行 (保序); topic 广播 / 真实写按 cell 类型构建时选定。
  **要不要存** 由 permission 通过 result.state 是否有值隐式决定。
- **KD6 生命周期对象**: 存储动作在 `__aenter__` 创建的生命周期对象的异步 task 里执行;
  取消随调用方 scope 传播 (cancel question 同路径)。
- **KD7 fail-open + 单例**: warrant 可选能力, 从 IoC 取, 单例; 拿不到 → 放行。需要
  fail-closed 的 concrete 自己声明, 抽象不预设。
- **KD8 命名 state**: config 有歧义弃用。静态授权参数在 `__init__` 配置, 存下来的是
  动态授权状态 (StateT)。
- **KD9 state 粒度 + 语言无关 key**: 按语言无关 key 存一份。key/type 是人工约定的路径
  字符串 (形如 a.b.c, 类似 topic_name/topic_type), 不依赖任何语言的模块结构 — Python
  import path 最不语言无关, 弃用。参数化 permission 的实体隔离在 state 文档内部由
  concrete 管 (check 拿整份, replied 还整份), warrant 保持哑。
- **KD10 存储作用域**: 默认 session-scope (会话内产物, 不跨会话污染); 跨 session 持久化
  的 concrete 显式覆盖 (逃生门)。
- **KD11 namespace 归 warrant**: 审批问题发往的 qa namespace 由 warrant 约定, 不在
  permission 声明。软身份靠 Question 内容/kind 区分; 未来需要分流时 warrant 内部按
  permission 映射。
- **KD12 弱类型载体**: 存储/传输用 PermissionStateData 弱类型载体 (key + created +
  data), 对齐 Topic/QA 的 Model + Meta 模式; 强类型还原靠 permission (default 返回
  实例的类型是 StateT 权威来源), warrant 保持哑。
- **KD13 模板方法**: require 闭环是唯一算法骨架, 在抽象层给默认实现; concrete 只填
  states/ask_question/store/list_states 四个原材料。取消沿调用方 scope 传播, 存储
  时序由生命周期对象保证。

## 落点

- 概念: `core/blueprint/warrant.py`。已按模板方法版重写 Permission/AuthorizationResult/
  Warrant + PermissionStateData 弱类型载体 (2026-08-12, 待 review 定稿)。
- warrant 从 IoC 取, 单例, 挂 session 语义但不绑 Session ABC (KD7)。concrete
  (SessionWarrant) 在 storage + qa 之上装线, 待做。
- 依赖: `qa-exchange` (`core/concepts/qa.py`)。

## 待做

- [x] 概念层实现: `core/blueprint/warrant.py` 按模板方法版重写 (2026-08-12, 待 review)
- [ ] warrant 存储层: session-scope state 读写 (PermissionStateData 载体), 队列落盘
      (topic 模式留扩展); concrete SessionWarrant 在 storage + qa 上装线
- [ ] 一个验证场景 + 测试 (驱动完整闭环)
- [ ] topic 模式接收侧 (存储权 cell 监听落盘) — 出现跨 cell 场景时再确认