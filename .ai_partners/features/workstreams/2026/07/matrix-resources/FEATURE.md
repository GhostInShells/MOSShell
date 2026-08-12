---
title: Matrix Resources
status: design-locked
priority: P1
created: 2026-07-23
updated: 2026-08-13
depends: [resource-http-endpoint]
milestone:
description: >-
  Matrix 层的资源投影, 统一寻址 scheme://{cell-address-short}/uri-path — host 位是 cell 身份.
  解析本地先查 (scheme, host), miss 走 zenoh. manifest 静态资源保持任意 host 不迁移 (向前兼容);
  动态资源经 provide_resource 纯声明入网. 协议极薄: 主路径 get (uri → messages), 全局 list/recall/query
  全部 channel 治理. 交换物为 Message, 默认渲染 meta JSON. 第一 milestone: text resource 跨场景获取.
---

# Matrix Resources

> Use `moss features set-status matrix-resources <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

Ghost 运行时的资源有两种来源：一种能经 OS 本地文件系统拿到（文件名即句柄，最省事）；
另一种只在**组网时才存在**——它活在某个 cell 的进程内存或 cell home 里，只能经 Matrix
网络拿到。

**统一寻址（2026-08-05 收敛）**：所有资源都是 `scheme://{host}/path`。host 位是 **cell 身份**
（cell-address-short，即 fullname）。解析规则一个谓词：本地 registry 按 `(scheme, host)` 先查
（命中即本地），miss 走 zenoh（把 host 当 cell 身份路由到对应 cell 的 queryable）。由此
project resources 与 network resources 不再靠 scheme 区分，区别退化为一个 bit——有没有被
provide_resource（有没有 declare queryable）。这个 bit 只决定"别人能不能解析我"，不决定
"我怎么寻址"。历史背景：resources 概念诞生时还没有 cell，只有 cell 的思路，所以旧设计没有
用 cell 身份寻址；现在补上了。

**操作分层**：resources 服务于 ghost 的认知，不是围绕 OS 做通用文件交换（不是网盘）。
协议主路径是 **get(uri)**——精确句柄、描述正确的资源。全局 list/recall/query（模糊搜索、
跨 cell 归并、排序）全部推给 channel 治理，协议不提供。资源**不是 ghost 的上下文全集**：
只有当它被 get + 正确描述（meta 够好）+ ghost 可理解时，才成为上下文变量；否则它只是
cell 运行时的动态计算逻辑，不占 ghost 历史。这同时是 **compact 不遗忘** 的机制——引用（短
URI 字符串）活着，内容才按需拉取。

驱动场景：nodes 重建后各自起 HTTP server。若有一个 `servers://` scheme，ghost 能查到网络
上现在有哪些 server 端点存活——但"列出"走 cell 的 channel 命令（cell 知道自己起了哪些 server），
resources 协议只负责 get 单个端点的详情。没有这层，ghost 上下文里就缺一种 compact 之后不遗忘
的机制——端点信息散落在历史消息里，压缩即丢。

**承接关系**：`resource-http-endpoint`（2026-06-24, completed）当时明确把"通用 streaming
resource 接口（`stream() -> AsyncIterator[bytes]` + content_type + size，不依赖本地文件）"
推迟为后续 feature。本 workstream 是那笔欠账的到期兑付点（见 Key Decision 4 的 data 面）。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`（2026-07-23 命名考据；2026-08-05 统一寻址收敛）
- 承接: `resource-http-endpoint` FEATURE.md（HTTP 访问层，data 面引用的逃生门之一）
- 归属: **Matrix 层**。读侧 = `matrix.resources` 换成/包一层 resolver（本地先查 + 网络 miss）；
  provide 侧 = `CellPresence.provide_resource`（matrix 提供对称糖）。

## Key Decisions

### 1. 交换物 = Message（+ ResourceInfo 作 meta），协议极薄（get 为主）

**选择**：跨网络传输的只有两种 Pydantic 对象——`ResourceInfo`（meta，`as_content()` 已是给
AI 读的 JSON）和 `Message`（"MOSS 体系上行给模型的消息体"，多模态、可嵌套、可去重）。zenoh 层
只需要**一个** scheme-agnostic 的 queryable 协议：`request = {op, args}`，`reply = error |
infos | messages`。

**协议极薄（2026-08-05 修订）**：全局 list/recall/query 是 channel 治理，不进协议。主路径
= `get(uri)` → `list[Message]`。每 storage 的 list/usage/help 是**可选便利**（单 host 无归并），
非承重。见 KD9。

**拒绝的替代**：wire 上传 `RESOURCE_TYPE`（任意 Python 对象）。拒绝原因：那样每个 scheme 都得
自带序列化约定，等于每 scheme 一个 zenoh 子协议。scheme 之间的差异应全部留在 owning cell 进程
内的 `ResourceStorage` 实现里，不渗透到传输层。

**推论**：`RESOURCE_TYPE` 泛型退化为**进程内便利**。`contracts/resource.py` 其实混着两个面：
- **认知面**（跨网络）：`ResourceInfo` / `as_messages` / `get as messages`
- **类型面**（仅进程内）：`ResourceItem[INFO, TYPE]` / `get_by_item_type()` —— 网络对面永远拿不到你的 Python 类型

**默认渲染定稿**：协议要求每个网络 storage 能把 item 渲染成 `list[Message]`。`ResourceItem`
补认知面出口 `as_messages() -> list[Message]`（默认实现 = 用 `ResourceInfo.as_content()` 的
meta JSON 包一个 text Message）。**默认是兜底，不是目标**：text 类 storage 必须覆写——meta JSON
对模型没用，模型要的是文本本身。`ResourceInfo.as_content()` 已是 meta 侧先例，item 侧补上就对称。

**克制纪律**：不发明新 message 协议体系。`Message`（tag/attributes/嵌套 xml/addition）已够表达
"这是一个资源回复"，顶多约定几个 tag。信封越无聊，scheme 实现者自由度越大。

### 2. manifest 静态 / provide 动态 — 统一寻址下的两分法

**选择**：
- **静态资源**（manifests 注册）→ 保持**任意 host**（如 `moss-howto`），只在本进程解析
  （没 declare queryable，其它 cell 拿不到——"静态不上网"以"没提供"表达，不需要特殊寻址）。
  现状 `InMemoryResourcesRegistry` 的注册路径不动。
- **动态资源**（"组网才存在"）→ cell 调 `provide_resource(storage)` 一步入网，host = 自己
  fullname。storage 内存里怎么 put/delete 都是 cell 私事，网络只看它 announce 的只读 queryable。

**关键性质**：scheme 天生自带通讯协议——因为协议统一（Message 进出），scheme 差异全留在 cell
进程内。cell 想暴露写动作，用自己的 channel command（如 `my_blogs.delete(...)`），不经 resource 协议。

### 3. 解析器 = 本地先查 + 网络 miss（自过滤删除）

**选择**：`matrix.resources` 的返回替换为**解析器**：本地 registry 按 `(scheme, host)` 先查
（命中即本地），miss 走 zenoh get 到 `{ns}/resources/messages/{scheme}/{host}`。

**为什么自过滤删除**：本地 registry **只装 self 拥有的**（manifest 静态 + 自己 provide 的），
远程的**永不物化**进本地——归并问题（"远程 scheme 与本地 scheme 归并"）从结构上消失。自己
provide 的资源只经网络出现一次，没有重复，不需要 self-filter。

**provide_resource 双注册是对的且必要**：注册进 self 本地 registry（host==self 时进程内解析）
+ declare queryable（host!=self 时经 zenoh 解析）。同一个 storage 的两个出口，不冲突。远程
storage 从不进本地，所以没有归并冲突。

**发现机制**：网络侧是 zenoh get。liveness 免费：cell 死了 queryable 自动消失，它的资源就查不到——
资源可用性 = queryable 存在性，与 presence 同一物理机制。悬空 URI get 返回 None。

### 4. key 布局 — get 面为主，path 进 payload

网络 key（2026-08-05 收敛后）：

```
{ns}/resources/messages/{scheme}/{host}    认知面 — 给模型的 get（主）
{ns}/resources/data/{scheme}/{host}        传输面 — 给代码的 get（可选）
{ns}/resources/meta/{scheme}/{host}        便利面 — 每 storage 的 list/usage/help（可选）
```

| 面 | 消费者 | 回复形态 | 承载操作 |
|----|--------|---------|---------|
| messages | Ghost context | `list[Message]` | get（op=get, args={path}） |
| data | Remote proxy / 代码 | bytes + content_type（或引用） | 代码 get |
| meta | resources channel 便利 | JSON: scheme_description, usage, served_by | 单 storage list / usage / help |

- **path 进 payload，不进 key**（修旧 KD4 字面 key 布局与 KD1 "selector 只承担 key 路由" 原则的
  自相矛盾）。queryable 一律声明**具体 key**（host 级），get 侧无需通配 queryable——与已验证模式
  一致（`tests/ghoshell_moss/matrix/test_zenoh.py` 金丝雀：具体 key 声明 + get 侧通配/具体查询）。
- **全局发现（wildcard meta get）删除**：list/recall 是 channel 治理，见 KD9。
- **data 面可选**，在 meta 的 `supports` 字段声明。纯认知资源（如 `servers://` 端点列表）不需要
  data 面。大 payload（视频）不从 zenoh 流过——`data`/`messages` 回复里放**引用**：本机文件放
  路径，跨机放 HTTP URL（`resource-http-endpoint` 在此归位）。引用逃生门长在信封内部，协议层不特判。
- 三面可独立演进：messages 面第一期即可用，data 面的通用 streaming 慢磨。

### 5. locator = scheme://{host}/path，host 位 = cell 身份（统一寻址）

**选择**：host 位语义是 **cell-address-short** = cell 的 `fullname`（category_name，稳定，
无斜杠），**不含 uid**——uid 每次 spawn 重新生成 → locator 随 cell 重启全体作废，"compact 不遗忘"
被 uid 易逝性偷走。网络资源一律 host = fullname。

**向前兼容（2026-08-05）**：现有静态资源保持任意 host（moss-howto 等），**零迁移**。本地先查
按 `(scheme, host)` 命中即本地，静态资源的 host 不需要是 cell 身份。只有新 provide 的网络资源
用 cell-address-short。破坏性改动收窄到网络资源寻址，对 moss 衍生项目零冲击。

**撞车纪律**：静态任意 host 恰等于某 cell 的 fullname → 本地先查赢（罕见，可诊断）。非 singleton
cell 提供网络资源必须自己声明不冲突的 host。跨 project 同名 cell 理论会撞 host（singleton 锁是
project 域的，network scope 可能跨 project）——第一期只记录不解决（`served_by` 带 project_id 可诊断）。

**实证支撑**：mesh channel 的 virtual_children alias 已经用 `cell.fullname`
（`channels/matrix_channel.py` `_refresh`），注释写着"未来场景倒逼时可加 uid 后缀去冲突"。
resource 投影用同一策略，与已运行的网络投影体系同构。

### 6. ABC 保留 put/delete；网络只投影 get；写走 channel command

**选择**：`ResourceStorage` ABC **完整保留** put/delete——那是 cell 本进程内的存储能力全集。
网络膜上只投影 get。写动作走各 cell 自己的 channel command。

**为什么写走 channel 不是权宜**：变更需要顺序保证（channel 内 command 有序）、需要归属（谁的 blog
谁的 channel）、需要各自的参数签名（每个域的 delete/put 语义不同，硬塞统一接口就是"一切皆 PUT"的贫瘠）。

**曾走过的弯路**（记录以免重犯）：一度主张从 ABC 删掉 put/delete。错在把"对模型暴露什么"和"接口有
什么"混为一谈。三者正交：ResourceStorage 管存储全集，channel 管对模型暴露的写子集，网络膜管对网络
暴露的读投影。删接口等于强迫实现者用非契约方法做 put，更乱。

### 7. 归属：读侧 = resolver（registry 替换），provide 侧 = CellPresence

**选择**（2026-08-05 定稿，取代旧 KD7 的模糊化）：
- **读侧**：`MatrixImpl.resources` 换成/包一层解析器（本地先查 + 网络 miss，见 KD3）。本地
  registry 职责从 "project 专用" 扩为 "self 拥有的全部"（静态 + 自己 provide 的）。现有 manifests
  注册路径（`ResourceStorageFactoryBootstrapper`）不动。
- **provide 侧**：`CellPresence` 加 `provide_resource`（matrix 提供对称糖，见 KD8）。
- 读取侧不再挂 `CellNetwork`——不需要；解析器就是 registry 本身。

**命名背景**（beta1 窗口）：原 `CellMesh` 已改名为 `CellNetwork`。`mesh` 一词封存给未来"微服务式
cell 间通讯基底"。Matrix = 网络的投影（去中心，host 拿到的 matrix"一直在变大"）；Network 是其中
"对等 cell 发现 + 连接"的一个切面，同时兜住 discovery + connection 两个语义。

### 8. provide_resource 契约 — cell 级通用声明机制

**选择**：`async def provide_resource(self, storage: ResourceStorage) -> None`，挂在 `CellPresence`
上，matrix 提供对称糖（`matrix.provide_resource` 委托 presence）。

- **无句柄**：storage 已活在进程内，不需要 task 治理（不像 provide_channel 要返回 provider 跑
  `arun_until_closed`）。注册即生效。
- **一 cell 可 provide 多个 storage**（每个 storage 调一次）。
- **纯声明，两个效果**：（a）注册进 self 本地 registry（host==self 解析），（b）按协议 declare
  queryable（messages 必须，meta/data 按 storage 能力可选）。
- **生命周期绑 presence** `__aenter__`/`__aexit__`，与 provide_channel 同规则（未 enter 调
  raise）。

**纪律**："matrix 不需要准备很多个声明机制，它需要的是一个 cell 级别的通用声明机制。" scheme 的
差异全留在 storage 实现内，provide_resource 只是把 storage 抬上网络膜。

### 9. 操作分层 — 不提供全局查询+搜索

**选择**（2026-08-05）：协议只做 **get(uri)**。全局 list/recall/query 全部 channel 治理：
- **list**：跨 cell 聚合 = 多路归并，排序无解。不做全局 list；单 storage 的 list/usage/help 是
  可选便利，非承重。
- **recall**：无统一 recall 协议，归并本来需要 agent。推给 channel（各 cell 自己的搜索命令）。
- **query**：是 concrete 自定义逻辑。推给 channel。

**资源进上下文的门槛**：被 get + 描述正确（meta 够好）+ ghost 可理解，才成为上下文变量。否则它
只是 cell 运行时的动态计算，不占 ghost 历史。这同时是 compact 不遗忘的机制。

**克制纪律**：与 KD1 "信封无聊" 并列的第二条纪律——"不提供全局的查询+搜索"。协议越薄，实现越稳。
模型发现资源的路径是：command 返回 URI 字符串 → 已知 URI get / 或经 cell 的 channel 搜索得到 URI。

### 10. URI 是跨工具货币，纯字符串（呈现层分离）

**选择**：URI（`scheme://host/path`）是**纯字符串**，不包壳（`RESOURCE(...)` 之类文字包装污染
货币）。任何 command 可返回 URI，支持资源的 command 可接受 URI。`matrix.resources`（解析器）是
唯一解析器——任何 cell 拿到 URI 都能解析（host 位是 cell 身份，在 network scope 内全局有意义）。

- **跨 scope 未定义**：URI 只在 network scope 内有意义。
- **呈现层与协议层分离**：历史里作为 context 变量持有时，由呈现机制让模型认出它（Message
  `tag="resource"` + attributes 挂 locator/meta，或新增 Resource Content 类型）。这是可选打磨，
  不是承重墙——ghost 完全看不到字符串也是可行的，因为 deref 是命令干的，ghost 只是搬运字符串。
  **open item**：content-type 是否新增，呈现层怎么落（见 Implementation Notes）。

### 11. 向前兼容 — 核心改动 = registry 替换 + 三块薄件

**选择**：整个重构的核心是**替换/包装 in-memory registry 为解析器**（本地先查 + 网络 miss）。
现有静态资源（任意 host）零迁移，衍生项目零冲击。围绕核心还有三块薄件：
- (a) `provide_resource` API（声明 queryable + 注册本地）
- (b) 网络读侧（zenoh queryable 声明 + 客户端 get）
- (c) 渲染契约（item → list[Message]，默认 meta JSON 兜底，现有 storage 不改也能 get）

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->

- **查询参数走 payload 而非 zenoh selector**：`path`（get）、list 的 query/limit 放 JSON payload，
  selector 只承担 key 路由。信封无聊原则。（实现细节，可调整。）
- **第一 milestone = text resource**：带 meta 的 text 块，跨场景获取。cell 提供 notes storage →
  模型拿到 `notes://{fullname}/{path}` URI → 跨上下文 get → 返回 Message。大 payload / http
  引用逃生门后置。
- **金丝雀测试（实现前必做）**：zenoh queryable 的线程模型（单线程 vs 线程池）+ **deferred
  reply** 安全性——messages/data 面在 queryable 回调里要做 async 渲染（`storage.get` →
  `as_messages`），现有 queryable 回调全是同步的。text-first 下可先用 sync-over-async 桥
  （ThreadSafeFuture，storage 廉价），通用解是 handler 只 enqueue + event-loop consumer 渲染后
  调 `query.reply`（qa-exchange 的 janus 队列模式）。两条路都要一条金丝雀确认。
- **resources 投影不产 signal**：资源出现/消失是低频、模型主动查询的，不像 CellEvent 需打断
  注意力。resources channel 保持纯 read-only。
- **channel 组装**：`matrix` channel 不改名，新增 `resources` 子 channel 与 nodes/mesh 平级。
  instruction 教模型 URI 约定 + 什么时候 get / 调 cell 的搜索命令。先挂 `matrix` 下，若实测模型
  总找不到再提升为一级 channel——可逆的组装选择，不预先赌。
- **content-type open item**：历史里持有一个资源时怎么呈现——Message tag 约定 vs 新增 Resource
  Content 类型。呈现层决策，不碰协议，可后定。
- **跨机 HTTP 引用可达性 open item**：resource-http-endpoint 默认绑 127.0.0.1，owner 组 reply
  时不知道 requester 在哪台机器。后置但显式记账，别 silent todo。
- **待清理**：`contracts/resource.py` 模块 docstring 还留着"验证版…验证通过后覆盖回…"的草稿头，
  它已是 contracts 本体，随本 feature 或改名轮一起清（优先级低，不着急）。

## Operator 落地映射 (2026-08-13)

matrix-operator（2026-08-09 completed → 08-13 reopened 修内核问题）提供了资源落地的 wire 层。
本 feature 仍 design-locked、零实现；**实现前必须先修完 operator 的 kernel review 清单**，
见 matrix-operator FEATURE.md。

### 前置确认：异步 queryable 已解决

FEATURE 原要求"实现前必做"的金丝雀（zenoh queryable 线程模型 + deferred reply）已被 operator
K2 桥内建：zenoh 回调 `_on_query` 只 enqueue → asyncio 消费任务 `await handler(q)` →
`asyncio.to_thread(query.reply, ...)`。resource handler 写成 `async def` 即可，无需 RPC 式协议。
实证（2026-08-13）：zenoh-python 回调跑在 `pyo3-closure` 线程（跨 closure 并行），operator 的
janus 队列是 pyo3→loop 的适配器。金丝雀仍要补一条锁死 deferred reply（慢 handler + 并发）。

### 键布局：KD4 被 operator 键推导取代

KD4 的 `{ns}/resources/messages/{scheme}/{host}`（host 进 key）被 operator 键推导取代：
`{services_ns}/{cell_addr}/{kind}/query/{business_key}`。host 退化为 service 发现维度
（按 `meta.data.host` 匹配），scheme 进 business key，path 进 payload。

### D1 三 queryable（每 scheme）

business key = `{scheme}/{face}`，face ∈ {meta, messages, data}；op (read|list) 在 payload：

```
{services_ns}/{addr}/resource/query/{scheme}/meta       payload: {op} → usage/help
{services_ns}/{addr}/resource/query/{scheme}/messages   payload: {op: read|list, path?, query?, limit?} → list[Message]（主）
{services_ns}/{addr}/resource/query/{scheme}/data       payload: {op: read|list, path?} → bytes + content_type（可选）
```

open：op 进 payload（3 queryable/scheme，押这个，协议更薄）还是进 key（6 个/scheme）。

### host → service 解析

URI `scheme://{host}/path`，host = cell fullname（无 uid，稳定）。解析器：
本地 registry 按 `(scheme, host)` 先查，命中即本地；miss 走 operator——
`get_services_by_kind("resource")` 按 `meta.data.host == uri_host` 匹配 →
`operator.get("resource", "{scheme}/messages", {op:"read", path})`。悬空 host get → []。

### provide 侧 = Matrix（不走 CellPresence）

走 presence 时 operator 不存在。`matrix.provide_resource(storage)`：
(a) 注册进 self 本地 registry（host==self 进程内解析）；
(b) 懒创建 cell 级 resource terminal（kind="resource" 单例），每 storage 加
`queryable("{scheme}/messages")` 等。
open：是否 publish_event——建议不发（service liveness 已覆盖上线/下线，CellEvent 更新
presence 载荷但不含 resource 信息，空通知）。

### 解析器 = 包装本地 registry

`MatrixImpl.resources` 保持 `ResourceRegistry` 表面（register/schemes/hosts 委托内层，
bootstrapper 与 inspector 零改动），新增认知入口 `async get_messages(locator) -> list[Message]`：
本地 `item.as_messages()`，远程 operator get 反序列化。`get(locator) -> ResourceItem` 保持本地语义。

### 依赖顺序

1. operator kernel 修复（create_task + 容错 + shutdown，见 matrix-operator FEATURE.md）
2. `contracts/resource.py` 补 `ResourceItem.as_messages()`（KD1 认知面出口）
3. `matrix/services/resource/`（ResourceDeclaration / ResourceServer）
4. 读侧解析器 + `provide_resource`
5. 金丝雀：跨 cell text get（双 ZenohOperator 单 session）
6. milestone 1 = text resource
