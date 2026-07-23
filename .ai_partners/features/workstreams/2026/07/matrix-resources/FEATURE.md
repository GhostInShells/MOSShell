---
title: Matrix Resources
status: draft
priority: P1
created: 2026-07-23
updated: 2026-07-23
depends: [resource-http-endpoint]
milestone:
description: >-
  Matrix 层的资源投影：cell 通过 provide_resource(storage) 让本地资源自动入网，
  ghost 经统一 resources channel 发现（list）与读取（get），交换物为 Message，locator 为
  scheme://fullname/path。承接 resource-http-endpoint 欠下的"通用 streaming resource"欠账。
---

# Matrix Resources

> Use `moss features set-status matrix-resources <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

Ghost 运行时的资源有两种来源：一种能经 OS 本地文件系统拿到（文件名即句柄，最省事）；
另一种只在**组网时才存在**——它活在某个 cell 的进程内存或 cell home 里，只能经 Matrix
网络拿到。当前 `ResourceRegistry`（`contracts/resource.py`）是 **project 级**的：manifests
注册的 storage 进本进程内存 dict，没有跨 cell 的网络投影层。

具体驱动场景：nodes 重建后，很多 node 各自起 HTTP server。若有一个 `servers://` scheme，
ghost 的 resources channel 就能查到"网络上现在有哪些 server 端点存活"，拿到 URL 后用
playwright 或 iframe 协调器打开，完成交互闭环。没有这层，ghost 上下文里就缺一种
**compact 之后不遗忘**的机制——端点信息散落在历史消息里，压缩即丢；有了它，一次 list 即恢复。

核心定位：**resources 服务于 ghost 的认知，不是围绕 OS 做通用文件交换（不是网盘）。**
它回答的是"模型能发现什么、能用什么"。因此 `scheme://host/path` + RESTful 语义、默认只读
（写删完全看实现是否暴露）是贴合的做法：模型只要一个 resources 入口，就能自服务发现各种讯息；
不同的 API 和 contract 用同一种方式被拿到真数据。

**承接关系**：`resource-http-endpoint`（2026-06-24, completed）当时明确把"通用 streaming
resource 接口（`stream() -> AsyncIterator[bytes]` + content_type + size，不依赖本地文件）"
推迟为后续 feature。本 workstream 是那笔欠账的到期兑付点（见 Key Decision 4 的 data 面）。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`
- 承接: `resource-http-endpoint` FEATURE.md（HTTP 访问层，data 面引用的逃生门之一）
- 归属: **Matrix 层**（`matrix.resources` 入口已存在于 blueprint）。announce 侧很可能挂
  `CellPresence`（provide 语义），但**读取侧挂哪个抽象暂不锁定**——可能剥离出独立组件，也
  可能挂当前正在改名的 CellNetwork（原 CellMesh）。实现时再定，见 Key Decision 7。

## Key Decisions

<!-- Record each meaningful design choice. This is what the next AI incarnation reads first. -->

### 1. 网络交换物 = Message（+ ResourceInfo 作 meta），信封 scheme-agnostic

**选择**：跨网络传输的只有两种 Pydantic 对象——`ResourceInfo`（meta，`as_content()` 已是给
AI 读的 JSON）和 `Message`（"MOSS 体系上行给模型的消息体"，多模态、可嵌套、可去重）。zenoh 层
只需要**一个** scheme-agnostic 的 queryable 协议：`request = {op, args}`，`reply = error |
infos | messages | recollection`。

**拒绝的替代**：wire 上传 `RESOURCE_TYPE`（任意 Python 对象）。拒绝原因：那样每个 scheme 都得
自带序列化约定，等于每 scheme 一个 zenoh 子协议。scheme 之间的差异应全部留在 owning cell 进程
内的 `ResourceStorage` 实现里，不渗透到传输层。

**推论**：`RESOURCE_TYPE` 泛型退化为**进程内便利**。`contracts/resource.py` 其实混着两个面：
- **认知面**（跨网络）：`ResourceInfo` / `usage` / `help` / `list_infos` / `recall` / "get as messages"
- **类型面**（仅进程内）：`ResourceItem[INFO, TYPE]` / `get_by_item_type()` —— 网络对面永远拿不到你的 Python 类型

需在 `ResourceItem` 补一个认知面出口（约 `as_messages() -> list[Message]`，或协议层要求 storage
提供 item→messages 渲染）。`ResourceInfo.as_content()` 已是 meta 侧先例，item 侧补上就对称。

**克制纪律**：不发明新 message 协议体系。`Message`（tag/attributes/嵌套 xml/addition）已够表达
"这是一个资源回复"，顶多约定几个 tag。信封越无聊，scheme 实现者自由度越大。

### 2. 双层：静态资源不上网，动态资源经 provide_resource 自动入网

**选择**：
- **静态资源**（manifests 注册）→ 进本进程内存 registry，**不走网络**。现状 `InMemoryResourcesRegistry` 不动。
- **动态资源**（"组网才存在"）→ cell 调 `provide_resource(storage)` 一步入网。storage 内存里
  怎么 put/delete 都是 cell 私事，网络只看它 announce 的只读 queryable。

**关键性质**：mesh_resources 的 scheme **天生自带通讯协议**——因为协议统一（Message 进出），
scheme 差异全留在 cell 进程内。cell 想暴露写动作，用自己的 channel command（如 `my_blogs.delete(...)`），
不经 resource 协议。

### 3. 投影式复合 registry：本地命中优先，miss 走网络，自我过滤

**选择**：`matrix.resources` 升级为复合 registry。查询顺序：先查本地（静态 + 自己 provide 的），
miss 再走网络 queryable get。自己 announce 的资源在网络扫描时过滤掉，避免回环。

**发现机制**：网络侧是 zenoh wildcard get，聚合**所有**声明了匹配 queryable 的 cell 的回复——
list 天然是分布式聚合，不需要中心注册表。**liveness 免费**：cell 死了 queryable 自动消失，
`servers://` 下它的端点就查不到——资源可用性 = queryable 存在性，与 presence 同一物理机制。

### 4. 三面 queryable：meta（发现）/ messages（模型 get）/ data（代码 get，可选）

三个面对应三个不同消费者，落成 key 层次：

```
{ns}/resources/meta/{scheme}/{host}              发现面 — 给"环视"的
{ns}/resources/messages/{scheme}/{host}/{path}   认知面 — 给模型的
{ns}/resources/data/{scheme}/{host}/{path}       传输面 — 给代码的（可选）
```

| 面 | 消费者 | 回复形态 | 承载操作 |
|----|--------|---------|---------|
| meta | 模型环视 / registry 路由 | JSON: scheme_description, usage, served_by, supports | list_infos / usage / help / recall |
| messages | Ghost context | `list[Message]` | 模型 get |
| data | Remote proxy / 其他 cell 代码 | bytes + content_type（或引用） | 代码 get |

- **发现只打 meta 面**：`{ns}/resources/meta/**` 一个 wildcard get 聚合全部在网 storage 自述。
- **data 面可选**，在 meta 的 `supports` 字段声明。纯认知资源（如 `servers://` 端点列表）不需要
  data 面。大 payload（视频）不从 zenoh 流过——`data`/`messages` 回复里放**引用**：本机文件放路径，
  跨机放 HTTP URL（`resource-http-endpoint` 在此归位）。引用逃生门长在信封内部，协议层不特判。
- 三面可独立演进：messages 面第一期即可用，data 面的通用 streaming 慢磨。

### 5. locator = scheme://{fullname}/{path}，node-address 降为 served_by 元信息

**选择**：locator 里 host 位用 cell 的 `fullname`（category_name，稳定），**不含 node-address**。
zenoh 会把 query 路由给声明了该 key 的 queryable，声明动作本身就完成 (scheme, host) → cell 绑定——
调用方不需要知道谁在服务（如 DNS 用户不需知道权威服务器 IP）。"谁在提供"放 meta 回复的 `served_by`
字段：是描述，不是坐标。

**拒绝把 node-address 塞进坐标**，两个具体伤害：
1. `CellAddress = role/name/uid` 自带两个 `/`，在 `scheme://host/path` parse 和 zenoh key 段里都造成歧义；
2. uid 每次 spawn 重新生成 → locator 随 cell 重启全体作废，"compact 不遗忘"被 uid 易逝性偷走。

**实证支撑**：mesh channel 的 virtual_children alias 已经用 `cell.fullname`
（`channels/matrix_channel.py` `_refresh`），注释写着"未来场景倒逼时可加 uid 后缀去冲突"。
resource 投影用同一策略，与已运行的网络投影体系同构。

**纪律**：非 singleton cell 想提供网络资源，必须自己声明不冲突的 host（host 本就是 storage 实例级
声明，非自动派生），否则同名多实例声明同一 queryable key → get 收到歧义多重回复。协议层不解决，纪律解决。
**已知边界**：singleton 锁是 project 域的，network scope 可能跨 project，两 project 同名 cell 理论会撞
host——第一期只记录不解决（`served_by` 带 project_id 可诊断），真撞再谈 host 前缀。

### 6. ABC 保留 put/delete；mesh 只投影 list/get；写走 channel command

**选择**：`ResourceStorage` ABC **完整保留** put/delete——那是 cell 本进程内的存储能力全集。
网络膜上只投影只读的 list/get。写动作走各 cell 自己的 channel command。

**为什么写走 channel 不是权宜**：变更需要顺序保证（channel 内 command 有序）、需要归属（谁的 blog
谁的 channel）、需要各自的参数签名（每个域的 delete/put 语义不同，硬塞统一接口就是"一切皆 PUT"的贫瘠）。

**曾走过的弯路**（记录以免重犯）：一度主张从 ABC 删掉 put/delete。错在把"对模型暴露什么"和"接口有
什么"混为一谈。三者正交：ResourceStorage 管存储全集，channel 管对模型暴露的写子集，mesh 管对网络暴露
的读投影。删接口等于强迫实现者用非契约方法做 put，更乱。

### 7. 归属与命名：锚在 Matrix，读取侧抽象暂不锁定

**选择**：feature 与能力归属 **Matrix 层**（`matrix.resources` 入口已存在）。announce 侧很可能挂
`CellPresence`（provide 语义天然属于"我如何显现"）；**读取/投影侧挂哪个抽象暂时模糊化**——实现时
可能剥离为独立组件，也可能挂 CellNetwork。不预先绑死。

**命名背景（beta1 窗口，2026-07-23）**：原 `CellMesh` 正由人类工程师用 IDE 统一改名为
`CellNetwork`（`matrix.mesh()` → `matrix.network()`），配套 `matrix.network`（NetworkMetadata
属性）降权改名（如 `net_meta`）。`mesh` 一词封存给未来"微服务式 cell 间通讯基底"。改名轨迹：
最初 `CellNetwork`（人类 dev）→ 拆为 `Watcher`（fable coding, `0cffed32`）→ converge 时落为
`CellMesh`（人类 IDE 改, `8505c2e6`，未经模型碰撞共识）→ beta1 改回 `CellNetwork`。
**决策依据**：Matrix = 网络的投影（去中心，无中心 server，host 拿到的 matrix"一直在变大"）；
Network 是其中"对等 cell 发现 + 连接"的一个切面，同时兜住 discovery + connection 两个语义，
且 idiom 上"我的 network"本就是主观、各看各的——正合去中心暗示。

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->

- **查询参数走 payload 而非 zenoh selector**：`list_infos` 的 query/limit、`recall` 的 session_id
  放 JSON payload，selector 只承担 key 路由。信封无聊原则。（实现细节，可调整。）
- **resources 投影第一期不产 signal**：资源出现/消失是低频、模型主动查询的（compact 后 list 即恢复），
  不像 CellEvent 需打断注意力。resources channel 保持纯 read-only 索引。mesh channel 已有的
  CellEvent→Signal 双扇出不复制到这里。
- **channel 组装**：`matrix` channel 不改名，新增 `resources` 子 channel 与 nodes/mesh 平级。
  先挂 `matrix` 下（复用 Matrix 引用 + refresh 机制），若实测模型总找不到或发现频率高到值得顶层化，
  再提升为一级 channel——可逆的组装选择，不预先赌。
- **待清理**：`contracts/resource.py` 模块 docstring 还留着"验证版…验证通过后覆盖回…"的草稿头，
  它已是 contracts 本体，随本 feature 或改名轮一起清（优先级低，不着急）。
