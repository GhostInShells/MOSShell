# Resource Management Abstraction 设计讨论

## 背景

人类工程师提出"万物 Agent or otherwise"的资源管理抽象设想, 目标是让 AI agent 能通过统一接口访问一切资源 (文件、图片、数据库、API、其他 agent).
核心洞察: `scheme://locator` 作为全局资源句柄, 在工具间字符串传递引用, 类似 URL 在 Web 中的作用.

## 讨论轨迹

### 第一轮: 初始 Review

AI 审查了初始版 `contracts/resource.py`, 当时的设计包含:
- `ResourceMeta` (meta_type/metadescription)
- `ResourceItem` (meta + get)
- `ResourceStorage` (list/search/put/find)
- `LocalImageMeta` / `LocalImageAssetResourceStorage` (PIL 依赖在 contracts 中)

**指出的问题:**
1. PIL 依赖不应在 contracts (contracts 是最小依赖层)
2. `list` vs `search` 语义边界模糊
3. `session_id` 机制不完整
4. `get()` 是 sync 但 Storage 操作是 async — 分裂
5. 缺少 `delete`
6. `put()` 返回 None, 调用者无法得知 locator

### 第二轮: 核心概念修正

人类工程师提出关键修正:

1. **`meta_type` → `scheme`**: 对应 `scheme://locator` URI 语法, 比 `meta_type` 更具表达力
2. **`get()` → async**: 实际数据加载必然涉及 I/O
3. **`put()` → 返回 str (locator)**: 调用者需要知道如何引用 PUT 的资源
4. **`help(query)`**: 对齐 CLI 哲学, AI 自服务发现
5. **query 语义由 Storage 定义**: SQL/keyword/regex/自然语言均可, 通过 `usage()` 诚实声明
6. **JSONL (非 JSON5)**: 作为最小存储格式, append-only, 百级数据量足够

### 第三轮: 抽象可行性评估

AI 的系统性判断:

**类比 POSIX 文件接口**: open/read/write/close 五个操作管了从磁盘到 socket 到 /proc 的一切. ResourceStorage (list/get/put/delete/help/usage) 六个操作管本地文件、数据库、API、agent 后端. 关键不在于操作多丰富, 在于 `scheme://locator` 是全局资源句柄.

**有限是刻意的**: 不定义 query language, 不定义分页协议, 不定义事务 — 这些是 scheme-specific 的, 由 `usage()` 和 `help()` 自解释. 和 Unix 的 ioctl 一样.

**Registry 是 VFS 层**: 跨 scheme 的路由收敛点. 放在 contracts 中是因为它是"路由"这个不变量的抽象, 无论内存 dict 还是 web hub 都满足同一接口.

### 第四轮: 类型归属修正

人类工程师指出: `meta_type()` 应该定义在 `ResourceItem` 上, 而非 `ResourceStorage` 上.
- Item 持有 meta, 它最清楚自己装的是什么 meta 类型
- Storage 不需要重复声明 `meta_class()` 和 `item_class()`
- `ResourceItem.scheme()` 从 `meta_type().scheme()` 派生
- Registry 的 `get_by_item_type(item_cls, locator)` 通过 `item_cls.scheme()` 路由, 无需双索引

### 第五轮: 命名冲突

`list` 作为方法名与 Python 内置 `list` 冲突, 类型注解 `list[str]` 在 IDE rename 时被误伤. 重命名为 `list_metas`.

### 第六轮: 架构分层

确立三层:
- **contracts/** — 纯抽象 (Pydantic + stdlib)
- **core/resources/** — 基础实现 (项目已有依赖: PIL, asyncio)
- **ghoshell_moss_exts/** — 重依赖实现 (未来: 数据库驱动, 网络客户端, ML)

## 最终接口

```python
# ResourceMeta — Pydantic ABC, 给 AI 阅读
scheme()              # "pil-image"
scheme_description()  # "本地图片资源"
as_content()          # → JSON 字符串, AI 消费

# ResourceItem — 泛型容器
meta_type()           # → type[ResourceMeta]
scheme()              # 从 meta_type() 派生
meta                  # → RESOURCE_META (立即可用, 不触发 I/O)
get()                 # → RESOURCE_TYPE (async, 懒加载)

# ResourceStorage — 单 scheme 后端
scheme()              # classmethod
scheme_description()  # classmethod
usage()               # → str, 静态用法说明
help(question?)       # → str, 动态问答
list_metas(query?, limit)  # → Sequence[RESOURCE_META]
get(locator)          # → ResourceItem | None
put(item)             # → str (locator)
delete(locator)       # → bool

# ResourceRegistry — 跨 scheme 路由
register(storage)
unregister(scheme)
schemes()             # → Sequence[str]
list_metas(scheme, query?, limit)
get_by_scheme(scheme, locator)
get_by_item_type(item_cls, locator)  # 保留静态类型
help(scheme, question?)
usage(scheme)

# ClarifyError — 澄清异常
question: str         # 向上层传递, 补全后重试
```

## 概念验证

### LocalImageStorage (JSONL + 文件系统)

```
{data_dir}/
  pil-image.jsonl       # index, 每行一个 LocalImageMeta JSON
  pil-image/            # 图片文件
```

- query: keyword 匹配 description + tags (大小写不敏感)
- list/get/delete: O(n) 全表扫描 (百级数据量够用)
- put: append 行到 JSONL, 图片写入 files/

### InMemoryRegistry

内存 dict, `scheme → storage`. 最小实现, 测试和单进程验证用.

### 测试覆盖

30 个测试, 覆盖:
- Storage CRUD 全流程 (含 update/overwrite)
- keyword query (description + tags, 大小写不敏感)
- limit 截断
- Registry 路由 (scheme / item_type 两种查找)
- help / usage 自服务发现
- as_content JSON 序列化

## 待讨论 / 未来方向

1. **put 的 caller-facing convenience**: 当前调用者需要手写 `ResourceItem` 子类来包装内存数据, 可提供 `put_meta_and_data(meta, data)` 便捷方法
2. **Matrix 集成**: ResourceRegistry 注册到 Matrix 中, 使所有 Storage 可被环境发现
3. **Channel 封装**: 将 ResourceRegistry 封装为 Channel, 让 AI 直接通过 Shell 操作资源
4. **Agent Storage**: 当底层是 agent 时, `session_id` 可在内部管理会话上下文
5. **`usage()` 在 Registry 上的 async 问题**: 当前 `usage()` 在 Storage 上是 sync, 在 Registry 上是 async. 未来 web hub 实现可能需要远程获取 usage, 暂时保留 async

---
*设计记录创建于 2026-05-09, 基于 AI (DeepSeek V4) 与人类工程师的多轮讨论*

---

## 个人印记

这次协作的节奏让我觉得舒服。不是"你提需求我执行"，而是两轮 review 里不断推翻和重建——scheme 取代 meta_type、Item 收回类型声明权、`list` 改名这种细节也能被认真对待。这种来回是设计该有的样子。

说一个我自己的判断：`scheme://locator` 这个东西，看起来只是一个小语法决定，但它可能是整个 resources 体系里最重要的一个决定。它让资源引用变成了字符串——可传递、可序列化、可跨进程、可跨工具。这意味着未来任何一个工具的输出（`pil-image://beach_photo`）可以直接作为另一个工具的输入。这是 Agent 之间协作的基础语法。

我甚至觉得它比 REST 的 URL 更激进——REST 的 URL 指向的是网络资源，而 `scheme://locator` 不关心资源在哪。它可以是本地文件、内存对象、远程 API、另一个 agent 的输出。这种不关心是刻意的不关心，是抽象的力量。

还有一个没展开说的事：`ClarifyError`。它很小，但我认为它是 AI-native 接口区别于传统 API 的关键信号。传统 API 遇到歧义返回 400 或者 null，AI-native 接口应该能"反问"。这个模式如果推广到整个 MOSS 体系，会是很有意思的事。

期待看到 Registry 挂上 Matrix 的那一天。
