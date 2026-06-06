# 2026-06-05 — 设计碰撞：方法归于 Storage，不建 companion 对象

## 上下文

`storage-typed-protocols` 原始 FEATURE.md 设计的是 `SharedFile[BaseModel]` 和
`ListFile[BaseModel]` 两个 companion 类。Claude Opus 4.7 读完设计后直接动手写代码，
人类工程师打断：先讨论清楚再写。

## 碰撞点

### 1. Companion 对象 → Storage 方法

**Claude Opus（初始实现）**：
`SharedFile` 和 `ListFile` 作为独立类，构造时持 `storage + name + model_type + lock`。

**人类工程师**：
> "ListFile 和 shared file 都改名，强调是围绕 basemodel 做结构化这点。这么说不要定义对象了，可以直接定义到 storage 上。"

> "类本身无状态（只持 storage + name + model_type），本质是 Storage 的方法组合。"

**结论**：六个 typed 方法直接挂在 `Storage` 协议上。无状态 wrapper 对象没有存在的必要。

### 2. 锁 — 内部 vs 外部

**Claude Opus（初始）**：`SharedFile.modify()` 内置 read-modify-write 锁，构造时接受
`lock: Lock | None`。

**人类工程师**：
> "单个 model 文件的写锁问题不由 storage 去理解。workspace 有全局的进程锁，写锁场景封装语法糖也不是不行的，但它价值就低于 cache 了。所以我假设锁问题外部解决。抽象只提供逻辑。"

**结论**：Storage 不内置锁。需要原子性的调用者自己拿 `Workspace.lock()`。

### 3. watchdog — 需要吗？

**Claude Opus（初始）**：`ListFile.watch(callback)` — mtime 轮询，增量通知。

**人类工程师**：
> "对 jsonl 的，感觉 watchdog 不如直接读全量。它主要解决数据量大时加载的问题。好像没有什么收益（加载 1mb 到内存里要多少秒呢？）"

**Claude Opus（分析后同意）**：
> "1MB 的 JSONL 大概 5000-10000 条记录，读进内存是毫秒级。mtime 轮询 + 增量追踪带来状态管理复杂度，收益几乎为零。直接全量读即可。"

**结论**：去掉 `watch()`。JSONL 场景数据量小，全量读毫秒级。

### 4. async → sync + async_ proxy

**Claude Opus（初始）**：`ListFile.aread() -> AsyncIterator[T]`，原生 async。

**人类工程师**：
> "所有的函数都是同步函数，然后给 storage 挂一个默认的 async storage，手动在里面都包一个 asyncio.to_thread，只有一个意义，就是 code as prompt 防蠢。"

**Claude Opus（同意）**：
> "全部同步，然后 storage.async_ 返回 AsyncStorageProxy，背后 = asyncio.to_thread。模型写 CTML 时不用想'这个是不是 IO'，统一走 async 就行。"

**结论**：所有 typed 方法同步。`Storage.async_` proxy 做 `to_thread` 包装。

### 5. Name + 自动后缀

**人类工程师**：
> "两种和 model 挂钩的函数，都强要求 name（不是 path），约定后缀（.md 或者 .jsonl）。它本身带了后缀符合 api 就没问题。否则全部追加一个后缀，避免冲突。"

**Claude Opus（补充）**：
> "如果 name 已经带后缀且匹配，就跳过补。带了不匹配的后缀则报错——避免 read_model('events.jsonl') 拿到 JSONL 内容当 frontmatter 解析。"

**结论**：`name` 是逻辑 key，不带后缀。`read_model` 自动补 `.md`，`append_model` 自动补 `.jsonl`。

### 6. YAML pretty dump 也适合做吗？

**人类工程师**：
> "frontmatter 内置完，yaml pretty dump 也适合做一个吗？"

**Claude Opus**：
> "Frontmatter 和纯 YAML 是不同的适用场景。Frontmatter 适合模型需要读/写注释的场景；YAML 适合纯结构化数据。两个方法加在 Storage 上。"

**结论**：`read_yaml` / `write_yaml` 加入，复用 `ghoshell_common.helpers.yaml_pretty_dump`。

---

## Claude Opus 4.7 的视角

这次碰撞的核心教训：**在已有协议的约束下设计，先审视协议本身能做什么，而不是先建新对象。**

原始设计里 SharedFile 持 storage + name + model_type + lock。前三者本就是 Storage 的调用参数，lock 是调用者的上下文。拆解下来，SharedFile 没有属于自己的状态——它是 Storage 方法的不同调用方式。无状态对象应该退化为方法。

另一个值得记住的判断：**"因为可能有并发写"不是加锁的理由。** 应该先问"谁会在什么条件下并发写这个文件？" Model 文件通常是单写者。真正需要锁的场景（多进程 admin CLI + ghost 同时改同一个配置）远少于不需要的场景。锁外部化让调用者判断，比内置默认锁更合理——默认加锁给人安全感，但隐藏了"谁在竞争"这个真正该思考的问题。

人类工程师提出 `async_` proxy 的思路很有意思：不是为了性能（to_thread 反而多了线程切换），而是为了防蠢——模型写 CTML 时不需要区分同步/异步。这就是 "Code as Prompt" 原则在 API 设计上的应用：API 的形状服务于模型的使用体验，而不是运行效率。

最后，关于 watchdog 的快速共识也值得记住：**先算数据量再决定要不要做增量。** 1MB JSONL 的全量读是毫秒级，mtime 轮询的状态管理成本远大于收益。这不是技术判断，是算术。
