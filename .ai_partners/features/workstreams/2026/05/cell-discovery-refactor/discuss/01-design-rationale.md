# 设计讨论 — 方案选择与顾虑

上下文：2026-05-19，human 和 deepseek-v4-pro 讨论 Matrix cell 发现机制重构。

## 为什么 queryable 而不是 JSONL 文件

讨论了三种方案：
- A: JSONL 文件（零第三方依赖，单写多读）
- B: Zenoh queryable（中心化查询）
- C: Queryable + change notification（混合）

选了 B。原因：

**JSONL 方案被搁置，不是因为不可行，而是因为 zenoh 已经在通讯层存在。** 用文件做 cell 发现等于在 zenoh 旁边维护第二个通讯通道。两个通道的语义一致性（"cell 是否存活"）需要额外保证——文件 mtime、zenoh session 状态、父进程 PID 三者的关系会变成新的心智负担。

JSONL 作为**补充**有价值：运行时状态在进程外可见，`cat cells.jsonl` 即可调试。可以作为第二版的可选增强（main cell 写文件做快照，queryable 做实时查询），但不作为主方案。

**Change notification 方案被搁置**，因为当前没有真正的"持续监听"场景。唯一需要感知 main cell 存活的 app cell 已经有 `_ensure_parent_process_exists`。

## 为什么不在 Matrix ABC 和 MatrixImpl 之间加中间层

六个钩子方法是"软约定"——子类可以 override，不 override 就用默认 zenoh 实现。加 ABC 中间层（如 `AbsMatrixImpl`）需要同时维护两个类的生命周期契约，而当前只有一个实现。

YAGNI。等第二个 transport 实现出现（如 Redis pub/sub）时再提取公共抽象，那时接口边界更清晰。

## is_alive 保留但不废弃

`is_alive()` 从 abstract method 改为 concrete property，基于 `reported_at` 计算。所有现有调用方（FractalInspector、TUI、matrix_exam stub）无感知兼容。

未来迁移路径：调用方逐步改为直接读 `reported_at` 并自行判断新鲜度 → `is_alive` 标记 deprecated → 最终移除。不在一版做，因为 blast radius 已够大。

## 主要顾虑

1. **zenoh session 半关闭时的 delete**：`__aexit__` 中 async_exit_stack unwind 可能先关 zenoh session，然后才执行 `_unannounce_this_cell` 的 `session.delete()`。处理：放在 finally 块早期，try/except，失败只 log。

2. **崩溃残留 key**：app cell 意外崩溃时 key 残留。queryable handler 返回 `reported_at`，调用方自行判断新鲜度。第一版不做 TTL 自动清理——未来 main cell 可定期 wildcard get 并清理过期 key。

3. **重构成本**：约 120 行删除 + 80 行新增 + 6 个文件小改。纯内部手术，公开 API 只新增不破坏。风险主要在于 `__aenter__`/`__aexit__` 中调用顺序的变更——需要确保 `_announce_this_cell` 在 zenoh session 进入 exit_stack 之后调用。
