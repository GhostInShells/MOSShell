---
name: g1-performance
description: G1 PC2 性能画像。MOSS 运行时的资源消耗基线，idle 状态作为对比参照。
---

# G1 性能技能

## 调查命题

1. MOSS 启动后进程树结构是什么？Matrix/Cell/Channel 各占用多少进程？
2. MOSS 运行时 CPU 分布如何？单个核心 vs 多核负载？
3. 内存占用 — MOSS 进程 RSS、系统总消耗、可用余量？
4. 磁盘 IO — MOSS 的日志写入和存储操作是否产生显著的 IO 压力？
5. PC2 在 MOSS 未启动时的 idle 基线是什么？

## 架构决策

- CPU/内存余量 → 决定 PC2 是否能同时跑 MOSS + DDS + Ghost 推理，还是需要外部分散
- 磁盘 IO → 日志和存储写入是否需要限速
- idle 基线 → 后续所有性能数据的对比基准。没有基线的性能数据没有判断意义

## 执行顺序

1. **先**跑 `05_idle_baseline.sh` — MOSS 未启动时的基线
2. 启动 MOSS
3. 依次跑 `01` 到 `04`

## 脚本

| 脚本 | 输入 | 输出 | 耗时 |
|------|------|------|------|
| `01_process_tree.sh` | 无 | MOSS 相关进程树、ps 摘要 | <2s |
| `02_cpu_profile.sh` | 无 | 整体 CPU 利用率、per-core 分布、load average | <3s |
| `03_memory_profile.sh` | 无 | 系统内存概要、MOSS 进程 RSS/VSS 排序 | <2s |
| `04_disk_io.sh` | 无 | iostat 磁盘吞吐、df 存储使用 | <5s |
| `05_idle_baseline.sh` | 无 | MOSS 未启动时的 CPU/内存/IO 基线 | ~10s |
