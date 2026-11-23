---
created: 2026-11-24
depends: []
description: 为 MOSS 的子进程执行提供文件效果级隔离（read-only / workspace-write / full-access）。
milestone: null
priority: P2
status: draft
status_note: 调研完成；实现选型待人类与模型共同确认
title: Subprocess Sandbox
updated: '2026-11-24'
---

# Subprocess Sandbox

> Use `moss features set-status subprocess-sandbox <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

MOSS 的 agent 能力会让模型产生并执行 shell 命令 / 子进程。执行前必须有文件写策略约束：
命令只能读（`read-only`）、只能在 workspace 与显式可写根内写（`workspace-write`），
或完全放开（`danger-full-access`）。没有这一层，模型的任意命令等同于宿主用户权限，
连"误操作"都防不住，更谈不上让用户放心把 shell 交给 Ghost。

本工作流研究：dsh 如何做子进程 sandbox，以及 Python 侧哪些库可以复刻、复刻到什么程度。

## Design Index

- [2026-11-24 调研：dsh 的做法与 Python 可选库](design/2026-11-24-sandbox-research.md) —— 完整调研记录：
  dsh 的 argv 包装模型、sandtrap / sandlock / landlockpy 的源码级对比、复杂度评估、待定问题。
- 源码参考：`.vendors/sandtrap`、`.vendors/sandlock`、`.vendors/landlockpy`、
  `.vendors/dsh-0.2.1`、`.playground/dsh-0.2.2-rc.3`（均在 `.gitignore` 内，不随仓库分发）。

## Key Decisions

- **dsh 的模型是"薄包装 + OS 原语"。** `confine(argv, policy)` 只在命令前拼 runner 与 profile 参数，
  隔离交给 `bwrap` / `sandbox-exec` / Landlock / Windows ACL；剩余成本在模式语义、
  拒绝与 runner 失败的分类、fail-closed、enforcement 如实上报。
- **sandtrap 不是 dsh 同层替代品。** 它是 Python 源码沙箱，kernel 模式是 worker 进程 fork 后自锁；
  无法直接沙箱任意 argv，Linux 上只 allow 沙箱根，任意二进制会因惰性加载库失败。
  其结构（每平台 backend + IsolationStatus 如实上报 + fail-closed）与 dsh 同构，值得借鉴，
  但不能作为实现基础。
- **sandlock 是功能上最接近的现成库，但仅 Linux + kernel 6.7+ / Landlock ABI v6，**
  且带 Rust core 与 supervisor 模型，功能面（COW、chroot、镜像、seccomp user notification）
  远超当前需求。是否采用取决于部署约束。
- **landlockpy 是积木不是方案**：纯 Python Landlock 绑定，不负责 spawn，只覆盖 Linux。
- **当前倾向（未定）**：macOS + Linux 场景下，按 dsh 的形状直接用 `sandbox-exec` / `bwrap`
  包 argv，成本约百行，语义层自写；Windows 暂列为未决项。

## Exploration paths

- 先按"找一个 Python sandbox 库"的方向调研，结论是**没有现成库同时满足
  "任意 argv + macOS + Linux"**：sandlock 缺 macOS，sandtrap 缺 argv 接口，landlockpy 缺 spawn。
- 曾认为 sandtrap 最接近 dsh，读源码后修正：接近的是**架构形状**，不是执行单元。
- dsh 0.2.1 的 `.vendors` 只有打包产物；完整源码在 `.playground/dsh-0.2.2-rc.3`，
  后续引用以 rc.3 为准。

## Methods

- dsh 的 profile 构造只有 `profiles.ts` 84 行，可直接作为 Python 后端的对照实现；
  Seatbelt profile 中 workspace 根的 canonicalize 是 macOS 上最易踩的坑。
- Python 里不要用 `preexec_fn` 做 Landlock 加锁（多线程进程不安全）；
  要么用 launcher 子进程，要么 `confine()` 后 `os.execvp`。
- 子进程封装后，denial 与 runner failure 需要分开识别：一个表示"命令被策略拒绝"，
  一个表示"沙箱自己起不来"，两者对模型的错误文案完全不同。

## Implementation Notes

- 待定问题（详见 design 文档第 5 节）：
  - 宿主平台范围：只 macOS + Linux，还是必须覆盖 Windows？
  - 沙箱对象是任意子进程，还是也包括"模型生成的 Python 代码片段"？
  - Linux 部署是否允许安装 bwrap？内核是否 ≥ 6.7（决定 sandlock 是否可选）？
  - 是否需要 seccomp 级系统调用过滤，还是文件写策略已足够？
- 在以上问题确认前，不写实现代码。