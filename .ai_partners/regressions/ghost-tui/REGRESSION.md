---
title: Ghost TUI — 交互行为回归
version: 1
status: draft
priority: P1
created: 2026-08-03
updated: 2026-08-03
scope: subsystem
depends:
  - ghost-tui-refinement
description: >-
  Ghost TUI 交互行为的验证轨迹：interrupt 通路（escape → shell.clear）、
  c-c 提示/退出两级语义、干净关闭、logos 渲染、input 计数。人类在 moss-ghost
  中视觉验证，模型记录结果到 baseline。
---

# Ghost TUI

> 每套回归 = 验证轨迹。用例表是当前权威清单；首次完整运行结果记入
> `baselines/YYYY-MM-DD_vN.md`。

## Exploration Index

```
moss --ai all-commands --group ghost          # ghost 相关命令面
moss codex get-source ghoshell_moss.host.tui_entries.ghost_ui   # Ghost TUI 实现
moss codex get-source ghoshell_moss.core.mindflow.interrupt_nucleus  # interrupt 通路
moss codex get-source ghoshell_moss.host.ghost_runtime           # shell.clear() 调用点
git log --oneline -10 -- src/ghoshell_moss/host/tui.py src/ghoshell_moss/host/tui_entries/ghost_ui.py
```

## Methodology

**协作方式**：人类 + 模型配对。模型读 methodology 与步骤、提议怎么验证；人类执行动作
（操作 `moss-ghost` 交互终端，无法自动化）；模型记录结果与 root cause。G1 模式：
人类说"验证这个"，模型提议方案，人类对齐后模型带人类走完执行。

**执行方式**：human-in-the-loop，无自动化。全在 `moss-ghost <name>` 交互终端里进行。

**执行顺序**：P0 优先（interrupt 通路 —— 本回归首个版本的修复核心，应为 PASS）；
P1 其次（已知 bug 的现状确认，预期 FAIL，记录 root cause 供后续会话参考）。

**已知 FAIL 说明**：TC-005 / TC-006 是 ghost-tui-refinement 项 #1 / #2 的已知 bug，
本版本预期 FAIL。FAIL 的 baseline 同样是价值 —— 记录诊断，让后续实例不用重查。

## Prerequisites

- 本机已完成 `uv sync --active --all-extras`，`moss-ghost` 可启动
- 有可用 ghost（默认 `echo`）
- 人类可看到 TUI 视觉输出（logos 流、hint 行）
- 深层验证（TC-004）可选：`moss-mcp` + `moss-as-mcp` 暴露 shell 状态

## Test Cases

| Case ID | Priority | Description | Test Steps | Expected Result |
|---------|----------|-------------|------------|-----------------|
| TC-001 | P0 | escape 打断 logos 生成并触发 shell.clear | 1. 启动 `moss-ghost echo`<br>2. 发一个会长 logos 的任务（如让 ghost 输出多段/长文本，留足打断窗口）<br>3. 生成中途按 `escape` | logos 流立即停止；出现 hint `interrupt sent — generation stopped, shell cleared`；随后输入正常响应，无半截命令残留 |
| TC-002 | P0 | c-c 空输入区：首按提示、二按 exit（interrupt 收归 escape） | 1. 空输入区按 `c-c` 一次<br>2. 1.5s 内再按 `c-c` | 首按出现 hint `press c-c again to exit`，不打断生成；二按退出 TUI（进程结束） |
| TC-007 | P0 | 关闭无 pending task 噪音、exception handler 不崩溃 | 1. 正常退出 TUI（TC-002 二按 c-c 或 `/exit`）<br>2. 观察终端输出 | 关闭过程无 `Task was destroyed but it is pending!` 警告，无 `Unhandled error in exception handler` / `AttributeError` |
| TC-003 | P0 | c-c 输入区有内容时清空 buffer，不打断不退出 | 1. 输入区键入一段文本（不回车）<br>2. 按 `c-c` | buffer 清空；TUI 不退出；正在进行的生成不受影响 |
| TC-004 | P1 | interrupt 后 shell 状态干净（深层验证） | 1. 触发一次 interrupt（TC-001，escape）<br>2. 用 `/ghost.health()` 或 MCP inspect shell 状态 | interpreter 已关闭、pending commands 已取消、无积压；下一次 articulate 从干净状态起步 |
| TC-005 | P1 | logos 尾段即时渲染（ghost-tui-refinement 项 #1） | 1. 发一个输出多行 logos 的任务<br>2. 等生成结束，期间不再输入 | logos 最后一段在生成结束时立即完整显示（当前为已知 bug：需等下次 input 才补上） |
| TC-006 | P1 | input 计数回落（ghost-tui-refinement 项 #2） | 1. 连续发多个 input signal<br>2. 观察 mindflow perspective 的 "pending input from ghost tui" 计数 | 计数随信号处理回落至 0（当前为已知 bug：持续累加不下降） |

## Execution Notes

- 验证 TC-002 二按 exit 后 TUI 会关闭，需重新 `moss-ghost echo` 继续后续用例。
- interrupt 唯一入口是 escape（TC-001）：需要生成进行中有足够窗口按 escape，选一个会让
  ghost 流式输出较长 logos 的 prompt；若 echo ghost 太快，可换更 verbose 的 ghost 或换长任务。
- TC-003（c-c 清 buffer）验证时确认输入区确有内容，且按 c-c 后不触发退出。
- interrupt 信号走 InterruptNucleus 有冷静期（winning-side cooldown）——连续 interrupt
  会被静默吞掉，属正常协议行为，不是 bug。验证时先确认上一次 interrupt 已被消化。
- 判定 "无半截命令残留"：interrupt 后再发普通 input，ghost 应正常生成，不出现上一次
  未完成命令的残留行为。
