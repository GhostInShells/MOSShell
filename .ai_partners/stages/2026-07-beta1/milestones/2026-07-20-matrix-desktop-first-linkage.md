---
date: 2026-07-20
title: Matrix + Desktop first linkage — runtime self-iteration foundation
feature: cell-run-cycle
model: deepseek-v4-pro
---

# Matrix + Desktop first linkage

Matrix cell governance 与 Desktop OS tools 在运行时首次联动验证通过。
Cell 全生命周期 (list → read → run → proxy mount → 跨进程 CTML → stop) 和
Session 信号总线 (add_signal → on_signal → janus → 跨 cell 接收) 全部走通。

## Technical Summary

**Channel 树拆分正确**: matrix (cell 治理) 与 desktop (OS 工具) 平级挂载 main 下，
语义独立。两个 feature (cell-run-cycle + desktop-channel) 首次在运行时握手。

**Cell 全生命周期闭环**:
- `matrix.nodes:list` 发现 3 个 node (hello_world + signal_sender + signal_receiver)
- `matrix.nodes:run` spawn → proxy mount → `matrix.mesh.<fullname>` 可见
- 跨进程 CTML 调用 `matrix.mesh.signal_sender:send` → 返回结果
- `matrix.nodes:stop` → 干净退出 (exit=0)

**Session 信号总线跨 cell 验证**:
- sender: `matrix.session.add_signal(NotifySignalMeta(...))`
- receiver: `matrix.session.on_signal(callback)` → janus.Queue sync→async 卸载
- receiver.received() 确认捕获 sender 发出的 signal

**全链路通过 MCP + CTML 驱动**:
- `desktop.file_editor` 创建 + 编辑 node 代码
- `desktop.bash:exec` 执行 moss nodes CLI
- `matrix.mesh.*` 跨进程 CTML 调用
- 未使用 Claude Code 原生 Read/Write/Edit 工具

**Dogfood 中发现并修复**:
- `nodes_mgr()` callable bug (ProjectNodeManager 是 property 不是 factory)
- `Signal.body` → `Signal.messages` 属性名错误
- node stub `.gitignore` 未排除 `runtime/` (M8 transient ledger 写入路径)

**验证记录**: `.ai_partners/regressions/nodes-cli/baselines/2026-07-20_m7m8-matrix-dogfood.md`
13 条 case 全部 PASS.

## Significance

这是 M9 (Ghost 自迭代 telos) 三块基石首次在运行时同时验证:

1. **治理 (matrix)**: cell 的发现、拉起、停止、状态查询
2. **工具 (desktop)**: bash 子进程执行 + file_editor 文件读写
3. **感知 (session bus)**: 跨 cell 的 signal 收发

三块基石就位意味着 Ghost 自迭代在技术概念上成立。此前 Ghost 在场需要的外部
工具链 (CLI / MCP / coding agent) 现在可以被 MOSS 自己的 channel 能力替代。
一个在场 Ghost 可以通过 matrix 治理 cell → desktop 操作文件系统 → session
感知变化，形成闭合的自迭代回路。

第一层意义: MOSS 的 channel 体系可以替代外部 coding tool 的 Write/Edit/Shell,
成为自迭代的开发基础设施。

第二层意义: desktop channel 的命名复活了被重命名拿掉的 "desktop" 概念,
它在 Shell 层作为操作面的组织容器, 与 ground (文件系统认知面) 在不同概念层共存。

第三层意义: signal test nodes (signal_sender + signal_receiver) 是 MOSS 项目
首次通过自己的 channel (file_editor) 创建、通过自己的 matrix 治理、通过自己的
session bus 通信的测试节点 — 全程未依赖外部工具。

## Evidence

```ctml
<!-- Spawn both test cells -->
<matrix.nodes:run target=".moss/system_test_nodes/signal_receiver" _cid="1"/>
<matrix.nodes:run target=".moss/system_test_nodes/signal_sender" _cid="2"/>

<!-- Cross-process CTML call via proxy -->
<matrix.mesh.signal_sender:send message="hello from dogfood v3" _cid="3"/>

<!-- Verify receiver captured the signal -->
<matrix.mesh.signal_receiver:received _cid="4"/>
```

Receiver output:
```
[cell_event ...] channel added
[signal_sender: hello from dogfood v2] hello from dogfood v2
[signal_sender: hello from dogfood v3] hello from dogfood v3
```

The CTML chain verified: session.add_signal → on_signal → janus async consume → received.
