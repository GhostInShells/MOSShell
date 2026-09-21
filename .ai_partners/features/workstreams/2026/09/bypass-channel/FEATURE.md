---
title: Bypass Channel
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P2
created: 2026-09-21
updated: 2026-09-21
depends: []
milestone:
description: >-
  单 runtime 内的治理分离：在 __main__ 上开一条旁路通道，其签发独立于主路时序，
  主路打断不影响旁路。感知面合并，执行面可分离。
---

# Bypass Channel

> Use `moss features set-status bypass-channel <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

MOSS 目前的时序模型是**单一时间线**：主解释器（`__main__`）持有当前 turn 的整个命令流，
`clear` / 打断沿树广播，父 occupy 冻结子树。也就是说，打断主路 = 打断一切。

这个默认本身是对的：绝大多数能力就该被回合边界截断，否则陈旧命令会跨轮泄漏，
模型"签发顺序 = 物理顺序"的心智模型也会失效。

**本机制是补充，服务于一类扩展能力**：when(某段执行必须与交互解耦——即主路打断
完全不影响它)，回合截断就是错的。对这类能力，设计动机只有一条：
**模型这一轮结束了，手还在动。** 口停手不停不是"治理好的话还能顺便做出"的附加效果，
而是本机制为这类能力开的口子。**默认路径不变，能力是叠加的。**

目标机制的形态与性质：

- **旁路是 `__main__` 上的一个 command**，不是第二个 shell。签发仍通过一个解释器，
  因此**每个 interpreter 内部仍然保有签发时序**（时序不消灭，只是从主路分离出去）。
- 典型用法：`<bypass:exec scope="xxx">CTML</bypass:exec>`，用 ChannelModule 组装，
  scope 是一组 defer 运行的控制槽——**同一 scope 再次签发时，先 clear 旧命令**。
- 提权语义：被旁路接管的子树，其调度脱离祖先 occupy 的冻结——这是让"执行不被回合
  截断"真正成立的机制，不是可选的优化。
- 仍然是**一个 runtime**：`push_task` / tracer / 感知轨迹 / ShellTrajectory 全部共用。
  这是它与"多 shell"方案的分水岭——多 shell 会把感知轨迹也分裂开，而本机制的前提是
  **感知面合并、执行与交互可分离**。
- 纪律：子通道仍要求全 channel path 签发命令，且**不允许越权**。

**为什么本期不做**：破坏面积不可预料。它推翻的是树的既有承诺（父 occupy 冻结子树、
父 clear 清空子树），而目前只有 skip 名单这一处治理来接住；旁路自身的生命周期由它的
scope 管，这部分是清楚的，但**旁路与主路打同一子树时谁仲裁**还没有结论。
留此 draft 供未来实验，下面是本轮已探明的落点与地雷，下次启动直接从"未决问题"接手。

## Design Index

本 feature 暂无 `design/` 与 `discuss/`。本轮分析只发生在一次会话里，
结论全部收在本文件 —— 下次实例不需要回溯原始对话。

## Key Decisions

### D1 — 不做多 shell，做单 runtime 内的治理口

**选了**：在一个 runtime 里动态分离治理。
**否掉了**：多 shell（每个旁路一条独立 shell）。
**理由**：多 shell 解决问题的方式是把整棵树再复制一份，代价是**感知轨迹也分离**——
`MShellEventTracer(Tracer)` 是按 shell 挂的，`ShellTrajectory` 收的是本 shell 的事件。
旁路机制要的是"执行面可分、感知面合并"，多 shell 正好相反。

### D2 — 旁路用 dry_run interpreter 纯解析，再直接推目标子树 runtime

**理由**：dry_run 分支 `callback=None`（`core/ctml/shell/ctml_shell.py:425`），
而 `self._interpreter` 只在 `callback is not None` 时被赋值（同文件 `:467`）——
**旁路的解析器不会顶掉主解释器**。这个前提不成立则整个设计需重做，现已成立。

**已就位的手柄**（无需新造机制，只有第 4 行需要新入口）：

| 需求 | 现有手柄 | 位置 |
|---|---|---|
| 解析但不派发 | `kind="dry_run"` | `ctml_shell.py:425` |
| 从非根节点进入这棵树 | `runtime.push_task_with_paths(paths, task)` | `core/runtime/_tree_channel_runtime.py:136` |
| 不允许越权（能力收窄成交集） | `config=` → `channel_metas(selection=config)` → `commands(config=config)` | `ctml_shell.py:445` |
| cid 不串（旁路 token 独立命名空间） | `stream_id=` | `ctml_shell.py:453` |
| 封装成可增删的旁路通道 | `with_module` / `PrimeChannel.build` | `blueprint/states_channel.py` |
| 开通道必须指定子树根名 + notice 提示 | `gate()` 渐进披露 + 保留名 `gated_children` | 同上 |

> 行号是 2026-09-21 的快照，代码在动，依赖前先核实。

### D3 — "提权到全局"不是新调度引擎，而是换一个节点进入

occupy 冻结子树不是运行时内部的锁，而是**祖先运行时的消费循环不再往下路由**造成的。
所以直接进子树 runtime 的队列，冻结自然失效。语义干净，不需要新增调度器。

### D4 — tree.clear 加 skip 名单，而不是把子树摘出树

`MOSShell._clear` 走 `tree.clear(self._main_runtime)` 全树遍历
（`ctml_shell.py:744-751`）。要做到"口停手不停"，旁路子树必须不被这个广播命中。
**选了**：clear 遍历时跳过名单内的旁路子空间，保留树结构完整。
**否掉了**：把子树真正摘出树——破坏面更大，且丢失了"子树还能被全路径寻址"的能力。

**红线**：skip 名单**只跳过 clear，绝不跳过 pause**。全局急停必须仍然停得住旁路，
与碰撞点 1 的 `_check_paused` 是同一条底线。

### D5 — 治理权归运行时模型：自己开、自己绑、自己关

**前提假设**：旁路的开与关不是构建期由人类配置的，而是运行时模型用自己的命令完成的
（模型自治理 / 反身性）。

**落点**：正是 `MutableChannelState.add_virtual_channel` / `remove_virtual_channel`——
它们的 docstring 原文就写着 "wrap this method into a command"。所以"绑定子 channel"
不是新抽象，是这个 API 的预期用法；`gate()` 提供渐进披露（声明的 catalog 先只出现在
notice 里，模型发命令才挂载）。

**由此强化的两条**：

1. **披露面就是控制面**。模型只能治理它看得见的东西：`gated_children` 告诉它能开什么，
   每个 scope 的 named fragment 告诉它已经开着什么、什么时候没了。所以碰撞点 4 不是
   "提示要给到位"的体验问题，而是**治理能力的前提**。
2. **不能出现"开了关不掉的手"**。模型自己开，就必须保证它自己关得掉：
   - 关闭通路不能被它自己要逃离的那条 occupy 链堵死；
   - scope 登记（谁开着）必须与执行面一起存活——clear 之后模型不能失去自己的账本。

### D6 — 宏策略未定（见"未决问题"第 1 条）

倾向：旁路 CTML 禁宏，解析期遇到 `meta.macro` 直接报 interpret error。

## Implementation Notes

### 四个碰撞点，前两个是红线

**1. 轨迹与急停只挂在 `MOSShell.push_task` 上。**

`ctml_shell.py:576` 是唯一做这三件事的地方：`_check_paused()`、
`_fire_on_task_pushed(task)`、`task.add_done_callback(self._on_task_done_hook)`。
而观测轨迹正是靠这两个 hook（`blueprint/shell_trajectory.py:476,535,538`）。

所以直接 `runtime.push_task_with_paths` 的旁路任务会：
① **不进 ShellTrajectory** —— 与"结果进轨迹"的目标直接冲突；
② **绕过 `_check_paused`** —— 全局急停（emergency-stop-tui）对旁路失效，安全红线。

**结论**：旁路不能停在 runtime 那一层，必须在 shell 上开一个新的进入点
（形如 `push_task(task, at=<runtime>)`），保留 pause + tracer hook，只改进入点。

**2. 宏与 dry_run 结构上不相容。**

`core/ctml/interpreter.py:412` 明写 `run_macro=not self._is_dry_run`；
`concepts/shell.py` 中 `parse_tokens_to_command_tasks` 的注释更直白：dry run 里
展开宏会 **await 一个永不完成的 task 而死锁**。

旁路的卖点是"解析完就推、不等"；宏展开的本质恰恰是"解析器停在原地等宏任务跑完再续"。
二者不可能同时成立。三个选项：

- (a) 禁宏，解析期直接报错 —— 最干净，"不等"是这个功能的定义，宏是唯一必须等的东西。
- (b) 旁路自己驱动展开（只对宏这一步阻塞）—— 可接受，但要在旁路通道内 await 宏任务。
- (c) 把宏语义从解析器搬到 runtime —— 更大的重构，超出本 feature 范围。

**3. 提权必须同时意味着"从 clear 广播摘出去"（D4 已给解法）。**

提权同时推翻树的两条承诺：父 occupy 冻结子树、父 clear 清空子树。
**这不是待补的破口，这就是设计动机本身**——when(执行必须与交互解耦) 时，
"模型这一轮结束了、手还在动"正是要的结果。所以 skip 名单是让动机成立的承重件，
不是对误伤的事后补救；旁路 scope 的生命周期由它自己管，不需要上一级代管。

由这个动机反推出的真正要求是：**旁路执行必须在下一轮的上下文里可读**。
否则模型看不到自己的手还在动，会签发冲突命令、或对世界状态判断错误——
"模型对自己的身体失明"恰好是 MOSS 要消除的失败模式。所以"感知面合并"落到这里
就具体化为：旁路 scope 的存活状态必须进披露面（见碰撞点 4），它是机制的一部分，
不是装饰。同理，turn 结束后才回来的旁路结果是**常态而非边界情况**，
回报规则默认应当是"累积进轨迹、下一轮可见"，signal / Re-Act 留给 `@observe`。

**4. 披露层级用错了会瞎。**

scope 来了又走，因此它属于 `named_notices` 的 `None`=移除；**不能用单值 `notice`**。
`blueprint/channel_builder.py` 写死了理由：`""` 已经被"未变"占掉，
单值 notice **无法宣布自己的消失**，只有 named fragment 能发出 `<scope-x removed/>`。
（若旁路通道只开一次不关，单值 notice 才够用。）在模型自治理的前提下（D5）这条是硬的：
模型看不见的 scope，它管不了也关不掉。

### 其它待验证

- **旁路与主路同时打同一子树**：两条 dispatcher 喂一个 runtime 队列，
  runtime FIFO 仍是仲裁者，但模型的"签发顺序 = 物理顺序"心智模型在此失效。
  需要明确：旁路一开，该子树是否对主路仍然可签发，还是被 defer 槽独占。
- **旁路结果的方向**：旁路任务在下一个 turn 之后才完成是常态，所以回报不能依赖
  "Interpretation 还在跑"——那会让结果等于静默。两条现成路径：
  `Interpretation.on_done_task`（累积进轨迹，下一轮可见）／
  `CommandUtil.send_signal` → Session → mindflow（会触发 Re-Act）。
  按碰撞点 3 的结论，默认走前者，后者留给 `@observe`；
  **不需要为旁路发明第二套回报规则**。但必须明确写死，否则"结果进轨迹"无法验证。

### 改动面判断

全部集中在 **shell 层**：一个新的进入点 + 一个 ChannelModule 封装 + `tree.clear` 的
skip 名单。runtime 内部不必改动——`push_task_with_paths` 本来就是"从某节点往下走"的
原语，缺的只是"从非根节点进入"的 shell 级封装与治理。
