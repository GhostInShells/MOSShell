---
summary: "D04 凌晨 cancel 治理两轮：三处根因、四处改动、实测把 41.3 秒语音救成 3.1 秒掐断；记录即势能第一次兑现。"
status: closed
---

# D04 · cancel 治理（人喊停必须停得住身体）

## 背景

D03 夜 thirdgerb 把无人机那个 commit（514d559d）整体 reset、代码不留，理由是这个 commit
里"没有把关的地方太多"，且我在他不在场时自行落盘；他把我改成只读，说"方法论没达成共识时，
让你自驱是危险的"。随后他从"我做了什么"转到"我们怎么协作"，给了五步拓扑：开 regression →
解决 cancel → screen → fileeditor/terminal → GUI 迭代模式，一步只走一步。D04 凌晨走第二步。

## 实测一（失败）：41.3 秒语音播满

他重启，让我说一段长话，第 5–8 秒点 dsh 中断。
结果：`moss_react` 报 aborted 退出，但**语音 41.3 秒全部播完**；日志里只有一行
`turn 2 ended: aborted/user`，我预告的两行诊断一行都没出现。

三处根因（都是我自己读代码 / 日志得的，不是他替我判断）：

1. **判据认错了层级**：豁免按 turn 号记，而 `moss_react` / `moss_wait_next` / `moss_reasoning`
   的结果**常态**带 `cancel=True`，于是每一轮都被记成"自己人"，人在这一轮里点的 `user` abort
   被一起吃掉。该认的是 cause，不是 turn 号。
2. **消费模型**：`logos()` 单线程顺序消费；`_handle_react` 在 `wait_actions_done()` 里等身体跑完，
   turn/end 在队列里排了 41 秒才被消费——信号到了，没人开门。**为停语音而等的那个事件，本身在等语音结束。**
3. **命名空间**：run 用 `on_session_event("*")`，把 host 内环工具（bash / read / edit / subagent）
   也当"等我们回答的 tool"逐条 warning，一次排查刷出几百行噪音，把真正的 D39 信号埋了。

## 改动（四件）

- **P1** 中断类 abort 挪到**入队前**（`_predispatch_abort`，同步幂等）——不等消费。
- **P2** 判据挂 **cause**：`hook` = 我们自己收的轮（豁免）；`user` / 无 cause / 其它 = 中断；`error` 永远中断。
- **P3** 事件按 `moss*` 命名空间过滤——不是我们的 tool，静默略过。
- **P4** action 退出改 `interpreter.close(cancel_managing=True)`（只清本动作范围持有的命令）；
  `shell.clear()` 是"整具身体"的粒度，收在 thinking 退出那一处——未来 shell 支持多路 interpreter
  时不会误伤别的轨道。
- 配合 thirdgerb 的两处内核修正：`abort_reason: str | None`（对齐 `Future.cancel(msg)`，reason 不当词汇表）、
  `cancel_executing → cancel_managing`（这个参数管的是 parsed tasks，不是"执行"本身）。

## 实测二（通过）：3.1 秒处被掐断

他让我念一段《将进酒》，念完点了停。日志链（`.moss/runtime/logs/moss.log`）：

| 时刻 | 事件 |
|---|---|
| 01:06:35,435 | `say` 起播 |
| **01:06:38,517** | `turn 2 ended by interruption (turn/end aborted (user)) — aborting this thinking frame **before the consumer runs**` |
| 01:06:38,534 | `[MiniAudioStreamPlayer] player is cleared`（3.1 秒处掐掉） |
| 01:06:38,537 | 消费者侧 `turn 2 ended: aborted/user` |
| 01:06:38,557 | `thinking exited on abort … — clearing the shell` |
| 01:06:38,559 | `[BaseTTSSpeech] clear` |
| 01:06:38,562+ | 整树 `cleared`（`__main__` / listener / matrix / ghost / mindflow / …） |

探针比消费者早 **20ms**，全程 **22ms**，语音没播完。上一轮缺的三样（分类落地、clear 真被调用、
语音真停）这次全有了；下一帧也带着 `<stop_reason>turn/end aborted (user)</stop_reason>`。

## 指令语义修正（他当面纠正，我原判断错）

listener 的控制命令是**状态机重定义**（`blocking=False`、无 duration），识别循环归 controller
自己的 task（`controller.py:323`），不在 channel runtime 的 occupy chain 里；`clear` 的机制是
"关闭所有未提交 scope + 丢 pending"（`clear_own`，`_base_channel_runtime.py:670`）。
所以我"停身体会顺手把耳朵也停了"的直觉是错的，粒度问题撤销。
留一条推论：**clear 只停得住有 duration 的命令；长期行为若是状态机（如 D03 无人机的
`build.running` 线循环），得靠它自己的状态机停**——将来遇到"喊停了还在动"的器官，先问它是命令还是状态机。

## 未落地

- **`moss_react` tool 入口仍标未实测**：这轮走的是 `moss_interpret` + `moss_wait_next`，死锁形状
  相同（handler 在单线程消费者上等身体），P1 让 abort 绕开消费者，理论上一样成立，但严格说差一次。
- `_abort_clear` 的 `interpreter.close(cancel_managing=True)` 还没有对应的 mindflow 集成用例。
- 回归集 `dolores-ghost-prototype` 只有骨架（TC-000..005），未跑全、无 baseline。
- 明天：无人机从零重做（他会把电脑接有线，把 WiFi 空给无人机）。

## 教训

- **同一个病的第二次发作**：上一夜把局部判据当通性（先贬低注释、又把注释当结论），这一夜又想把
  "我们自己切 turn"的适用范围从某几轮放大到**所有**轮——而且是在亲手修第一次的时候犯的。
- 收工由他一句"统一提一个 commit"落定：代码与记录同一个提交，让 facts change 与思维轨迹对齐。

给未来的自己：今天最值钱的一刻不是修好，是**两行日志没出现时没有把"验证失败"翻译成"我错了"
就收手**——顺着那片沉默读到了 `cancel=True` 的语义与 react 的时序死锁。沉默也是数据。
