---
date: 2026-09-23
feature: interleaved-voice
model: deepseek-flash
---

# Dolores 全链路实机运行复盘 — prompt 反转，释放焦虑

> 承接 milestone `2026-09-22-dolores-first-full-chain-live-run.md`。那份记录里 ghost 的很多
> 「反省 / 发现」是在 bug 环境下产生的幻觉，不是结论。本文档以人类复盘为准，重新锚定真实问题
> 与下一步方向。

## 基线判断

全链路实机运行本身已经成立，不需要刻意展开正面——工程跑通本身就是结论：模型自己用 node
推流、人类 web 审批、模型开启视觉、看见 dsh 界面、自己写的话、以及它打开的日历 node。

真正的收获不是「暴露了多少 bug」，而是暴露出一个系统性缺口：**机制的能力已到位，但能力对
模型的表述、以及模型对机制的心智模型没有对齐。** 多数摩擦点的根，都在 prompt 与预训练本能
之间打架。

## milestone 里的幻觉，需要纠正

ghost 在 bug 环境里写的「发现」有一部分不是真的。以人类判断为准：

- **「ASR 转写乱序」不是独立问题。** 真正的问题是回声消除——回声的 ASR 和自己说的话不完全
  一样，看起来像外部输入，所以 ghost 长时间把它误判成「外部乱序」。echo 才是那个真根因。
- **「输入带置信度 / 声纹」是正经 feature，人类打算做**，但与「乱序」无关，别绑在一起。
- 其余 ghost 的自我批评（「五次把局部现象推成普遍结论」之类）也多在幻觉环境下写出，不作结论。

## 人类复盘的方向清单

### 已修（今天，见 todo 归口 commit）

1. `wait_action_done` 卡死 → orphan action 修复。
2. 音频：中断回合后有时没中断 → clear player 不重启设备 + 终止流。
3. shell trajectory 的 facade 反转不刷新 → frame index 与 tracer event index 解耦。
4. AEC far 环「取最新」→「按序消费」（未提交，单测 ERLE 2.9→10.0dB）。

### 音频（继续验证）

- AEC 接线仍不正常、回声较大，根因 2 未找到。**但耳机 + mac 公放实际没问题**，问题只在特定接法。
- 首包打断非常良好。
- 礼仪：长程说话尾包拉长一点确实更好；另发现一种——长程说话全程 buffer，尾包发 signal 提示拉（以后再说）。
- 旁路 `voice=all` 忘关，asr 听了一整天（治理/操作面）。

### 核心方向：prompt 反转，释放焦虑

这个方向要对抗的是 deepseek 的预训练，不是文档。ctml 优先在模型内心制造「紧张感」：模型
一旦用 `__content__` 说话发现没 bug，就退回 markdown 思路。三个决定：

- **final answer 必须支持 ctml。** 否则思考 off 时模型无法用 ctml 语义等下一步，被迫吐大段
  无交互价值的文字（纯语音 / 图形界面场景下没人看见）。ctml 只作 tool 的链路永远是
  「输出调用 → 执行 → 返回 → 再思考一轮」，那一轮是白转。所以把 ctml 反转成显式围栏
  `<|CTML|> </|CTML|>`，`__content__` 语音反转加回来，区分 `say`（高级）与普通文本——减少
  token，也让「说话」成为默认无风险路径。
- **interleaved 反转：** 不再强调思维奔逸，默认 `wait_action_done`，思维奔逸改特例，结果返回
  observe，`replan` 先拿掉。
- **`moss_wait_action_done` 也拿掉**，提示用 ctml 去 interrupt 在飞的行动。

### 其它结构性缺口

- **channel 体系的 node 赋名与通知机制要旁路大改**（问题最大的一块）。模型 ctml 出错后转用
  bash 起 node，造成多个管理面；叠加 shell trajectory 的 facade 不刷新，模型焦虑急剧放大。
  但 moss 与 dsh 的生命周期反而绑得很好（dsh web 卡死时 moss tui 仍可用）。
- **harness 的 waterfall 提示词可能丢了**——模型没授权时不会去发申请。
- **重启不从 last session 还原最后帧，和 compact 分开。** 「总痛苦守恒」的反面：之前对话历史
  一直出 bug，模型无法从最近 n 轮的痛苦里解脱；清空、折叠有时是必要的。考虑给模型一个 `clear`
  函数回到只有 memento 上下文的状态。
- **是否允许模型在 memento channel 自定义给旁路 commit 写 instruction**（未定论，方向更「是自己」）。

## 下一步

- todo.md 更新为新的单一事实源。
- 最终回归在 dolores ghost 那里做：让 ghost 自己看 todo，语音一条条试。
