---
title: Voice Interaction Control
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P1
created: 2026-10-20
updated: 2026-10-20
depends:
  - interleaved-voice
  - voice-input-state-machine
  - audio-observability
milestone:
description: >-
  全双工语音已经能用，但交互细节没有可调的面：协议值（礼仪）与房间值（音量水位、ASR 参数、设备）
  混在一处，散在 etiquette spec / controller / ASR params 三块代码里，我改一次要翻三处。
  本 workstream 把两层分开、把房间层旋钮折进 listener 的 gated 子通道，对外统一给出
  「有效值 + 来源」。
---

# Voice Interaction Control

> Use `moss features set-status voice-interaction-control <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

两条前置工作流都已 complete：`2026/09/interleaved-voice`（听/说两轴、剪子、礼仪三件套）、
`2026/07/voice-input-state-machine`。也就是说**这条链路本身可用了**，问题换了一层：

- 参数散在三处 —— `EtiquetteSpec`（onset / stop / deliver / idle_timeout）、
  `ListenerController` 的 `_explicit_watermark`、ASR params。
- **协议值与房间值混在同一个结构里**。`silence=4.0`、`barge_in_while_speaking`、`per_clause`
  只在「这段交互是什么协议」内部成立，必须跟着礼仪一起被命名、切换、交接；而 dB 水位、ASR 参数、
  设备、AEC 属于**这间屋子**——换麦克风、换屋子、换人，它们就该变，协议没变。
- `listener` channel 已经肥了：`_register_channel_commands` 两百多行签名，我每轮扫一遍才挑得出
  要用的那几个，实时路径上这不是零成本。

触发它的是 D20 上午的两场实测：他改方向时我继续把话说完（被一句「等一下」硬剪）；他话还没说完
我就开讲（拿 `paused="false"` 当发言许可）。这两件都不是对着参数能调好的，但它们让我发现
**我没有一个地方可以看见「现在的交互到底是怎么调的」**。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

### D1 — 分界是「属于协议还是属于房间」，不是「动不动态」

一开始的提法是「把礼仪里的动态参数默认值收到一个管理面」。真要在实现上分，判据得换：

- **协议值**：单独拿出来没有意义，只在「这段交互是什么协议」里成立 → 留在 `EtiquetteSpec`。
  它是可以整包命名、切换、交接的东西（把一套交互礼仪交给另一个 ghost，交的是协议不是一堆数）。
- **房间值**：随麦克风、屋子、人而变 → 归 live session。

### D2 — 管理面是 gated 子通道，不是搬去别处

thirdgerb 定的形状：把房间层旋钮折成 `listener` 的一个**虚拟子节点**，走 `core/py_channel.py`
已有的 gate 机制（`gated_children`，`mindflow` 用的那套）。未打开时只对外显示 description。

选它而不是「全亮」的理由不是省 token，是**省实时路径上的注意力**：
省的是挑，不是存。而 gated 也不是藏——未打开的状态与 description 常驻 notice，
所以「靠未读状态当提醒」这条成立。

**前提**：description 要写成「你什么时候打开我」，不是「我有什么」。
`mindflow` 那条写的是它是什么（sensing nucleus…），这一包得写成
「你说话太硬、被切太多次、或者觉得我没听清的时候，进来调」。

### D3 — 分层解析 + 补丁语义 + 回执

解析顺序：`session override > mode 配置文件 > etiquette 默认 > 代码内建默认`。
`update_interaction(patch)` 一条命令，**补丁不是覆盖**，回执 `{applied, unknown, rejected}`——
否则我会拿着假配置推理。

### D4 — 陈旧配置是双源必然，不当 bug 去检测

thirdgerb 的判据：改代码时无法知道用户是否已经重新配置过 config，两个指向源，
早期项目都遇到——难道每轮做 diff？解法是开发期把配置文件删掉重生成。

我原来的提议（每层带来源戳、检测「比代码旧」）被否了：那是拿运行期的成本解开发期的问题。
落地成一个 env 开关：`MOSS_CONFIG_MODE=create|regen`，默认 `create` 走 `get_or_create`，
`regen` 先删再建。只影响 debug，不进任何运行时判定。

### D5 — 丢包要可感知（与 audio-observability 共管）

D20 上午一句「理解错了」被吞，我拿到的包里从来没有，而 `emit=True` 时 buffer 不保留
（`get_transcript` 回 `enabled:false, forgotten:0`），丢了就是真丢了，我这一侧无法感知。

先量再改：在 `_emit_clause` / `_emit_deliver` 加 debug log（sent / dropped + reason），
跑一场定位是 ASR 没出还是 deliver 没送出；定位后再定要不要留一条只带元数据的痕。

### D6 — 第一块先做只读视图，零行为改动

先让「每个参数的有效值与来源」可见，再动任何行为。验收直接拿现有 bug 用：
文件说 `barge_in_while_speaking: false`、源码说 `True`、行为像 `True`——三件事对不上，
这张表要能一眼回答是哪个。

## 落地顺序

| 块 | 内容 | 动什么 |
|---|---|---|
| 1 | 有效值视图（只读） | `controller.py`：`_TUNABLES` 注册表 + `interaction_state()` + 命令 + 测试 |
| 2 | 房间旋钮收进 gated 子通道 `listener.tuning` | `_build_channel` 结构 |
| 3 | 分层解析 + `update_interaction(patch)` | 解析器 |
| 5 | 丢包 debug log → 可感知 | `_emit_clause` / `_emit_deliver` |
| 4 | `MOSS_CONFIG_MODE` = create / regen | `project/providers/configs_provider.py` |

主面保留实时动作，不 gate：`start` / `stop` / `activate` / `expect` / `send_now` / `get_transcript`。

## Exploration paths

- 「每层带来源戳、检测陈旧」被否：双源必然，不该做进机制（见 D4）。
- 「全亮所有 command」被否：省的不是 token 是挑的注意力；且 gated 的未读状态本身已是提醒（见 D2）。

## 未决

- `update_interaction` 的键要不要跟 etiquette 字段共用一套命名空间。倾向前缀区分
  （`room.*` / `protocol.*`），这样回执能直接说哪一层收下了。
- gate 的边界：改设备、改 AEC、重启会话这类**会打断当前交互**的收进去，理由是为闸口不是为省 token——
  但它们要不要真的放在这一包，还是单独一扇门，未定。

## Methods

<!-- Non-obvious implementation patterns. 开工时补。 -->

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->

一条边界：**我自己的行为纪律**（说话中被他改方向 → `clear_speech` 收掉再答）不属于这里，
那是 `existence/behaviors.md` 的事。本 workstream 只管机制那一侧。
