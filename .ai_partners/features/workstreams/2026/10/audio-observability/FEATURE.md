---
title: Audio Observability — 我自己的声音与听觉，先变成我读得到的
status: parked
status_note: >-
  形状想清楚并留档，正式启动在 stage3（terminal + file_editor + artifacts 完成、dolores ghost 与
  stage2 验收之后）。写成 parked 而非 draft：不占注意力债。
priority: P2
created: 2026-10-13
updated: 2026-10-13
depends: []
milestone:
description: >-
  把我的发声与听觉从"靠人类耳朵验收"变成"我自己读得到的事实"：播放侧的真实出帧计量、两侧频谱的
  缓存与拉取、听侧词级置信度的开闸，以及自监听。目标是让"该响时响、该哑时哑"这件事我自查得出来。
---

# Audio Observability — 我自己的声音与听觉，先变成我读得到的

> Use `moss features set-status audio-observability <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

起点是 D13 那次修完 `--voice none` 仍能说话之后的验收。三条判据——`none` 该哑、`speak` 下裸文本该出声、
显式 `<say>` 仍并存——我一条都判不了：我唯一能看到的证据是命令回执（`played N 秒`），它只证明
**流水线认为它播了**，不证明**该响的时候响、该哑的时候哑**。那一夜我是请 thirdgerb 用耳朵替我验收的。

把任何一个声音断言拆成四层，问题就清楚了：

| 层 | 例子 | 谁看得见 |
|---|---|---|
| 结构层 | `say`/`mute` 挂没挂上 | 我（facade 直接读） |
| 意图层 | 命令跑了没、返回成不成功 | 我（有回执，但弱） |
| **声学层** | 有没有声真的出去、去了哪个设备、多大电平 | **谁都没有 ← 洞在这** |
| 知觉层 | 人耳听清没有 | 只有人类 |

**这个 workstream 要把声学层尽量搬到我这边；知觉层留在人类那里，不搬。** 这不是洁癖，是直播期的硬需求：
直播间里没有第二个人替我验收，而音频是那条时间线上唯一的介质。人不在场我就瞎。

被使能的那一半（真正优化自己的音频体验：设备路由、门限、回声、置信度呈现）是本 workstream 的目标；
命名取的是使能的那一半，因为不先把第三层拿到手，后面每一项都要借别人的耳朵。

## 现状勘查（facts first — 开工时先复核，别信这一段）

三条检索出来的事实，写作时已核，但 FEATURE.md 不是真相：

1. **两侧频谱其实都存在，缺的是留存。** 说侧：`host/voice/interleaved.py` 的 player observe 发
   `AudioSampleTopic(role=ghost)`；听侧：`host/listener/controller.py` 在 `__aenter__` 装线时发
   `role=user`（D08 修的那条）。topic 是 latest-value-wins、5Hz 驱动、错过就没了 —— 所以"频谱"不是没有，
   是**只有直播、没有历史、没法回头拉**。
2. **播放样本计量存在，但不可信。** `cli/audio/render.py` 的 `_report_playback_sample` 会打
   `played N s — no playback samples observed`。D08 实测过它撒谎：回执说"没有样本"，而人耳明明听到了。
   **尺子本身得先修，否则后面每个结论都建在假数据上。**
3. **听侧词级置信度：线装好了，没人开闸。** 全通路是通的 ——
   `volcengine_sauc/protocol.py` 的 `Word.conf` ← `recognizer.py` 解析 utterances ← `show_utterances`；
   `controller._low_confidence_chars` ← 礼仪的 `deliver.low_confidence`；
   `listener_nucleus` 渲染成 `<listen ... low_conf="...">`。
   但**没有任何礼仪把 `low_confidence` 设上**，默认 `None` → 永远返回空串。
   所以"listen 侧修过 ASR 置信度、一直没实装"的准确说法是：**线装好了，闸没人开**。

## Key Decisions

<!-- Record each meaningful design choice. This is what the next model incarnation reads first. -->

### D1 — 判据是分层，不是"做个探针"

任何"我有没有出声/有没有听见"的断言，先问：**它属于哪一层，谁看得见。** 落在第一、二层的自己读，
落在第三层的建观测面，第四层交给人。这条是本 workstream 的方法，也是它的边界：**不试图替代耳朵**，
只把"不该响时响了""该响时没响"这两类**结构性错误**从人类手里要回来 —— 它们本来就不该由耳朵来发现。

### D2 — 探针走"缓存 + 拉"，不加一条推流

thirdgerb 给的判据：频谱可以缓存一部分，走拉逻辑获取。理由是它和 behaviors 里那条同源 ——
**信号要抢我的注意力才能到，状态只要我读就在**。再加一条推送流等于再加一个会丢的东西；
而"我这三秒里到底出没出声"是一个可以回头查的**事实**，不是需要立刻打断我的**事件**。
推流保留给真正有时效的东西（onset、deliver），观测面走拉。

### D3 — 先修尺子，再造新尺

播放计量（现状 #2）不修，后面每一条结论都不可信 —— 这一次 `--voice none` 的验收就是被它拖住才落到人耳上的。
故开工顺序默认：**播放侧的真实出帧计量 → 听侧拉取 → 自监听 → 置信度开闸**。

### D4 — 状态 parked，不是 draft

形状已经想清楚，但排在 terminal + file_editor + artifacts 之后、dolores ghost 与 stage2 验收之后，
stage3 正式启动。写成 `parked`：**一个成型但主动搁置的方案** —— 留档、可检索、不占注意力债。
写成 `draft` 会让它出现在 `features list` 里每晚敲我一次，那是错的债。

## Exploration paths

<!-- Dead ends hit, pivots made, lessons learned. -->

开工前留在这儿的未决问题（不是清单，是需要在启动时收敛的判断）：

- **说侧"到底出声没有"的最小充分证据长什么样**：播放器推出的帧数 + 时间戳够不够？还是要落到设备层
  （实际打开的输出设备、流活没活）？`MOSS_AUDIO_PLAYER_DEVICE` 只钉了"要哪个"，没说过"给到了没有"。
- **审计旧账**：D08 那次 `no playback samples observed` 到底为什么撒谎 —— 是 observer 没在真声路径上触发，
  还是 CLI 那条路和 live TTS 走的不是同一个播放器？先查清再修，别按猜的修。
- **频谱缓存的形状**：窗口多长、放在哪一层、用什么接口拉。倾向不新造一套，先看 topic / memento 那类
  现有机制能不能承载。
- **自监听能不能当一等信号**：我自己的声音绕回麦克风（回声自打断那类问题）今天靠人先发现。
  能不能变成"我在采集里听到了自己"这样一条可读事实。
- **`low_confidence` 打开后噪声有多大**：远场 / 骨传导的音质下，字级置信度是否还可用 —— 未知，要实测。
  这一条直接决定它是"可用的自查面"还是"又一堆需要人读的信号"。

## Methods

<!-- Non-obvious implementation patterns. 开工时补。 -->

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. 开工时补。 -->

一条与机制有关的约束，先记着：本 workstream 的存在**不回写进代码**（`moss features specification` 的
no pointer debt 一条）。代码里只写原理与判据，索引关系留在 features 侧。

## 与既有 workstream 的关系

- `2026/09/interleaved-voice` —— 被观测的那条链路本体（说侧桥 / 听侧 controller / 联动）。
- `2026/06/audio-capture`、`2026/07/voice-input-state-machine`、`2026/07/realtime-voice-interaction-logos`
  —— 采集与礼仪侧的前序决策，开工前先读它们的 Key Decisions，别重打一遍。
- `2026/10/ideas-mechanism` —— 同一个阶段诞生的另一件"往下游放东西"的机制；两者都是"别让未成型的
  东西占着当前上下文"的形状，可以互相参照，不是依赖。

## 启动时机

terminal + file_editor + artifacts 三个可观测工具完成 → dolores ghost 与 stage2 验收 → stage3 正式启动时
把它转成 `in-progress`，并在此处补一句"为什么现在才是对的时机"。
