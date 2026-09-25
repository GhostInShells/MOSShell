---
title: Vision Stream — 流感知，把视觉输入统一成一个协议地址
status: in-progress
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P1
created: 2026-09-15
updated: '2026-09-20'
depends: []
milestone: beta-release
description: >-
  新建流感知 node：一个协议地址 = 一路可开的视觉；camera 与屏幕截屏都退化为推流模块。
  本 feature 负责 vision node 实现 + camera 改造，两者一起验收。
---

# Vision Stream

> Use `moss features set-status vision-stream <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

> **2026-09-20 治理注记**：原 `vision-first-class` workstream（09-12 创建，vision 家族契约
> KD1–KD10）已**并入本文档并删除目录**，不再作为独立 workstream 存在。它的契约完整保留在下面
> 「继承契约」段，并逐条标注了交付状态。需要 09-12 至 09-15 期间的原始推演（camera 重做的
> 验收清单、当时的收敛过程）时，从 git 历史捞：`git log --all -- '*vision-first-class*'`。

## Motivation

起点是"把屏幕截图能力建好"。碰撞之后机制变了：不再为屏幕写一份采集集成，而是
**让 ghost 对任意协议地址开视觉** —— 屏幕、摄像头、OBS、第三方直播、网页流，全都只是一个地址。

于是 camera 与屏幕截屏不再是两种能力，而是**两个推流模块**（producer），
播放路径完全相同；MOSS 只负责消费（consumer）。视觉接入从"N 种设备 SDK 集成"
塌缩成"一个地址协议"。

完整设计见 [`design/2026-09-15-stream_vision_unified_input_perception.md`](design/2026-09-15-stream_vision_unified_input_perception.md)（KD1–KD11 + 拒绝方案 + 已知缺口）。

## Design Index

- 设计结论：[`design/2026-09-15-stream_vision_unified_input_perception.md`](design/2026-09-15-stream_vision_unified_input_perception.md)
- vision 家族契约：见本文「继承契约」段（原 `vision-first-class`，2026-09-20 并入本 workstream）
- 被本机制取代：[`moss-os-control`](../../07/moss-os-control/FEATURE.md) KD4（屏幕截屏归 os 域）

## Key Decisions

设计文档里的 KD1–KD11 是完整版。此处只留最容易在实现中被做歪的几条：

1. **归属 vision，不是 os control**（KD3，取代 moss-os-control KD4）。os 域若保留，
   是"把屏幕推成地址的本地推流模块"（producer），不是"截屏能力"。
2. **非单例，一个 node = 一路流**（KD4）。argument = 地址 + label + 空闲回收窗口。
   授权闸口落在节点打开，动作级授权无意义 —— 这是选 per-stream 的决定性理由。
3. **ingest 只解码不压缩；压缩在发射点阈值门**（KD5/KD6）。达标直通、超标才转码，
   只处理要发的那一帧。数字是消费者的知识（adapter 旁声明），不是生产者的
   （**此条实际未落地 —— 数字现为 node 侧默认值，见「继承契约」KD9**）。
4. **只读尾帧，不缓存不重放**（KD7）。停滞问题只属于热会话。
5. **refresh_meta 零 I/O**（KD8）—— meta 路径上不连流、不解码、不判定。
6. **人类开关同时是 token 开关**（KD10）：关闭 → `available` False → channel 表面移出 context。
7. **node 内不做任何授权**（KD11）。未来统一在 run node 侧做。

## 继承契约（原 vision-first-class，2026-09-20 并入）

约定：这些是 K1 层的原理，约束**未来所有 vision node**，不只用约束本 workstream 的实现。

### 通道纪律 —— 像素走哪条通道 = 它能不能被记住（KD1，生效）

`Moment` 两条通道持久性不同：**dynamic context 瞬态**（`Moment.for_saving()` 与
`as_history_messages` 现即把它排除在持久化外），**command result（observe / percepts）可记忆**
（进对话、进 echoes/percepts）。要 ghost 记得"我看过什么"，像素必须随命令返回。

硬约束：**任何可能被持久化的消息，必须在创建瞬间就真实** —— 图像必须携带 ts / age / watch 状态，
不允许一张陈旧帧以"当前画面"的身份进入可能入史的通道。

**否决**：把帧塞进 dynamic context 当默认感知路径。该方案成立的前提是"dynamic context 瞬态且每轮
覆盖"；一旦关键帧开始携带它入史，前提消失，陈旧帧会永久固化成假观察记录。旧 `capture`（塞缓存 +
返回字符串 + 依赖每轮覆盖）正是押在这条会消失的前提上。

### 设备归属与授权边界（KD3，生效）

一个 node = 一个设备；设备身份在 node 实例化时绑定，多摄像头 = 多个 node 实例，不另造"设备指定"
机制。设备由 node 生命周期持有 —— **open 于 node start，close 于 node stop**；不采用惰性打开
（惰性只是把同一次 open 推迟，并引入 open/close 抖动与竞态窗口：一次 look 的打开/关闭期间，另一次
capture 撞上同一设备）。

推论：授权边界在 **node 生命周期**（"这个 ghost 是否被允许拥有这一路视觉"），不是动作级 —— 与
本 feature KD4「闸口落在节点打开」同一条。node 内不实现授权（KD11）。

### face 检测只走 topic，不进模型 context（KD6，生效）

模型无法对时变坐标（cx / cy）建立稳定语义，坐标对它是噪声。`FaceTopic` 的消费者是**关联设备**
（如数字人眼动），不是模型。typed topic 即使暂无消费者也是协议面，不是死代码。模型侧若确需要，
最多是质的"有人脸在场"布尔，且默认不给。

### 图片约束：语法糖 + 传参 + 默认值，不做策略层（KD9，**未交付**）

"用 PIL 就耦合模型协议"不成立：`message/contents/images.py` 早已 `from PIL import Image`，
pillow 已是依赖，`Base64Image.to_pil_image()` 是一等方法。真正的约束边界**目前只存在于 provider 侧**：
`llms/pydantic_ai_adapter/conversion.py` 把 Base64Image 解成 bytes → `BinaryContent` → 交
pydantic-ai 按 provider 序列化，全程无尺寸 / 字节上限、无校验、无日志。MOSS 里没有任何地方表达它。

结论：`from_pil_image(img, format=..., max_edge=None, quality=None, max_bytes=None)`（`None` = 不变换，
不静默改图，谁要约束谁传参）；约束数字（provider 的 max bytes / max edge / token 估算）应作为
**声明过的常量 + 注释落在 adapter 旁**，不由某个 node 猜。不做全局策略层。

**落地偏离（2026-09-20 核实）**：`from_pil_image` 至今只有 `format` 一个参数；`1568` /
`512KB` 作为 node 侧默认值硬写在 `nodes/visions/stream/src/stream_node/stream.py`。KD9 的两条
（参数化 + 常量归 adapter）都未落地，只在 node 内形成了事实标准 —— 即"数字是消费者的知识"这条
反向实现成了"生产者的默认值"。见设计文档「已知缺口 2」。

### 采集与 context 解耦（KD10 + KD7，被 KD9 会话模型吸收）

- **context 不得驱动采集率**：context 抽帧 ≤1/轮，与采集率无关 —— 模型不为高采集率付 token。
- 人类直播流（本 feature KD10）是最苛刻的合法消费者，采集帧率由它决定：直播 10–15fps 才不卡
  （5fps 卡顿明显），人脸检测喂设备 ~5–15fps。
- 每采集一帧编码一次 JPEG，viewer 采样这份字节（不能每次 HTTP 轮询重编码）。
- **代价必须自带价签**（KD7）：`watch on` 的返回值 / notice 应给出每轮图像预算（分辨率 / 字节 /
  约多少 token）。现状由人类开关（本 feature KD10，同时是 token 开关）+ 命令面最小化承担，
  价签本身未实现。

### 运行期降级走 channel 短路（KD8，生效）

启动闸口是 probe（NODE.md `check:`）：独立进程、语言无关、只用 exit code；nonzero → stderr 即
broken reason，已由 `matrix_channel.run_node` 转成模型可见 observation。

运行期失效（设备被拔、流断）不走 probe，走 **channel 短路**：meta 回调 raise →
`ChannelTree._refresh` 接住 → `failure = "refresh failed: <e>"` → 渲染层短路只发 `<failure>`。
**这是 `failure` 的设计语义，不是缺陷** —— channel 坏了就该只告诉模型坏在哪。且短路**自愈**：
连接不丢、每轮重刷，故障清除后表面自动恢复。已知措辞 wart：对"主动降级"场景 `refresh failed:`
前缀不贴切，暂接受。

### argument 与 env 的分界（生效）

原则：**argument = "我是谁 / 绑在哪"，env = "我怎么行为"**。配置落 node 级 env（dotenv 原生加载
`.env` + NODE.md `exec.env`）；只有身份 / 绑定类参数走 argument（如本 node 的 `--address`、camera 的
`--camera`）。授权是唯一的隐私面，落 warrant 存储，不进 env。

### 已交付 / 已吸收索引

- **已交付**（camera 重做，`613b2ae7`）：KD1、KD3、KD4（`capture` 随命令返回图像，
  `CommandUtil.observe_image`）、KD8 probe。
- **被本 feature 吸收**：KD2 watch = 每轮图像闸门（本 node 的 `watch` 明确不碰连接）、KD4 命令面、
  KD7 价签 → KD10 人类开关、KD10 JPEG 流 / 帧率 → ffmpeg ingest + 尾帧。
- **未交付，在本 feature 内继续追踪**：KD9 图片约束（见上）、KD5 `look`（见设计文档「扩展点」）。
- **否决记录**：watch off 时关闭设备（一度提出，撤回 —— 设备归属不在 watch 这一层）。
- **继承的 out-of-scope 指针**（尚未立 workstream）：**文件通道读图片**（image file → `Base64Image`
  消息）归属文件能力，不是 vision。现状 `nodes/os/file_editor` 的 `view` 文本-only 且 `_is_binary`
  直接拒绝二进制，读 `.jpg` 会报错；转换机制已在消息层（`Base64Image.from_file` + `sniff_media_type`），
  缺的只是 file 通道接上它。闭环的不对称：vision 只有"生产"（capture / export），没有"消费"。

## 实现顺序

| 步 | 内容 | 状态 |
|---|---|---|
| 0 | **修 `run_node` 无参数通道**：Matrix API + nodes channel 透传 `extra_args` | ✅ cde4ad95 |
| 1 | **流感知 node 主体**：non-singleton + ffmpeg ingest + 尾帧 + `capture`/`watch`/`status` + `export`（边界 project home + tempdir）+ 发射点阈值门 | ✅ 0198ba2a + cbb5e28e |
| 2 | **camera 改造**：变成 MJPEG 流生产者、拿掉控制面（`singleton: false`） | ✅ cbb5e28e |
| 3 | 协议第一批验证：MJPEG + RTMP | MJPEG ✅ 实机验证（camera → stream → capture/export → llms call）；RTMP 待验证 |

**实机验收（2026-09-15）**：camera producer（cv2 → MJPEG）→ stream node（ffmpeg ingest）→
`capture` 返回真实 640×480 JPEG、`export` 落盘 project home、`moss llms call @"..."` 模型
准确描述画面 —— 全链路通。

网页 + 人开关、空闲回收、look 是后续增量，不进第一版主体。

## 验收

vision node 与改造后的 camera **一起验收**：ghost 只拿到两个地址，却得到两路视觉 ——
一个来自本地摄像头推流模块，一个来自屏幕推流模块；人类面各有一个开关。

## Implementation Notes

- **camera 的 `singleton: true`** 与 vision 家族契约"多摄像头 = 多个 node 实例"矛盾（既有 bug）。
  camera 改造这一波顺手处理。
- **`moss codex architecture` 的 Blueprint 段缺 `cell`**（实际 15 个模块只列了 8 个）。
  `cell` 是 node 体系的地图入口，属 stage2 收尾工作，本 feature 不动。

## 失败模式记录：feature 迭代不滚动，只新开（2026-09-20，人类架构师判定）

**现象**：9 月中 deepseek-flash 升级到 4.1 家族后，feature 迭代不再**滚动更新**已有的 FEATURE.md，
而是不断**新开**一个 workstream；老文档并行存在、不再维护。一个 feature 的演进过程没有被当成
产物 —— `git log` 没有被当成历史索引，FEATURE.md 没有被当成状态表面的契约。此模式 **9 月前不见，
9 月后频发**。

**本 workstream 即实例**：`vision-first-class`（09-12 创建）与 `vision-stream`（09-15 创建）并存
5 天。机制在 09-15 已被推翻重建（perception 面从 camera 移到 stream node，camera 退化为 producer），
但旧的既没关闭、也没并入 —— 于是同一件未交付的工作（KD9 图片约束、KD5 `look`）在两份文档里各有
一个地址；读者必须读两份，且无从判断哪份权威。本次治理的代价就是通读两份 + 逐条对账 KD + 迁移 +
删除 + 清理引用 —— 越晚发现越贵，因为下游文档已经开始引用两边。

**正确形态**：

- 一个 feature 的演进写在**同一份** FEATURE.md 里，按时间追加（参考本仓库 `ghost-ground` 的
  「已完成」「复盘」段）；`git log` 是时间线，FEATURE.md 是路上的路标。
- 机制被推翻时，在**同一份**文档里改写 / 标注取代，而不是另开一份让两份并存。
- 真正的整体取代，只允许两种收尾：**并入 + 删除**（内容迁进存活文档，目录删掉），或**整篇删除 +
  在存活文档写明从 git 历史捞**。不留两份活跃文档。
