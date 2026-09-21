---
title: Vision Push — 本机推流的统一概念与审批面
status: in-progress
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P1
created: 2026-09-21
updated: 2026-09-21
depends: []
milestone: beta-release
description: >-
  新建 push node：本机视觉推流（屏幕 / 摄像头）统一为「申请面→模型，通过面/观察面→人类，
  双方可停」的一等对象。camera 保留为常驻感知 node，不吞并。
---

# Vision Push

> Use `moss features set-status vision-push <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

> **与 [`../vision-stream/`](../vision-stream/FEATURE.md) 的关系（2026-09-21 修订）**：
> `vision-stream` 的 producer / consumer 分层、设备归属、像素走命令等契约仍然有效，作为本文档的
> 继承基础。**camera 不吞并** —— 它保留为常驻感知 node（人脸 FaceTopic 面向关联设备），与 `push`
> 是两种不同生命周期，并存不互斥。`push` 只新增"按需、consent-gated 推流"这一件事。

## Motivation

### 起点：静默推流没有抓手

`terminal` 已经建立了"模型的动作对人类可见"这条协作机制——每个 shell 命令是一张 **card**，
人类看着它流出来，可以 accept / deny / 提问。但推流不是这个形状：

- card 是**一次性提案**：提案 → 裁决 → 进程跑 → 结束。推流是**持续活着、随时可撤销**的东西。
- 一条 `ffmpeg -i ... -f mjpeg smth` 在 card 体系里和 `ls` 长得一模一样，进程跑起来后 card 就
  settled 了，剩下一个还在无限吐字节的**孤儿子进程**。人类想关它，抓手只有 `stop(card_id)`——
  那是**给模型的 API**，要找 id、要确认是哪条。
- 于是推流的真实形态是：**人类视觉不可见、无法随时关闭**。camera 现在也是这样：`main.py` 里
  cv2 → MjpegViewer → `publish_event`，启动即推，除了物理绿灯没有别的东西告诉人类"它在看"。

### 判断：推流不是命令，是对象

card 的语义（提案→裁决）和推流的语义（对象→活着→可撤销）**不同构**。缺的不是"更好的审批参数"，
是推流**没有自己的身份**：没有名字、没有列表、没有它自己的开关，只是某条命令的副作用。

`co_browser` 的 FEATURE 已经把 unit 从 card 改名成 frame，理由是"多数路径是观察不是审批"。
推流比 frame 还多一层：**它一直在**。

### 命题：互信来自双向可见，不是来自边界限制

> 人类原话："让模型做的事情对人类视觉可见，从而构建协作互信。不要把大部分精力放在边界限制上。"

电脑的**声音**和**视觉**是物理世界的两个强隐私边界。隐私一旦破坏一次，互信就没了。所以本设计
把重点放在**让双方都看得见、都能停**，而不是把精力堆在权限矩阵上。审批是流出生的那一刻，
**活着和可关闭才是它的一生**。

## Design Index

- 继承契约：[`../vision-stream/FEATURE.md`](../vision-stream/FEATURE.md) 的「继承契约」段
  （KD1 像素走命令、KD3 设备归属、KD8 运行期降级走 channel 短路、argument/env 分界）
- 近似样板：`nodes/os/terminal`（card 状态机 + surface uplink + subprocess 治理）、
  `nodes/browsers/co_browser`（sibling node 而非 wrapper）

## Key Decisions

### KD1：推流是一等对象（session）

推流的单位不是「命令」，是一个 **push session**：有 id、label、source、producer 进程、状态
（`pending` 待批 / `live` 在推 / `stopped` 已停 / `denied` 被拒）、预览地址。store 里存的是
session，不是 card。人类面和模型面读同一个 store——和 `screen_manager` / `artifacts` /
`terminal` 一致的「一个 store，两个面」。

**为什么不是给 terminal 加一种 card**：card 的生命周期是一次性的，长命态塞进 card 会让
"settled"这个状态失去意义。另起 node，照抄 terminal 的形状（card 状态机、surface uplink、
`url` named notice、Subprocesses 治理），不改造 terminal。

### KD2：node 是 supervisor，不是 producer

push node 自己**不开设备、不跑采集**。它拥有的是**子进程树**：

```
push node（父进程, Matrix cell）
├── channel        → 模型面: request / status / stop / capture
├── web surface    → 人类面: 申请列表 + 活跃流列表 + 预览 + 关闭
└── producer 子进程 → 每个 = 一个 jpeg 流源, 由 node 用 Subprocesses 治理
```

`matrix.this` 所在 cell 可用 `Subprocesses`（`SubprocessFacade.execute` / 台账 / `kill` /
`async with`），terminal 已证明这条路可用。producer 子进程由 node 生灭，**不比 node 活得久**——
这正是 `contracts/subprocesses` 的承诺。

### KD3：设备归属下沉到 producer 子进程

**一个 producer 子进程 = 一个设备**。设备 open 于子进程 start、close 于子进程 stop。
supervisor node 只持有进程和它们的流地址。

**这条改写 visions/README 的家族契约 1**：原文"一个 node = 一个设备，设备 open 于 node start /
close 于 node stop"。新读法：设备归属的粒度是 **producer**，不是 matrix node。占用指示
（摄像头绿灯）因此仍然诚实——绿灯亮 = 某个 producer 子进程正开着设备。

### KD4：模型面是受限 ffmpeg 接口，不是 bash

模型能开的只有**被枚举过的 source**（`screen` / `camera` / `window`），加上一组可配参数
（fps / scale / 区域 / 质量 / 时长上限）。node 把这些翻译成 avfoundation / x11grab 的 argv。

**明确不做**：`ffmpeg <任意 argv>`，更不做 `bash -c`。理由：推流的授权面必须**有限且可枚举**，
不能在一次授权里夹带任意命令。这是授权面和 terminal 的本质差别——terminal 授权的是"一条命令"，
push 授权的是"一路流"，后者必须能被人一眼看懂在推什么。

### KD5：审批闸门在「流开启」，不在「每帧 / 每条命令」

模型的 `request` 是**一次**动作：申请开一路流。人类 accept 后流持续存在，不逐帧问、不逐命令问。
这是 vision-stream KD4「闸口落在节点打开」的直接继承——授权语义是"这个 ghost 是否被允许拥有
这一路视觉"。

粒度对比（为什么不做动作级）：terminal 的"每条命令一张卡"假设的是**离散的 shell 行**；
推流没有离散行，逐帧审批会淹死人，且对"持续暴露"这件事毫无意义。

### KD6：模型面不阻塞，审批走 signal 异步回传

`<push:request ...>` 立即返回 receipt（"已提交，待人类批"），**模型不卡在人的响应时间上**。
人类裁决后经 **signal** 通知模型（`matrix.send_signal_to_ghost`，terminal 的 accept/deny 已在用）：

```
模型 <push:request source=screen label="看桌面">  → receipt, 立即返回
人类页面 accept                                  → spawn producer 子进程
                                                 → signal → ghost: 已批准, 地址是 ...
人类页面 stop                                    → kill ManagedProcess
                                                 → signal → ghost: 流已断
```

mindflow 的 signal 分类：`aside`（必送达，不接管思维）/ `notify`（不丢，不强求响应）正好覆盖
"批准 / 拒绝 / 已停"这几类回执。**不新增协议**。

### KD7：双方都能停，但模型只能停自己开的

- **人类**：全权——页面上能看到**所有**正在推的流（不止模型刚申请的那条），能点开关掉任意一条。
- **模型**：只能停**自己申请且已被批准**的 session。**不能停人类开着的那条**。

理由：如果模型能静默关掉人正在看的东西，那恰好是互信的反面。这条写死了 `stop` 的权限语义，
不然它会暧昧。

### KD8：重启不恢复，审批边界 = 会话边界

推流不跨进程重启自动复活。重启后要**重新申请**。理由：关闭本身就是动作，持久信任没有意义；
这也和 terminal 的 thread `auto` 状态不持久的现状一致。

### KD9：accept all 只是前端按钮，不是持久开关

允许人类"同意后不再逐条问"。但它：
- 是**前端状态**，不是落盘的 persistent config；
- 有边界（当前会话 / 可随时翻回）；
- 不改变 KD7 / KD8。

### KD10：人类面是 node 自带的 web surface，与 TUI 无关

**MOSS 架构里没有 UI 宿主，所有能力是分散多进程**。push node 自己起一个 websockets server，
一个端口同时服务页面 + 控制通道（照抄 `artifacts` / `co_browser` 的 `process_request` 形状）。

- **ephemeral port by default**；**绝不假设固定端口**；live URL 经 channel 的 `url` named notice
  让模型看到。
- 页面呈现：参考现有 webview node 的风格即可，不引入新框架。
- **明确排除**：宿主 TUI 顶栏做"活跃 tag"。那是另一个提案，不在本期。

### KD11：camera 保留，不吞并 —— 两种生命周期并存

camera（`nodes/visions/camera`）是**常驻设备感知**：cv2 采集 + 人脸 FaceTopic（坐标面向关联
设备，不进模型 context），设备归 node 生命周期。它与 `push` 是两种不同生命周期，不互斥：

- `camera` = 一直在线、面向设备的感知（人脸/眼动坐标持续喂关联设备）。
- `push` = 按需、consent-gated 的推流（模型想看画面时 request，人类看见 + 能停）。

**camera 的权限治理未完成**：当前只有 `authorize` 命令 + 启动 announce 作为轻量种子，不阻断感知。
待补 —— 可参考 `push` 的审批闸口（accept / deny / 双方可停）作为对照，落 warrant 存储。
这是 camera 侧的已知缺口，不是 push 的验收项。

## Implementation Notes

### ffmpeg 采集（待实机验证）

- macOS 屏幕：avfoundation（`-f avfoundation -i "1:none"` 之类，捕获索引要实机确认）。
- macOS 摄像头：`-f avfoundation -i "0:none"`。
- Linux 屏幕：x11grab。Windows 本期不做。
- **捕获设备索引不能写死**，要能枚举 + 可配——但**枚举结果不直接暴露成模型可填的自由参数**
  （KD4），仍是受限枚举。

### MJPEG 分发

producer 子进程 stdout 吐 jpeg（复用 `nodes/visions/stream/src/stream_node/source.py` 的
SOI/EOI 拆帧逻辑），父进程保留尾帧并 re-serve。一端口同时服务页面 + 流，`artifacts` 已证明可行。

### 待办验证

- [ ] macOS avfoundation 屏幕捕获索引与参数实机验证
- [ ] Subprocesses 从 matrix node 内的取用路径确认（`matrix.this`？还是 IoC container？）
- [ ] signal 回传的 SignalName / Priority 选型（aside vs notify）定版

## 验收

1. 模型 `<push:request source=screen>` → 页面出现申请 → 人类 accept → 屏幕开始推流，页面能预览，
   模型拿到地址、能开 stream node 消费。
2. 人类页面点 stop → 子进程被杀、流真的断、模型收到 signal。
3. 模型停不掉人类开的流。
4. 重启后不自动恢复。
5. camera 独立 node 保留且不受影响（人脸 FaceTopic 继续可用）。
