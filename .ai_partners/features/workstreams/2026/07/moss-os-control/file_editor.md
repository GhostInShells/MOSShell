---
title: File Editor — 人机共享感知的文本工作副本
node: nodes/os/file_editor
created: 2026-09-15
updated: 2026-09-19
status: in-progress
---

# File Editor

> os-control 的子文档（KD5 展开）。能力落在 `nodes/os/file_editor/`。
> 它是**人机共享的一条感知线**，不是编辑器——对话与共享是目的，export 落盘是唯一真实副作用。
>
> **机制来源（2026-09-19）**：上一版（对话线 + confirm 门控）代码没成型，人类协作者
> 先在 `terminal` node 上把整套设计模式重建并实测（卡片第一公民 / 双面一 store /
> 回执 + 信号 / 审批即对话），本 node 反哺同一机制。逻辑与最初聊的一致，机制换血。

## 定位

核心目的：**让人与模型共享关于同一份文本的感知**。模型读什么、改什么，人类在同一条
卡片流上看到同一件事。所以：

- **读是共享感知**：`read` 立刻把内容返回模型（进上下文），同时落一张卡，让人类看到
  模型看到的同一段文本。读无副作用。
- **编辑立即内存生效**：`write` / `append` / `str_replace` / `rewind` 落笔即生效，
  不等人。卡片只是让人类看着流式产出边看边问。
- **唯一闸是 `export`**：落盘是唯一真实副作用，它等价于其它工具的 approve。

与 `terminal` 最大的区别只有三点：**命令不走 subprocess**（进程内 Python 调用）、
**读接口立刻返回**（不是异步回执，卡片只为观测面）、**只有 export 待批**。

## 数据模型

| 实体 | 语义 |
|---|---|
| `Thread` | 一个可编辑对象：`label` / `path`(可空=空白新建) / `base`(open 时文本) / `actions` |
| `Action` | 线上一次动作 = 一张卡。`n`(per-thread 单调)/ `kind` / `label` / `state` / `effect` / `dialogue` |
| `Effect` | `before` / `after` / `diff` —— 长在 action 上，append 时算 |
| `Dialogue` | 挂在卡上的对话条目（人 `ask` + 模型回应），不决定任何事 |

`kind`：`read / write / str_replace / append / rewind / export`。`open` **不是 action**
（它是建线，线程 chips 已表达它），因此 version = 最后一条改内容的 action 的 `n`，
第一个编辑就是 v1。

**三张账**（回应"不全进 append-only log"）：

| 层 | 内容 | 寿命 |
|---|---|---|
| 内存轨迹 | thread 的 actions + effect + version | 随 thread，export/close 释放 |
| draft tempfile | 每 thread 一份当前全文，生效动作后覆盖写 | open 建，export/close 删 |
| 真实文件 | 唯一真实副作用 | 永久 |

不做 append-only log：durable 的是 draft 而非历史，文件系统因此有界（一个活 thread
一个小文件）。内存轨迹崩了即失（可接受），draft 保证内容不丢。

## 规则

1. **effect 在 append 时算**，基准是当时的 `content`，与人的裁决无关。confirm 不再是
   "内容生效的闸"，`content` = 最后一条非 rejected action 的 `after`（这里 rejected
   只可能出现在 export，而 export 无 effect）。
2. **rewind = 普通 action**：payload 指向 `n`（`0` = baseline），只重算 effect。
   append-only，永不擦除。
3. **str_replace 一卡一 op**：`text__` 承载一段 JSON（str_replace 协议 `old_str`/
   `new_str`，单条或列表），每条 op 一张卡、顺序生效、后 op 看到前 op 结果。
   `old_str` 必须恰好出现一次，否则报错。
4. **卡片不含内容**：卡片只给 kind + thread label + state（+ 机械效果字段）。内容
   点卡后懒加载（`detail` 帧返回 effect source + full）。不推 delta。
5. **export 即终章**：accept → 落盘 → thread 冻结、payload 释放、draft 删除。
   想继续改 → 对同 path `open` 新线。`close` 是放弃出口（删 draft、不碰磁盘）。

## 关键决策

- **K1 共享感知**（见定位）：读留卡、人看同一段文本，是全部机制的目的。
- **K2 卡 = 动作，但 open 除外**：read 也占 seq（无 effect，不计入 version），
  open 不占——它只是建线，不是"被修改的文件"的一次动作。
- **K3 三层账，不做 append-only log**：durable 面收窄成 draft mirror，避免文件系统
  爆炸与启动 GC 一堆日志。孤儿 draft 按 TTL（7 天）启动时 sweep，TTL 内进 notice
  供 `open(draft=...)` 认领。
- **K4 export 即终章 + 回收区（静默）**：export 落盘后 thread 冻结（继续改 = 开新线），
  但**不释放 payload**——人类仍可追溯整条线。清理是独立步骤：ended thread 超过窗口
  （`MAX_ENDED_THREADS=8`）时淘汰最旧的一整条（类似 subprocesses 的回收区），人无感。
  活 thread 上限 16，出口显式（export/close）。
- **K5 named notice 承载当前 version**：`<thread_<id>>` 片段 = `v{n} | label | N lines
  | path`，文本变了才重发——上下文压缩后模型仍知道自己在 v7 上（照 SpeechModule 的
  "命令面写契约、状态走 notice"分工）。
- **K6 卡片不含内容 + 点卡懒加载**：action 帧只带 identity + 机械效果字段，content
  走 `detail` 请求。流式长文档因此便宜。
- **K7 markdown 渲染走 CDN**：`marked@4.3.0` + 失败回退转义纯文本（照
  `nodes/webview_apps/artifacts` 先例，不 vendor）。
- **K8 信号分级**：人 `ask` → notify(next=True)（要当场答）；export accept/deny →
  aside（不打断）+ awaiting 归零一条 notify；export 完成/失败 → notify。
- **K9 per-thread auto，且 auto 只在最终目标已建立时成立**：`Thread.auto` 是每文档
  信任（照 terminal 的 per-thread auto）。但"最终目标建立才能 auto"——auto 只对
  `path` 已确立的 thread 有意义，pathless 的 thread 无法 auto（`set_thread_auto` 拒绝）。
  `export` 只在**落到已确立目标**时走 auto；写到一个新 path = 建立新目标，仍要审批。
- **K10 root 边界（默认授权）**：`open`/`export` 的路径必须落在 `project_home ∪
  系统 tempdir` 内，范围外一律拒绝（默认不授权，不是"再问一次"）。tempdir 是即用即弃
  的低爆半径空间，适合"暂存一份给人看"；draft 按名访问，天然限定在 node runtime 内。
- **K11 双边界防 10 小时爆炸**：服务端回收区（ended thread 超 8 淘汰最旧整条）；
  客户端 `MAX_CARDS=300` 卡片 Map 上限（最旧静默淘汰）。ended thread 的卡片**保留**
  供追溯，只在服务端淘汰（thread 从 threads 帧消失）时才同步清 DOM——不随 export
  立即清空。

## 三轴与状态

| 轴 | 内容 | 状态 |
|---|---|---|
| 1 数据结构 | `structure.py`(纯) + `store.py`(DocStore: thread/draft/verdict/auto/root) | 落地 |
| 2 通讯协议 | `channel.py`(命令集 + named notice) + `surface.py`(WS + 信号 + auto) + `projection.py` | 落地 |
| 3 UI | `index.html` 卡片流 + 三 tab + auto 开关 + 卡片动作(accept/deny/ask) | 落地 |

单测 56 个（structure 17 / store 21 / channel 14 / surface 4）。

## 未决 / 待办

- 人类对已生效内存卡的反对目前只能 `ask` → 模型自行 rewind；是否给"撤回"一个显式
  动作留待打磨期。
- 旧设计里"人类就地编辑 diff 后回复"（`Reply` + anchor）本轮未做，降级为纯文本
  `Dialogue`；需要时作为带 anchor 的 Dialogue 变体补回。
- `focus` 命令未做：人的视图靠点击 thread chip + 最近卡片自动跟随，够用；显式 focus
  留作极便宜的后续 kind。
- 全屏聚焦某张卡的阅读体验（当前右侧 detail）——与 terminal 的"全屏 command 主题"
  并列待办。
