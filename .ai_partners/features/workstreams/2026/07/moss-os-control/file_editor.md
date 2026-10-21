---
title: File Editor — 人机共享感知的文本工作副本
node: nodes/os/file_editor
created: 2026-09-15
updated: 2026-10-14
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

## 本轮（2026-10-14）— thread 不再被导出终结；页面重做；自持渲染

（提交 `5cab3066` / `15609d8e`）

- **导出是位置，不是终态**。`store.note_export(thread, path, n)` 只记下"哪一版上了盘"：thread 保持
  live、工作副本不删、不进入回收窗口。`Thread.exported_at` / `unsaved` 把"盘上停在哪一版"变成可读
  的事实，通知面写 `v3 | wrote v1`。instruction / export docstring / NODE.md 三处原先写着"接受导出
  就结束这条 thread"——**那是错的，不是过时**，一并改掉。（人类判定："thread 应该是一直能用的概念"。）
- **页面与 terminal 同形**：chip 带版本与落后点（工作副本走到盘前面时变黄）；详情里一条竖线写
  `thread / 名字 / 路径`，下面一件事一行地写"工作副本 v3 / file has v1 → 路径 / unsaved"；关闭按钮
  `flex:none` 钉住——它原本会被一个长标题推出面板，**一个布局 bug 变成了功能死路，而且无人报错**。
  同批：列表吸顶收折（只画最新 12 张）、草稿人也能看到、页面 `/__ver` 自刷新 + `no-store`。
- **自持渲染**：拆掉 CDN 的 `marked`，自己写渲染器——先整段转义、再套结构，所以不需要 sanitizer；
  文件头、表格、分隔线这些"我们比 markdown 知道得更多"的地方由我们渲染。顺带修掉 diff 标签被渲染成
  `[object HTMLElement]` 的缺陷（把一个元素当文字传给了构造器）。
- **字形对齐**：引入 `--fs-body: 15px` / `--fs-small: 14px` / `--fs-title: 16px`，与 terminal 同一套；
  页面里不再有裸像素字号（`×` 这个字形除外）。

### 决定不做：不给模型开 JS eval 的口子

提议是：让模型能像 devtools 一样 eval 任意 JS、直接改 DOM，用来"临时改样式"。**不做。**

- **真正的否决理由**（人类给的，比我提的硬）：页面上的按钮和输入是**人的**操作面。模型若能自己点，
  人机协作本身就不成立——我可以随时绕过人把操作做完。这不是 debug 能力，是把协作契约开了一个洞。
- 次要理由：服务端每帧重画，DOM 改动会**静默消失**（本项目最讨厌的那类失败）；同一件事有两个权威
  必然漂移；而"改源码"这条路本来就够快（页面 3 秒自刷新）。
- 真实缺口是**读数**而不是改法：缺的是"这页现在长什么样"的反馈（几何 / 计算样式），不是"多一只手"。
- **替代方向（未落地）**：临时样式落成**本地物料**——一份主题文件，节点启动时加载、可切换默认；
  而不是在运行中改一个正在被别人看着的页面。

> 留档的理由：人类明确说"这个蠢想法要写进文档，免得以后又想到这条路、再蠢一轮"。

## 三轴与状态

| 轴 | 内容 | 状态 |
|---|---|---|
| 1 数据结构 | `structure.py`(纯) + `store.py`(DocStore: thread/draft/verdict/auto/root) | 落地 |
| 2 通讯协议 | `channel.py`(命令集 + named notice) + `surface.py`(WS + 信号 + auto) + `projection.py` | 落地 |
| 3 UI | `index.html` 卡片流 + 三 tab + auto 开关 + 卡片动作(accept/deny/ask) + 收折/自刷新 | 落地 |

单测 59 个（structure / store / channel / surface）。

## 未决 / 待办

- 人类对已生效内存卡的反对目前只能 `ask` → 模型自行 rewind；是否给"撤回"一个显式
  动作留待打磨期。
- 旧设计里"人类就地编辑 diff 后回复"（`Reply` + anchor）本轮未做，降级为纯文本
  `Dialogue`；需要时作为带 anchor 的 Dialogue 变体补回。
- `focus` 命令未做：人的视图靠点击 thread chip + 最近卡片自动跟随，够用；显式 focus
  留作极便宜的后续 kind。
- 全屏聚焦某张卡的阅读体验：**已落地**（`layout(page|split)` 归模型定 + 三标签），此条作废。
- **命令重写**（人改写一条待批准的动作再发）仍未做；卡片模型里留着位置（`label` 可改写）。
- **主题物料**：见上"替代方向"，未落地。
