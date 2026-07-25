---
title: Text Blocks — 人机共享文本块载体（流式可写、可批注、diff 上行）
status: in-progress
priority: P1
created: 2026-07-25
updated: 2026-07-26
depends: []
milestone:
description: >-
  双原生共享载体的第一个专职实现：模型经 chunks__ 流式写编号文本块
  （token 流可见即模型光标），人类在 web 界面任意位置修改/批注，
  submit 产生 unified diff 经 signal 上行。窗 = URL，
  不依赖 screen；dump(path) 即资源协议，不依赖 matrix-resources。
  S1 落地：Reflex node，store + channel + UI 三层，dialog 锚定交互。
---

# Text Blocks

> Use `moss features set-status text-blocks <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

长篇文本内容在 MOSS 里没有归宿：`__main__` 的 `__content__` 走语音，
适合讨论不适合承载；chat 式对话流是劣质载体——append-only、不可寻址、
不可就地操作。文本创作类协作需要一个**共享文本块序列**作为共同上下文：
模型写一块，人类改一块，块序列即共同工作区。

哲学基础见 `.design/2026-07-24-shared_carrier_collaboration_philosophy.md`
（共享载体判据：任何一方的动作直接落入另一方的感知，无需翻译劳动）。
本 workstream 是该命题的第一次专职推导：

- **token 流即视觉交互**——人类看着文字长出来，模型的流式写入位置就是
  模型的光标被人类看见（coding agent 改文件时把这个藏掉了，是损失）；
- **共享的是内存中的编号文本块**，不是文件；文件是可选的导出目标；
- **上行通路统一**——界面 submit 与 TUI 打字、语音输入同构，最终都是
  mindflow 的 signal。id 不是回执反馈，是**共享坐标系**：让人类的
  动作能指回模型自己生成过的文本（生成史即共享记忆，token 成本为零）。

用户故事：人类与 Ghost 文本创作对话。语音讨论"为什么改"，屏幕呈现
"改成了什么"，文本块承载"是什么"，`dump` 之后 `cat $FILE >> target.txt`
让成果离开对话。四种通道是同一次对话的四个面。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`
- 哲学锚点：`.design/2026-07-24-shared_carrier_collaboration_philosophy.md`
- 相邻轨迹（参考，非依赖）：
  - `screen-node`（draft）— 窗 = URL；Decision 8 的 peek/drain 双面桶协议是
    本 workstream 上行通路的直接来源；本 node 将是 screen 的第一扇非 mock 窗
  - `desktop-gui`（in-progress）— K3 线程模型（GUI 主线程 + Matrix daemon
    线程）；注意本 workstream **不适用**其 K1（Ghost 无感）——这里模型是
    主动写者，channel 是模型的表达器官
  - `reflex-fix`（completed）— stream chunk 连续渲染丢失已修，正是本
    workstream 的核心渲染需求
  - `matrix-resources`（draft）— 明确不依赖；唯一可能的孵化点是
    "items 作为可寻址资源"，等 dogfood 真实疼痛再评估（or not）

## Key Decisions

### 1. 定位：双原生共享载体，不是 GUI 工具也不是文档编辑器

模型的动作原语对模型原生（`chunks__` 流式输出即动作），人类的动作
原语对人类原生（点击、键入、submit），落在同一个可寻址表面上。
它介于 desktop-gui 的"人类工具（Ghost 无感）"与普通 channel 的
"Ghost 工具"之间——**双方共享的载体**是第三类。

它也不是共享文档编辑器（Canvas/Artifacts 那类回合制形态）：人类的
操作不是"合并进共享状态的写"，而是**一条引用了某段文本的消息**。
单写者模型——模型写块，人类动作以 diff 形式到达，模型自己决定怎么
响应。OT/CRDT、块级锁全部不需要。

### 2. 块编号：server 自增；result 是地址注册，不是控制流

- 块 id 由 server 自增分配——人类在界面插入的内容也产生块，统一由
  server 编号才只有一套坐标系。
- 模型写块后经 `<result>` 收到 id（"item #42 created"）。这是**地址
  注册**：下次用户带这个 id 提交的讯息与这段文本有关。模型不等待它、
  不依赖它继续生成（非 @observe）。
- 模型不需要回读自己写过的内容——**生成史即共享记忆**，id 把人类的
  手指和模型的记忆对上。

### 3. 上行 = unified diff + 引文锚定，走 peek/drain 双面桶

- diff 格式用 **unified diff**——预训练迁移量最大，认知成本为零
  （与人类工程师日常 staged-diff 协作工作流同构）。
- 块内定位用**引文锚定**（"在 '…xxx…' 之后插入"），不用行号——模型
  看不见渲染，行号对它无意义。id 定位到块，引文定位到块内。
- 通路照抄 screen-node Decision 8 / g1 listener 的双面桶：
  - **peek 面**：context_messages 每帧 tail-N 只读 pending diffs +
    块 id 摘要映射（短址纪律），永不 drain；
  - **drain 面**：ghost 场景 drain 后转 signal 进 mindflow；
    **MCP 场景无 signal 通路，diff 留桶由 context_messages 呈现**——
    同一协议自然降档，不是两套实现。
- submit 是"交互的**交付物**"（用户刻意提交），不是"过程"，符合
  "交付物入上下文，过程不入"纪律。submit 按钮同时天然解决防抖。

### 4. 屏幕即真相

人类修改直接生效——块内容当场变，diff 通知模型"现状已如此"。模型把
屏幕状态当环境事实（像传感器读数）。不做"修改即提案"的 pending 状态机。
推论：屏幕状态与模型生成史会分叉，diff 是唯一同步线；`read(item_id)`
命令兜底（compaction 后模型生成史被摘要、或 diff 感知过期时的逃生口，
平时不用）。

### 5. 窗 = URL，不等 screen；dump(path) 即资源协议，不等 matrix-resources

- node 自己 serve URL，浏览器直接打开即可用。screen 落地后免费成为
  一扇窗——"作为 screen 的具体实现"是验收关系，不是依赖关系。
- `dump(path)` 写入文件系统即完成资源化——文件系统是 OS 级共享命名
  空间且预训练迁移量最大，`cat`/重定向/pin 全部免费。不发明协议。

### 6. 技术栈：Reflex node，K3 线程模型

- nodes 大目录下的独立 node（`moss nodes create` 出壳，INSTALL.md
  门控依赖），先例 `nodes/skins/desktop-gui`。
- Reflex：状态同步内置（WebSocket），stream chunk 渲染已由 reflex-fix
  修复验证。线程模型照抄 desktop-gui K3 / `build-a-gui-app` howto。

### 7. 验收标准：运行时自迭代

S3 的验收不是功能清单，是**在这个 surface 里协作演进它自己的设计
文档**：`moss-as-mcp` + coding agent，模型经 MCP 发 CTML 流式写块，
人类在浏览器改，模型从 context_messages 感知 diff，迭代本 FEATURE.md
与实现。回路闭合即验收通过。

## Implementation Notes

- **Stages**:
  - S1 骨架：node 出壳 + Reflex app + channel definition + 纯 UI 回路。
    Reflex 独立运行，浏览器可创建/编辑/提交 block，diff 在 UI 内闭环。
    29 tests pass on store layer。**已完成，待体验测试。**
  - S2 上行：channel 挂 Matrix，NoopScreenPush → ReflexScreenPush
    （WebSocket push chunk），human edit → signal 上行，context_messages
    呈现摘要。MCP 降档形态即可验证。
  - S3 dogfood 自迭代（Decision 7 验收）。
  - S4 ghost 场景 drain→signal + `dump(path)` + screen 落地后挂窗免费。

- **Channel 命名**：定为 `blocks`。短、可键入、不撞产品词（`draft` 撞
  Google Docs，`artifact` 撞 Anthropic，`canvas` 撞 web 技术词）。
  描述用中文写清楚。

- **路径**：`nodes/webview_apps/text_blocks/`。新类目 `webview_apps`
  （与 skins/tools/screens 平级）：WebView 承载的双原生载体 node。

- **数据模型**（2026-07-26 设计会话落地）：
  - Block：id + title + versions 链 + lock(g/u/None) + status(streaming|sealed|error)
  - BlockVersion：version(单调递增) + source(g/u) + content + created_at
  - lock 只有三种状态，同一时刻只有一方持有
  - content/revise/append 在 streaming 态写当前 version，seal 时快照

- **回合制交互**：
  - streaming(g) → 人类只读（dialog view，文字流式增长）
  - sealed(lock=None) → 人类可编辑（dialog edit → submit → diff → signal）
  - 人类 edit 不改变 block 状态，diff 是独立 event
  - 人类创建 block 走 dialog，source=u，直接 sealed

- **channel 命令（9 个）**：
  ```
  content(chunks__, title="", done=True) -> str   # stream → new block
  done(block_id) -> str                            # release lock, seal
  revise(block_id, chunks__, done=True) -> str     # model rewrites
  append(block_id, chunks__) -> str                # continue held block
  replace_line(block_id, line_no, new_text, count=1) -> str
  read_block(block_id, version=None) -> str         # cat -n style
  list_blocks() -> str                              # index
  read_file(path, title="") -> str                  # file → surface bridge
  dump(path="", ids=None) -> str                    # export to filesystem
  ```

- **instruction 只写地址+交互规则**，不重述命令（interface 自动展开）。
  context_messages 放 block 摘要 + action log tail-5，diff 走 signal 上行。

- **引擎**：Reflex。WebSocket native、双向、在项目栈里、K3 线程模型现成。
  对纯文本表面偏重但换轻量方案的 IPC 成本更高。

- **read 格式**：对齐 file_editor 的 `cat -n` 风格（`     1\tcontent`）。

- **dump 默认路径**：`tmp/text_blocks_{session_uid}/`，按 `{id:03d}_{title}.md`。

- **本 FEATURE.md** 由两轮设计对话沉淀：
  - 2026-07-24~25：Fable 5 + 人类工程师，哲学基础 + workstream 创立
  - 2026-07-26：deepseek-v4-pro + 人类工程师，完整交互设计 + 数据模型 + S1 实现
