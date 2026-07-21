---
title: Desktop GUI — 人类与 Ghost 共享 desktop 知觉空间
status: draft
priority: P0
created: 2026-07-21
updated: 2026-07-21
depends: []
milestone:
description: >-
  desktop channel 的人类 GUI 交互面。Node cell 独立进程，Reflex web server，
  双栏布局 + 呼吸灯状态 + 审批即对话。Ghost 无感——只通过 desktop channel
  正常操作，人类通过 GUI 观察、审批、对话。
---

# Desktop GUI

## Motivation

desktop channel 为 Ghost 提供了 OS 操作面（bash / file_editor / ground），但与
人类的交互仍然是终端文本。Claude Code 的审批系统展示了一个关键洞察：审批不只是
权限闸门，它是一种对话模式——人类拒绝、追问、确认的过程本身就是与模型的交流。

desktop GUI 是 desktop 的一种交互形式。与 moss-cli、moss-repl、moss-as-mcp 平级——
同一个 desktop 运行时的不同"观看角度"。它是给人类的窗口，Ghost 不需要知道它的存在。

MOSS 架构天然支持"多皮囊"：一个 runtime 实体可以有 CLI/TUI/MCP/GUI 多种交互面。
Desktop GUI 是这一原则的实践——后续可以有 voice GUI、AR GUI 等。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

### K1. GUI 是人类的知觉空间，Ghost 无感

Ghost 只用 `desktop.bash:exec`、`desktop.file_editor:view` 等标准 desktop channel
命令，与没有 GUI 时完全一致。GUI node cell 通过 Matrix session 订阅 desktop 活动，
渲染到界面。人类的审批、追问通过 Matrix 事件传回。Ghost 感知到的是"来了一个信号"，
不是"GUI 发来了消息"。

否决了"Ghost 通过 CTML 控制 GUI 渲染"的方案——GUI 是人类工具，不是 Ghost 工具。

### K2. Reflex 作为 UI 框架

技术选型：Reflex。理由：
- Node cell 独立进程 + 独立依赖，不影响 MOSS 核心依赖重量
- 状态同步内置（WebSocket），省掉手工 SSE/管道
- 组件库直接覆盖双栏、列表、状态灯、表单等需求
- 可 iframe 嵌入其他 UI 体系（需处理跨域）
- `.moss_ws/apps/ui/reflex/` 已有先例，模式可复用

否决方案：FastAPI + SSE（需手工管道）、aiohttp + WS（过重）、裸 http.server（无实时推送）。

### K3. Reflex State 即共享 State

不引入中间数据结构。Reflex State 是唯一的状态源：
- 主线程：Reflex event loop + State + UI 渲染
- 后台线程（daemon）：Matrix asyncio + Channel 注册
- Channel 的 command adapter 直接修改 Reflex State（线程安全写入）
- Reflex 检测变更，通过 WebSocket 推前端
- 审批/追问走 Reflex event handler，写回 State

架构参考：`moss --ai howtos read build-a-gui-app` 的核心模式——
主线程 GUI + 后台线程 Matrix + 线程安全状态共享。Reflex 替代 pygame 的位置。

### K4. 多窗口实例：singleton

Node cell 使用 singleton 模式。`DuplicatedError` 处理同名 node 重复拉起。
打开多个 GUI 窗口时，新实例踢旧实例（互踢），保证只有一个活 GUI 实例。

幂等方案（同一 server 服务多 Ghost session）作为未来扩展保留，不是本轮目标。

### K5. Command 多态渲染

不同 command 类型使用不同的详情渲染组件。Reflex 的 `rx.cond` / `rx.match` 根据
`command_name` 分发：

| command | 渲染组件 |
|---------|---------|
| `bash:exec` | 命令原文 + stdout/stderr 输出 |
| `file_editor:view` | 文件内容展示 |
| `file_editor:str_replace` | diff 视图（旧/新对比，+/- 行高亮） |
| `file_editor:write` | 写入内容预览 |
| 其他 | 通用参数 + 结果展示 |

### K6. diff 用 difflib stdlib

`difflib.SequenceMatcher.get_opcodes()` 返回结构化操作码
（replace/delete/insert/equal），直接映射到 UI 渲染。不需要引入第三方 diff 库。
一行行映射到 Reflex 组件的红/绿色块样式即可。

### K7. 呼吸灯状态机

| 状态 | 灯光 | 含义 |
|------|------|------|
| `pending` | 灰色常亮 | 排队等待 |
| `running` | 蓝色呼吸 | 执行中 |
| `awaiting_approval` | 橙色脉冲 | 等待人类审批 |
| `approved` | 绿色常亮 | 已批准，继续执行 |
| `rejected` | 红色常亮 | 被拒绝 |
| `completed` | 绿色熄灭（可淡出） | 执行完毕 |
| `error` | 红色闪烁 | 执行出错 |

stale 命令（超过阈值时间未活跃的已完成/已拒绝项）通过 toggle 切换显示，默认隐藏。

## Implementation Map

### Phase 1: Node 骨架 + 依赖
- 创建 node 目录结构 + NODE.md
- INSTALL.md 声明依赖（reflex + extras）
- `moss nodes list` 可发现

### Phase 2: 界面原型
- 双栏布局（左侧列表 + 右侧详情）
- 命令列表项 + 呼吸灯（CSS animation）
- 最新/stale 切换 toggle
- 通过 MCP 启动，浏览器验证布局

### Phase 3: 交互逻辑
- 审批流：pending → running → awaiting_approval → approved/rejected
- 状态自动流转
- stale 判定与过滤

### Phase 4: Channel 集成
- 后台线程 Matrix + `provide_channel`
- Command adapter 写 Reflex State
- 审批 event handler 通过 Future 回传

### Phase 5: 差异化渲染
- `str_replace` → diff 视图
- `exec` → stdout/stderr
- 通用 fallback

### Phase 6: 打磨
- 视觉效果调整
- tutorials 同步更新（create-node howto）
- 端到端测试

## Implementation Notes

- Reflex server 的 `run()` 必须在主线程——macOS GUI 限制
- Node 进程通过 `matrix.run_node(target)` 由 host 拉起
- 跨域 iframe 嵌入需要 Reflex 配置 CORS（未来需求，本轮不做）
- Feature 期间同步维护 `tutorials/` 下的 create-node howto
