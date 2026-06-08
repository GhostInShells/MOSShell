---
title: Feishu Channel Integration
status: in-progress
priority: P2
created: 2026-06-04
updated: 2026-06-11
depends: []
milestone:
description: >-
  飞书 IM 集成。MOSS App（group=im）作为独立进程运行，lark-channel-sdk WebSocket 长连接接入飞书。
  推轻量 Signal 给 Ghost，Ghost 通过 Channel 命令 pull/send。Push-pull 分离，Builder 模式，
  不新增抽象，不修改 MOSS 核心。
---

# Feishu Channel Integration

## 1. 目标与动机

让 MOSS Ghost 成为飞书上的一个"用户"——能感知消息、能回复消息，作为一个 MOSS App 独立运行。

**一句话：幽灵上飞书，敲门不递快递。**

飞书 2026 年开放了 Channel SDK 体系。Python SDK `lark-channel-sdk` v1.0.0 提供 WebSocket 长连接、消息归一化（`InboundMessage`）、流式输出、去重等能力。

## 2. 参考材料

| 材料 | 路径 | 用途 |
|------|------|------|
| 飞书 Channel SDK 文档 | `feishu_SDK.md` | SDK 能力边界、接入流程 |
| Open Claw 飞书实现 | `openclaw_FEISHU_INTEGRATION.md` | 架构参考、功能地图 |
| Vercel Chat SDK 适配器 | `vercel_chat_sdk.md` | Python/Node.js Channel SDK 参考 |
| Channel SDK Python | https://github.com/larksuite/channel-sdk-python | 官方 Python Channel SDK |
| Voice App 参考 | `../../sensors/voice/main.py` | App 入口模式、Signal 发送模式 |
| Vision App 参考 | `../../sensors/vision/main.py` | App 入口 + provide_channel 模式 |
| Design Index | `ghoshell_moss.core.blueprint.{app,matrix,mindflow}` | MOSS 核心抽象 |

## 3. 架构

### 3.1 整体拓扑

```
┌──────────────────────────────────────────────────────────────┐
│                     MOSS Host (Matrix)                        │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐    │
│  │ AppStore │    │ Session Bus  │    │   Mindflow        │    │
│  │          │    │   (Zenoh)    │    │   InputSignal     │    │
│  │ start/   │    │              │    │   Nucleus→Impulse │    │
│  │ stop     │    │ add_signal() │    │   →Ghost 被唤醒    │    │
│  └────┬─────┘    └──────┬───────┘    └──────────────────┘    │
│       │                 │                                     │
│  provide_channel()      │                                     │
│       │                 │                                     │
└───────┼─────────────────┼─────────────────────────────────────┘
        │                 │
   Channel proxy     Signal (轻量元信息)
        │                 │
┌───────┴─────────────────┴─────────────────────────────────────┐
│              im/feishu App (独立进程, Cell type=app)            │
│                                                               │
│  ┌──────────────────┐     ┌──────────────────────────────┐   │
│  │  lark-channel-sdk │     │  Channel (Builder 模式)       │   │
│  │  ├─ FeishuChannel │     │  ├─ pull_messages()          │   │
│  │  ├─ WS (后台线程)  │     │  ├─ send_message()           │   │
│  │  └─ EventDispatch │     │  ├─ get_status()             │   │
│  └────────┬─────────┘     │  └─ mark_read()              │   │
│           │               └──────────────┬───────────────┘   │
│     事件回调 (SDK 线程)                   │                    │
│           │               matrix.provide_channel()            │
│  ┌────────┴──────────────┐              │                    │
│  │  MessageBuffer         │              │                    │
│  │  ├─ per-chat deque     │              │                    │
│  │  ├─ _seen 去重         │              │                    │
│  │  ├─ _replied 追踪      │              │                    │
│  │  └─ context_messages   │              │                    │
│  └───────────┬────────────┘              │                    │
│              │                           │                    │
│    生成 Signal → add_input_signal()      │                    │
└──────────────┼───────────────────────────┼────────────────────┘
               │                           │
               ▼                           ▼
       Ghost 被敲门唤醒           Ghost 主动 pull/send
  (Signal 携带完整元信息 →    (通过 AppStoreChannel → Channel
   Ghost 自主决策)              proxy → FeishuAppChannel)
```

### 3.2 数据流

```
[飞书用户发消息]
    │
    ▼
[Feishu Server] ──WSS──→ [FeishuChannel / SDK 后台线程]
    │
    ▼
[_on_message() — SDK 线程]
    │
    ├→ ① MessageBuffer.put() → 去重 (message_id)
    ├→ ② 写入 per-chat deque (maxlen=100)
    ├→ ③ Signal body 构建: "[飞书|私聊|来自:xxx|chat_id:xxx]\n{text}"
    └→ ④ call_soon_threadsafe() → add_input_signal() → Session Bus
    │
    ▼
[Session Bus] → [InputSignalNucleus] → [Impulse] → [Ghost 被唤醒]
    │
    │  每个 think cycle:
    │  ├── context_messages 自动注入未回复消息
    │  ├── instruction 提示 Ghost 必须用 send_message 回复
    │  └── Ghost 直接调用 send_message(chat_id, content, reply_to)
    │
    ▼
[飞书回复] ← [_fs.send(OutboundText)]
```

### 3.3 Signal 格式

Signal body（Ghost 直接看到）:
```
[飞书|私聊|来自:张三|chat_id:oc_xxx]
你好，帮我查一下天气
```

Signal description（Mindflow 路由用）:
```
飞书[张三][私聊]: 你好，帮我查一下天气 | chat_id=oc_xxx | msg_id=om_xxx | chat_type=p2p | sender_id=ou_xxx
```

## 4. 关键设计决策

| # | 决策 | 状态 |
|---|------|------|
| 1 | **Push-pull 分离** — Signal 只带元信息+文本，Ghost 主动 send_message 回复 | ✅ 实施 |
| 2 | **App 而非 Nucleus** — 飞书是 MOSS App (group=`im`)，不是 Mindflow Nucleus | ✅ 实施 |
| 3 | **分组命名 `im`** — 业务域命名，非架构角色。Slack/微信等同类共享 | ✅ 实施 |
| 4 | **双层设计** — Layer 1: `im/feishu` 对话；Layer 2: `im/feishu_ops` 文档/日历 | P3 |
| 5 | **不新增抽象** — 全复用 Matrix + AppStore + Session + Mindflow | ✅ 实施 |
| 6 | **反身性管理** — Ghost 通过 Channel 命令管理 buffer 状态 | ✅ 实施 |
| 7 | **全放 apps/** — 不拆 contrib 库，等第二个 IM 再抽共性 | ✅ 实施 |
| 8 | **lark-channel-sdk** — 选高层 Channel SDK 而非 `lark-oapi` 底层 API | ✅ 实施 |

### 与原计划的偏差

| 偏差 | 计划 | 实际 | 原因 |
|------|------|------|------|
| SDK | `lark-oapi>=1.6.0` | `lark-channel-sdk>=1.0.0` | 高层 SDK 提供 InboundMessage 归一化、内置重连/去重/Token |
| Channel | ChannelState 子类 | Builder 装饰器模式 | Phase 1 无需生命周期钩子，与 vision app 一致 |
| Signal meta | SignalMeta 子类 | description 字符串编码 | Phase 1 够用，后续确认 Mindflow 对 metadata 暴露再引入 |
| API | `add_signal` | `add_input_signal` | MOSS 实际 API 是 `add_input_signal` |

## 5. 开发守则

### 必须遵守

1. **不修改 MOSS 核心** — 不碰 `src/ghoshell_moss/core/`
2. **遵循 App 模式** — 入口 `Matrix.discover().run(main)`，和 voice/audio_capture 一致
3. **不新增 manifest 类型** — app 发现走现有 `AppStore.from_apps_directory()`
4. **单文件 Phase 1** — 所有逻辑在 `main.py`
5. **凭据不写死** — 环境变量读取，APP.md 只放非敏感配置

### 参考但不照搬 Open Claw

Open Claw 有完整 Plugin SDK + Agent 路由 + 配置系统 + 安全审计。MOSS 不同：
- Agent 路由 → Ghost 自己处理（单智能体）
- Plugin SDK → Matrix Cell + Channel（更轻量）
- 配置系统 → 环境变量 + APP.md frontmatter
- 动态 Agent 创建 → 不适用

### SDK 能力边界

SDK 负责（App 不实现）: WebSocket/重连/心跳、消息归一化、Token 管理、去重
SDK 不覆盖（Ghost 处理）: Agent runtime/prompt、多用户上下文隔离、Session 持久化、凭据存储

## 6. 代码速查

### 6.1 文件清单

```
.moss_ws/apps/im/feishu/
├── main.py              # 核心实现（~310 行）
│   ├── FeishuConfig     # 环境变量读取
│   ├── MessageBuffer    # per-chat deque + 去重 + 已回复追踪
│   ├── Channel (Builder) # pull_messages / send_message / get_status / mark_read
│   ├── _on_message()    # SDK 回调（后台线程）→ call_soon_threadsafe → Signal
│   └── main()           # Matrix.discover().run() 入口
├── APP.md               # frontmatter: uv, respawn=true
├── pyproject.toml       # lark-channel-sdk + ghoshell-moss[host]
└── CLAUDE.md            # AI 协作者上下文
```

### 6.2 Channel 命令

| 命令 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `pull_messages` | `chat_id, limit=20, before=""` | `list[dict]` | 历史消息（always_observe=True，拉取后触发 Re-Act） |
| `send_message` | `chat_id, text__, reply_to=""` | `str` | **主要回复命令**，text__ 走 CDATA 传参（无转义），返回自然语言结果 |
| `send_stream` | `chat_id, chunks__, reply_to=""` | `str` | **流式回复命令**，CardKit 卡片实时刷新，Ghost 边生成边推送 |
| `mark_read` | `chat_id, message_id=""` | `bool` | 标记已读（通常无需手动调用） |

### 6.3 Ghost 上下文注入

- **context_messages**: 每个 think cycle 自动展示连接状态（`[飞书 | 已连接 | Bot:xxx]`）+ 未回复消息列表
- **instruction**: `@channel.build.instruction` 提示 Ghost 必须用 `send_message` 回复飞书消息

### 6.4 关键 API

```python
# SDK
from lark_channel import FeishuChannel, ChannelConfig, InboundMessage, OutboundText, SendOpts
fs = FeishuChannel(config=ChannelConfig(app_id=..., app_secret=..., domain=...))
fs.on("message", handler)       # handler 签名: (msg: InboundMessage) -> None
await fs.start_background(30)   # 启动 WSS 连接
result = await fs.send(to=chat_id, message=OutboundText(text=...), opts=SendOpts(reply_to=...))

# MOSS
matrix.provide_channel(channel)  # 注册 Channel（无 await）
matrix.session.add_input_signal(text, description=...)  # 推 Signal

# InboundMessage 关键属性
msg.id              # message_id（去重用）
msg.chat_id         # 会话 ID
msg.chat_type       # "p2p" | "group"
msg.sender_id       # open_id
msg.sender_name     # 显示名（需 contact:user:readonly 权限）
msg.content_text    # 纯文本内容
msg.mentioned_bot   # 是否 @bot
msg.reply_to_message_id  # 被回复的消息 ID
```

### 6.5 线程安全

`_on_message` 在 SDK 后台线程执行。通过 `asyncio.get_running_loop().call_soon_threadsafe()` 桥接到 MOSS 主事件循环。

### 6.6 环境变量

```bash
FEISHU_APP_ID=cli_xxx        # 飞书 App ID
FEISHU_APP_SECRET=xxx        # 飞书 App Secret
FEISHU_DOMAIN=feishu         # feishu | lark | 自定义 URL
```

写入 `.moss_ws/.env`，`load_dotenv` 自动加载。

## 7. 实现进度

### Phase 1：最小闭环 ✅ 完成

| # | 功能 | 状态 |
|---|------|------|
| P1.1 | WebSocket 长连接 | ✅ |
| P1.2 | 事件监听与解析 | ✅ |
| P1.3 | 消息去重 | ✅ |
| P1.4 | Push: Signal 生成 | ✅ |
| P1.5 | Pull: Channel 暴露 | ✅ |
| P1.6 | 文本消息发送 | ✅ |
| P1.7 | 凭据与配置 | ✅ |
| P1.8 | App 生命周期 | ✅ |

测试结果：WebSocket 连接、DM 消息接收、Signal 推送、Ghost 回复飞书均通过。

### Phase 2：体验增强

- [x] P2.1 卡片回复（Card Kit 一次性渲染，`text__` 参数，非真流式）
- [x] P2.1b **真流式回复**（`chunks__` + SDK `stream()` / `MarkdownStreamController` 桥接）— 见 §13.1
- [ ] P2.2 富文本解析（Post/图片/文件/Sticker）
- [ ] P2.3 访问控制（DM/群聊策略、@检测）
- [ ] P2.4 Mention 处理（剥离 XML mention 标签）
- [ ] P2.5 连接状态流（pub_stream_delta）
- [ ] P2.6 消息缓冲策略（可配置 TTL、per-chat 隔离）
- [x] P2.7 sender_name 解析（自定义 `name_lookup` 注入，`contact.v3.user.aget` 单个查询 + 10min 缓存）— 见 §13.2

### Phase 3：完整飞书能力

- [ ] P3.1 飞书操作 Channel（文档/知识库/云盘/多维表格）
- [ ] P3.2 交互式卡片
- [ ] P3.3 媒体收发（图片/文件/音频）
- [ ] P3.4 表情反应
- [ ] P3.5 多账号

### 已知不足（Phase 1 遗留）

| 问题 | 影响 | 优先级 |
|------|------|--------|
| sender_name 未解析（显示 open_id） | ✅ 已修复。自定义 `_name_lookup` 注入 `FeishuChannel`，绕过 SDK 静默失败 | P1 |
| 只支持文本消息 | 图片/文件/Post 被忽略 | P2 |
| 群聊未过滤 @bot | 所有消息都推送 Signal | P2 |
| 无访问控制 | DM/群聊全开放 | P2 |
| buffer 纯内存 | 重启后历史丢失，可能重复回复 | P3 |
| 无单元测试 | 回归风险 | P3 |

## 8. PR Review 追踪

PR #78 收到 17Wang 和 thirdgerb 共 20 条 review。意见已全部处理完毕，见下文"已实施"表格。

### 已实施

| # | 内容 | 状态 |
|---|------|------|
| #1 | `.env.example` 创建于 app 目录，workspace `.env` 继续用于实际凭据 | ✅ |
| #3 | MessageBuffer 存 InboundMessage 替代 dict | ✅ |
| #4 | `pull_messages` 保留 + `always_observe=True` 触发 Re-Act | ✅ |
| #5 | `send_message` 返回值改为自然语言字符串（成功/失败描述） | ✅ |
| #6 | 删除 `get_status`，连接状态迁入 `context_messages` | ✅ |
| #7 | logger 改为 `matrix.logger`（保留 basicConfig） | ✅ |
| #8 | 流式输出 — 新增 `send_stream` 命令，CardKit chunks__ 实时刷新 | ✅ |
| #9 | `@channel.build.instruction` 引导 Ghost 回复 | ✅ |
| #17 | `send_message` 改用 text__ CDATA 传参 | ✅ |
| #2 | ConfigStore 替代手写 FeishuConfig | ✅ |
| #10 | 全局可变状态重构为 AppState + Matrix.discover() 单例 | ✅ |
| #12 | 线程→协程桥接优化为 janus.Queue | ✅ |
| #11 | MODE.md bringup_apps 移除 im/feishu | ✅ |

### 长期债务

| # | 内容 | 状态 |
|---|------|------|
| #16 | ChannelState + on_running 架构升级 | P3 |

---

### 2026-06-10/11 新增变更（本次会话）

| 变更 | 说明 |
|------|------|
| Signal body 增加 `msg_id` | Ghost 第一轮能从 percepts 获取正确 `msg_id`，不受 context 延迟影响 |
| #2 ConfigStore | `FeishuConfig(ConfigType)`；`feishu.yml` 声明配置结构，`$VAR` 自动解析 |
| #12 janus.Queue | `_on_message`→`sync_q.put(msg)`，`_signal_consumer`→`async_q.get()`，线程/协程解耦 |
| #10 AppState | 5 个 `Optional` 全局变量→`AppState` 类；`Matrix.discover()` 进程单例提供 `logger`/`session` |
| #11 MODE.md | default mode 移除 `bringup_apps: im/feishu` |
| MessageBuffer 清理 | 删除死代码 `_unread`/`mark_read`/`status`；新增 `BUF_*` 状态变更日志 |
| Channel 前缀修复 | `app.im_feishu`→`apps.im_feishu`（context 模板 + instruction） |
| 关闭竞态修复 | SDK stop→queue close 顺序；`SyncQueueShutDown`防御；连接失败设 `fs=None` |
| context 延迟诊断 | 根因是 MOSS `_prepare_moment()` 同步读取 `_own_metas_cache`，async refresh 永远晚一轮。非 app 侧可修 |

## 9. 平台配置（飞书开放平台）

已由人类用户完成。关键配置：
- 连接方式：WebSocket（长连接）
- 事件订阅：`im.message.receive_v1`
- 权限：`im:message` / `im:message:send_as_bot` / `im:chat` / `im:chat:readonly`

## 10. 测试

```bash
# App 单独测试
moss apps test im/feishu

# 完整集成测试
moss-run-ghost echo --mode default

# 飞书端：发送消息到 bot，观察 TUI 中 Signal 和 Ghost 回复
```

## 11. 已知问题：context_messages 交付延迟

### 现象

Ghost 看到的 `context_messages` 总是慢一轮——收到新消息后的第一个 Moment 不会展示未回复消息，
要到第二个 Moment 才出现。导致 Ghost 的 `reply_to` 总是指向上一轮的消息 ID。

### 根因分析

`_feishu_context()` 返回值和 Ghost 最终读取的 `<moss_dynamic>` 快照之间存在竞态。
新消息到达 → buffer.put() → CTX 日志显示正确 → 但 MOSS Channel 代理层在生成 `<context>` 区块时
使用的是旧快照，新 context 内容要到下一次 `refreshed` 才被纳入。

### 已排除的假说

- ~~飞书 SDK 重放消息~~ — MSG_IN 日志确认每条消息只投递一次，msg_id 各不相同
- ~~`_replied` 集合时序问题~~ — CLEAR 日志确认 mark_replied 在 send_message 后立即执行成功
- ~~Ghost 自主选择错误的 reply_to~~ — Ghost 忠实复制 context 模板里的 msg_id，问题是模板展示的是旧值
- ~~Signal 描述中的 reply_to 字段污染~~ — 已改名为 `quoted_msg`，问题依旧

### 信号发现

Signal body（`[飞书|私聊|来自:xxx|chat_id:oc_xxx]`）在每个 Moment**都准时到达**，
不依赖 `<context>` 区块刷新。Signal 走 session 通道，与 context_messages 是不同的交付路径。

### 探索方向

1. 将 `msg_id` 写入 Signal body，让 Ghost 在第一轮就能获取正确消息 ID
2. 或调通 MOSS Channel 代理层确保 context_messages 首轮即达

## 12. 变更记录

| 日期 | 变更 |
|------|------|
| 2026-06-04 | 初始 draft，设计决策记录 |
| 2026-06-08 | 整合参考材料、架构讨论、分阶段计划、Channel API 设计 |
| 2026-06-08 | Phase 1 实施完成。SDK 变更为 lark-channel-sdk，Builder 模式 Channel，Signal 纯字符串编码 |
| 2026-06-09 | Signal body 加入 [飞书|来源|chat_id] 富上下文；新增 context_messages + reply 追踪 + instruction |
| 2026-06-09 | PR #78 review 意见整理；#3/#7/#9 实施；三文档统一为 FEATURE.md by deepseek-v4-pro |
| 2026-06-09 | #1 `.env.example` 创建；#4 `pull_messages` 保留 + always_observe；#5 `send_message` 返回自然语言；#6 删除 `get_status`，连接状态迁入 context_messages by deepseek-v4-pro |
| 2026-06-09 | #17 `send_message` 改用 text__ CDATA；#8/P2.1 新增 `send_stream` 命令，CardKit 流式回复 by deepseek-v4-pro |
| 2026-06-10 | #2 ConfigStore 替代手写 FeishuConfig（`configs/feishu.yml` + `ConfigType` 子类，`$VAR` 自动解析）；#12 `call_soon_threadsafe` → `janus.Queue` 线程/协程桥接；#10 全局可变状态重构——5 个 `Optional` 全局变量收拢为 `AppState` 类，`Matrix.discover()` 进程单例提供 `logger`/`session`；Signal body 增加 `msg_id` 字段 by deepseek-v4-pro |
| 2026-06-11 | MessageBuffer 清理死代码（`_unread`/`mark_read`/`status`），新增 `BUF_*` 状态追踪日志；Channel 前缀修复 `app.im_feishu`→`apps.im_feishu`；关闭顺序修复（SDK stop→queue close）；`SyncQueueShutDown` 防御；连接失败时 `fs=None`；#11 MODE.md 还原 default bringup_apps；context_messages 延迟根因诊断为 MOSS 核心 async/sync 竞态 by deepseek-v4-pro |
| 2026-06-11 | P2.1 确认为伪流式（`text__` 一次性传参）；新增 P2.1b 真流式方案（`chunks__` + CardKit 逐块更新）与 P2.7 sender_name 诊断；启动 sender_name 诊断 by deepseek-v4-pro |
| 2026-06-12 | P2.7 完成。自定义 `_name_lookup` 注入 `FeishuChannel`（参考 Open Claw 模式）：`contact.v3.user.aget` 单个查询 + ID 前缀自动检测 + 10min TTL 内存缓存 + code=99991672 权限错误显式提示。SDK 内置 `default_name_lookup` 静默失败根因未定位（不重要） by deepseek-v4-pro |
| 2026-06-12 | P2.1b 完成。`send_stream` 从伪流式（`text__` 一次性传参）改为真流式：`chunks__` AsyncIterator + SDK `stream({"markdown": _producer})` / `MarkdownStreamController` 桥接。移除手动 CardKit 管理（`create_card_instance`/`update_card_element_content`/`finish_streaming_card`），代码 60→37 行。移除 `new_card` 导入 by deepseek-v4-pro |

---

## 13. 待实施技术方案

### 13.1 P2.1b：chunks__ 真流式回复

#### 现状

当前 `send_stream` 命令签名使用 `text__: str`。MOSS CTML 对 `text__` 的语义是"等待模型生成完全部内容后一次性传入"（`DeltaIsTextElement` 解析器）。虽然飞书端卡片设置了 `streaming=True`，但内容是生成完后一次性 update 的——用户看到的是卡片瞬间出现全部文本，不是逐字出现。

#### 目标

模型每生成一段文本，飞书卡片实时追加显示，用户看到逐字/逐段出现的流式效果。

#### MOSS chunks__ 机制（已确认）

`CommandDeltaArgName.CHUNKS = "chunks__"` 映射到 `TEXT_CHUNKS_STREAM` 类型。CTML 解析器在检测到命令签名含 `chunks__` 参数时，创建 `DeltaIsTextChunkElement`：
- 每个 delta token → `sender.append(chunk)` → `AsyncIterator[str]` 实时产出
- 标签闭合 → `sender.commit()` → 迭代结束

命令侧用 `async for chunk in chunks__` 逐块消费。

#### 飞书 CardKit 能力

SDK 已封装三个关键方法（当前代码已在使用，但只用了一次 update）：
```python
card_id = await fs.create_card_instance(card_spec)        # 创建流式卡片
await fs.update_card_element_content(card_id, elem_id, content, seq)  # 增量更新
await fs.finish_streaming_card(card_id, final_seq)          # 结束流式
```

#### 实现方案

```python
@channel.build.command()
async def send_stream(chat_id: str, chunks__, reply_to: str = "") -> str:
    """流式回复飞书消息，边生成边推送卡片。"""
    if _state.fs is None:
        return "发送失败：飞书未连接"

    card_spec = (new_card().markdown("...").streaming(True).build().data)
    card_spec["body"]["elements"][0]["element_id"] = "stream_md"

    try:
        card_id = await _state.fs.create_card_instance(card_spec)
    except Exception as e:
        return f"飞书流式回复失败：无法创建卡片 ({e})"

    result = await _state.fs.send_card_by_reference(
        to=chat_id, card_id=card_id, reply_to=reply_to or None,
    )
    if not result.success:
        return f"飞书流式回复失败：{result.error.message if result.error else '未知错误'}"

    # ── 真流式：逐块消费 chunks__，累积到阈值后推更新 ──
    buf = ""
    seq = 1
    try:
        async for chunk in chunks__:
            buf += chunk
            # 节流：至少 18 字符 + 上次更新距今 160ms（参考 Open Claw）
            if len(buf) >= 18:
                await _state.fs.update_card_element_content(
                    card_id, "stream_md", buf, seq,
                )
                seq += 1
        # 推送最终剩余
        if buf:
            await _state.fs.update_card_element_content(
                card_id, "stream_md", buf, seq,
            )
            seq += 1
    except Exception as e:
        try:
            await _state.fs.finish_streaming_card(card_id, 0)
        except Exception:
            pass
        return f"流式输出中断：{e}"

    await _state.fs.finish_streaming_card(card_id, seq)
    return f"已流式回复到飞书 chat_id={chat_id}"
```

#### 注意事项

- **节流必需**：飞书 API 有频率限制，不能每个 token 调一次 API。Chunk 可能在每个 token 到达，需累积到阈值（如 18 字符或 160ms 间隔）
- **降级路径**：卡片创建失败时返回错误信息给 Ghost，由 Ghost 改用 `send_message`
- **reply_to 清理**：与当前 `send_message`/`send_stream` 一致，回复后 mark_replied
- **Builder 兼容性**：`chunks__` 参数名被 `Command` 反射自动识别为 `delta_arg`，Builder 模式下无需额外配置

---

### 13.2 P2.7：sender_name 解析诊断

#### 现状

飞书消息的 `sender_name` 始终显示为 `open_id`（如 `ou_xxx`），而非用户昵称。`contact:user:readonly` 权限已在飞书开放平台开启。

#### SDK 内置解析链路（已追踪源码确认）

```
InboundMessage.sender_name
  → self.sender.display_name (Identity 类型, 初始值 None)
      │
      ▼ normalize/pipeline.py:275
  if not display_name and resolve_names:
      │
      ▼ identity.py:87 IdentityResolver.resolve_names()
  NameCache 命中？ → 直接返回缓存
      未命中 ↓
      │
      ▼ _api_helpers.py:22 default_name_lookup()
  lark_client.contact.v3.user.abatch(open_ids)
      │
      ▼ 飞书 API: POST /open-apis/contact/v3/users/batch
  (需要 contact:user:readonly 权限)
```

解析是**自动的**——`FeishuChannel.__init__`（channel.py:397-400）在未传 `name_lookup` 时，默认注入 `default_name_lookup`。

#### 可能失败原因

| # | 假说 | 可能性 |
|---|------|--------|
| A | API 调用异常被静默吞掉（`_api_helpers.py:59` `except Exception: logger.debug(...)`） | 高 |
| B | NameCache 缓存了首次失败的空结果，TTL 内不再重试 | 中 |
| C | 飞书事件 payload 中 sender 字段直接赋了 open_id 作为 display_name，导致 pipeline 跳过解析 | 低 |
| D | 权限修改后未重新发布应用（飞书要求发布后才生效） | 中 |
| E | `batch_user_request` 模块 import 失败（sdk 安装不完整） | 低（已验证文件存在） |

#### 诊断步骤

1. **在 `_on_message` 中加诊断日志**，打印 sender 原始四字段：
   ```python
   _state.logger.info(
       "MSG_SENDER: open_id=%s display_name=%r union_id=%s user_id=%s",
       msg.sender.open_id, msg.sender.display_name,
       msg.sender.union_id, msg.sender.user_id,
   )
   ```
   如果 `display_name` 是 `None`，说明 SDK 解析未生效。如果已经是 `open_id` 字符串，说明假说 C 成立。

2. **检查 SDK debug 日志**：`default_name_lookup` 失败只打 `logger.debug`。临时将日志级别调到 DEBUG 观察是否有 `"default_name_lookup failed"`。

3. **自定义 `name_lookup` 注入**：如果 SDK 内置解析确实静默失败，自己实现 `name_lookup` 传给 `FeishuChannel.__init__`，加显式错误日志：
   ```python
   async def _name_lookup(open_ids):
       try:
           result = await _state.fs._client.contact.v3.user.abatch(...)
           _state.logger.info("NAME_LOOKUP: resolved %d names", len(result))
           return result
       except Exception as e:
           _state.logger.error("NAME_LOOKUP_FAILED: %s", e)
           return {}
   ```

4. **确认权限发布**：飞书开放平台 → 权限管理 → 确认 `contact:user:readonly` 已授权且**应用已发布**（仅保存不生效）。

#### 计划

先用步骤 1 的日志定位 `display_name` 的实际值，再根据结果选择后续路径。如果 SDK 自动解析确实不工作，改用手动 `contact.v3.user.batch` API 调用 + 本地缓存。
