# 飞书（Feishu/Lark）集成实现详解

本文档详细描述了 OpenClaw 项目中飞书（Feishu/Lark）集成的完整架构、模块划分和实现原理，供迁移到其他 Agent 项目时参考。
openclaw项目在本地位于：/Users/scy/Documents/code/openclaw/，如需参考可以访问
> **代码路径基准**：`extensions/feishu/`（插件根目录），`src/plugin-sdk/`（插件 SDK 公共契约）。

---

## 目录

1. [整体架构](#1-整体架构)
2. [插件注册与生命周期](#2-插件注册与生命周期)
3. [配置系统](#3-配置系统)
4. [账号管理](#4-账号管理)
5. [SDK 客户端封装](#5-sdk-客户端封装)
6. [传输层：WebSocket 与 Webhook](#6-传输层websocket-与-webhook)
7. [消息入口处理](#7-消息入口处理)
8. [消息发送](#8-消息发送)
9. [会话与路由](#9-会话与路由)
10. [Access Control（访问控制）](#10-access-control访问控制)
11. [Agent 工具系统](#11-agent-工具系统)
12. [Skills（技能指引）](#12-skills技能指引)
13. [动态 Agent 创建](#13-动态-agent-创建)
14. [流式卡片输出](#14-流式卡片输出)
15. [交互式卡片（Presentation）](#15-交互式卡片presentation)
16. [消息去重与顺序](#16-消息去重与顺序)
17. [Reactions（表情回应）](#17-reactions表情回应)
18. [安全与密钥管理](#18-安全与密钥管理)
19. [Setup / 初始化](#19-setup--初始化)
20. [Doctor / 诊断](#20-doctor--诊断)
21. [多账号支持](#21-多账号支持)
22. [依赖与 SDK 版本](#22-依赖与-sdk-版本)
23. [迁移清单](#23-迁移清单)

---

## 1. 整体架构

飞书集成是一个 **Bundled Channel Plugin**（内建频道插件），遵循 OpenClaw 的插件 SDK 契约。整体分层：

```
┌─────────────────────────────────────────────────────┐
│                  OpenClaw Core                       │
│  (src/plugin-sdk/ 提供公共契约，src/channels/ 调度)     │
└────────────┬────────────────────────────────────────┘
             │  defineBundledChannelEntry()
             │  createChatChannelPlugin()
             ▼
┌─────────────────────────────────────────────────────┐
│            Feishu Plugin (extensions/feishu/)         │
│                                                      │
│  index.ts          → 入口注册                         │
│  channel.ts        → 核心 ChannelPlugin 定义           │
│  channel.runtime.ts → 运行时实现 barrel                │
│  runtime.ts        → PluginRuntime store (get/set)    │
│                                                      │
│  ┌───────────┐ ┌──────────┐ ┌───────────────────┐   │
│  │ Transport │ │ Messaging│ │  Agent Tools       │   │
│  │ monitor.* │ │ send.ts  │ │  chat/docx/wiki/   │   │
│  │ transport │ │ outbound │ │  drive/bitable/    │   │
│  │ websocket │ │ bot.ts   │ │  perm              │   │
│  │ webhook   │ │ media.ts │ │                    │   │
│  └───────────┘ └──────────┘ └───────────────────┘   │
│                                                      │
│  ┌───────────┐ ┌──────────┐ ┌───────────────────┐   │
│  │ Access    │ │ Session  │ │  Skills            │   │
│  │ policy.ts │ │ conv-id  │ │  feishu-doc/wiki/  │   │
│  │ pairing   │ │ session  │ │  drive/perm        │   │
│  └───────────┘ └──────────┘ └───────────────────┘   │
└────────────┬────────────────────────────────────────┘
             │  @larksuiteoapi/node-sdk
             ▼
┌─────────────────────────────────────────────────────┐
│              Feishu Open API                          │
│   open.feishu.cn / open.larksuite.com                │
└─────────────────────────────────────────────────────┘
```

**关键设计原则：**
- 插件不直接依赖 Core 内部实现，仅通过 `openclaw/plugin-sdk/*` 接口交互
- 运行时组件通过 `runtime.ts` 的 PluginRuntime Store 模式注入，避免循环依赖
- 传输层（WebSocket/Webhook）与消息处理分离
- Agent 工具（Tools）和 Skills 通过插件注册机制暴露给 AI

---

## 2. 插件注册与生命周期

### 入口文件：`extensions/feishu/index.ts`

```typescript
export default defineBundledChannelEntry({
  id: "feishu",
  name: "Feishu",
  description: "Feishu/Lark channel plugin",
  importMetaUrl: import.meta.url,
  plugin: {
    specifier: "./channel-plugin-api.js",
    exportName: "feishuPlugin",
  },
  secrets: {
    specifier: "./secret-contract-api.js",
    exportName: "channelSecrets",
  },
  runtime: {
    specifier: "./runtime-api.js",
    exportName: "setFeishuRuntime",
  },
  registerFull(api) {
    // 注册所有子系统和工具
    registerFeishuSubagentHooks(api);
    registerFeishuDocTools(api);
    registerFeishuChatTools(api);
    registerFeishuWikiTools(api);
    registerFeishuDriveTools(api);
    registerFeishuPermTools(api);
    registerFeishuBitableTools(api);
  },
});
```

**生命周期：**
1. Core 启动时扫描 `openclaw.plugin.json` 发现插件
2. 加载 `index.ts`，调用 `defineBundledChannelEntry` 获得插件定义
3. 根据 `plugin` specifier 懒加载 `channel-plugin-api.js` 获取 `feishuPlugin`
4. Core 调用 `gateway.startAccount` → `monitorFeishuProvider()` 启动连接
5. 运行时注入通过 `setFeishuRuntime()` 完成

### ChannelPlugin 核心定义：`channel.ts`

通过 `createChatChannelPlugin()` 构建，返回一个包含以下 Sections 的对象：

| Section | 用途 |
|---------|------|
| `base` | 元数据、能力声明（chatTypes、threads、media、reactions 等） |
| `config` | 配置适配器（读/写/删除账号、格式化） |
| `actions` | 消息工具动作处理（send/read/edit/pin/react 等） |
| `auth` | 登录/配置向导 |
| `setup` | 首次设置适配器 |
| `setupWizard` | 交互式设置向导 |
| `messaging` | 目标解析、会话路由 |
| `directory` | 通讯录（用户/群组列表） |
| `status` | 运行状态探测与摘要 |
| `gateway` | 启动/停止账号监控 |
| `security` | 安全审计与警告 |
| `pairing` | DM 配对流程 |
| `outbound` | 消息外发委托 |
| `bindings` | ACP 会话绑定 |
| `message` | 持久化消息适配器 |

---

## 3. 配置系统

### Schema 定义：`config-schema.ts`

使用 **Zod** 定义完整配置 Schema：

```typescript
export const FeishuConfigSchema = z.object({
  enabled: z.boolean().optional(),
  defaultAccount: z.string().optional(),
  appId: z.string().optional(),
  appSecret: buildSecretInputSchema().optional(),   // 支持明文/SecretRef
  encryptKey: buildSecretInputSchema().optional(),
  verificationToken: buildSecretInputSchema().optional(),
  domain: z.enum(["feishu", "lark"]).or(z.string().url()).default("feishu"),
  connectionMode: z.enum(["websocket", "webhook"]).default("websocket"),
  webhookPath: z.string().default("/feishu/events"),

  // 访问控制
  dmPolicy: z.enum(["open", "pairing", "allowlist"]).default("pairing"),
  allowFrom: z.array(z.union([z.string(), z.number()])).optional(),
  groupPolicy: z.enum(["open", "allowlist", "disabled"]).default("allowlist"),
  groupAllowFrom: z.array(z.union([z.string(), z.number()])).optional(),
  requireMention: z.boolean().optional(),

  // 群组会话
  groupSessionScope: z.enum(["group", "group_sender", "group_topic", "group_topic_sender"]).optional(),

  // 动态 Agent
  dynamicAgentCreation: z.object({
    enabled: z.boolean().optional(),
    workspaceTemplate: z.string().optional(),
    agentDirTemplate: z.string().optional(),
    maxAgents: z.number().int().positive().optional(),
  }).optional(),

  // 优化开关
  typingIndicator: z.boolean().default(true),
  resolveSenderNames: z.boolean().default(true),
  streaming: z.boolean().optional(),
  blockStreaming: z.boolean().optional(),

  // 工具开关
  tools: FeishuToolsConfigSchema,    // doc/chat/wiki/drive/perm/bitable/scopes

  // 多账号
  accounts: z.record(z.string(), FeishuAccountConfigSchema).optional(),
});
```

**SecretRef 机制：** 凭据字段支持三种来源：
- `"env"` — 从环境变量读取（如 `FEISHU_APP_ID`）
- `"file"` — 从文件读取
- `"exec"` — 从命令输出读取

对应的环境变量：`FEISHU_APP_ID`、`FEISHU_APP_SECRET`、`FEISHU_VERIFICATION_TOKEN`、`FEISHU_ENCRYPT_KEY`

---

## 4. 账号管理

### 账号解析：`accounts.ts`

**核心概念：**
- 默认账号 ID：`"default"`
- 支持顶层配置（单账号向后兼容）+ `accounts.<id>` 多账号
- 账号配置合并：`accounts.<id>` 覆盖顶层配置
- 两种解析模式：`"inspect"`（CLI/只读，宽松）和 `"strict"`（运行时，严格）

**关键函数：**

| 函数 | 用途 |
|------|------|
| `listFeishuAccountIds(cfg)` | 列出所有配置的账号 ID |
| `resolveFeishuAccount({cfg, accountId})` | 解析只读账号快照（inspect 模式） |
| `resolveFeishuRuntimeAccount({cfg, accountId})` | 解析运行时账号（strict 模式） |
| `listEnabledFeishuAccounts(cfg)` | 列出所有已启用且已配置的账号 |
| `resolveDefaultFeishuAccountId(cfg)` | 解析默认账号 ID |

**账号合并策略：**
1. 顶层 `channels.feishu.*` 作为基础默认值
2. `accounts.<id>.*` 覆盖顶层同名字段
3. 排除 `defaultAccount` 字段
4. `tools` 使用嵌套对象合并

---

## 5. SDK 客户端封装

### 客户端创建：`client.ts`

使用 **`@larksuiteoapi/node-sdk`**（Lark 官方 Node SDK v1.66.0）：

```typescript
// 创建 API 客户端（带缓存）
const client = new Lark.Client({
  appId,
  appSecret,
  appType: Lark.AppType.SelfBuild,    // 自建应用
  domain: resolveDomain(domain),       // Feishu / Lark / 私有部署 URL
  httpInstance: createTimeoutHttpInstance(timeoutMs),
});

// 创建 WebSocket 客户端
const wsClient = new Lark.WSClient({
  appId, appSecret,
  domain: resolveDomain(domain),
  onError, onReady, onReconnected, onReconnecting,
  wsConfig: { PingInterval: 30, PingTimeout: 3 },
});
```

**关键特性：**
- 客户端缓存（以 accountId 为 key），相同凭据复用
- 自定义 User-Agent：`openclaw-feishu-builtin/<version>/<platform>`
- HTTP 超时可配置（默认 30s，最大 300s，环境变量 `FEISHU_HTTP_TIMEOUT_MS`）
- 域名支持：`feishu`（飞书）、`lark`（Lark）、自定义 URL（私有部署）

**事件分发器：**
```typescript
const dispatcher = new Lark.EventDispatcher({
  encryptKey: account.encryptKey,
  verificationToken: account.verificationToken,
});
```

---

## 6. 传输层：WebSocket 与 Webhook

### 传输管理：`monitor.transport.ts`

**两种连接模式：**

#### WebSocket（默认，推荐）

```
Feishu Server ←──WSS──→ OpenClaw Gateway
                          │
                   WSClient (Lark SDK)
                          │
                  EventDispatcher
                          │
              message/reaction/cardAction handlers
```

- 自动重连（指数退避：1s → 30s max）
- 心跳：30s PingInterval，3s PingTimeout
- 支持 HTTP 代理（通过 `resolveAmbientNodeProxyAgent`）

#### Webhook

```
Feishu Server ──HTTP POST──→ OpenClaw Gateway (HTTP Server)
                               │
                       签名验证 (HMAC-SHA256)
                               │
                       EventDispatcher
```

- HTTP Server 绑定到配置的 host:port
- 签名头验证：`x-lark-request-timestamp`、`x-lark-request-nonce`、`x-lark-signature`
- 请求体大小限制：可配置 `FEISHU_WEBHOOK_MAX_BODY_BYTES`
- 速率限制：内置 rate limiter
- 需要额外配置 `encryptKey` 和 `verificationToken`

### 监控启动：`monitor.ts` / `monitor.account.ts`

```
monitorFeishuProvider()
    │
    ├── 单账号？→ monitorSingleAccount()
    └── 多账号？→ 并行 monitorSingleAccount() × N
                      │
                  fetchBotIdentityForMonitor()  ← 获取 bot 身份
                      │
                  ├── WebSocket：monitorWebSocket()
                  └── Webhook：monitorWebhook()
                      │
                  handleFeishuMessage()  ← 事件回调
```

---

## 7. 消息入口处理

### 核心流程：`bot.ts` → `monitor.message-handler.ts`

```
Feishu Event (im.message.receive_v1)
    │
    ▼
parseFeishuMessageEvent()          ← 解析事件 payload
    │
    ▼
dedup check                        ← 消息去重（Redis/SQLite）
    │
    ▼
parseMessageContent()              ← 提取文本/媒体/mention
    │
    ▼
checkBotMentioned()                ← 群聊 @ 检测
    │
    ▼
resolveFeishuGroupSession()        ← 确定会话 Key
    │
    ▼
resolveFeishuDmIngressAccess()     ← DM 访问控制
resolveFeishuGroupIngressAccess()  ← 群聊访问控制
    │
    ▼
maybeCreateDynamicAgent()          ← 动态 Agent 创建（DM）
    │
    ▼
resolveConfiguredBindingRoute()     ← Agent 路由
    │
    ▼
createFeishuReplyDispatcher()      ← 创建回复分发器
    │
    ▼
replyDispatcher.dispatch()         ← 触发 Agent 推理 + 回复
```

### 消息事件结构：`event-types.ts`

```typescript
type FeishuMessageEvent = {
  sender: {
    sender_id: {
      open_id?: string;
      user_id?: string;
      union_id?: string;
    };
  };
  message: {
    message_id: string;
    chat_id: string;
    chat_type: "p2p" | "group" | "topic_group" | "private";
    message_type: string;      // text, post, image, audio, file, sticker...
    content: string;           // JSON string
    root_id?: string;
    parent_id?: string;
    thread_id?: string;        // "omt_*" for native topics
    mentions?: Array<{
      key: string;
      id: { open_id?: string; user_id?: string };
      name: string;
    }>;
  };
};
```

### 输入能力（Receive）

| 类型 | 支持 | 处理方式 |
|------|------|---------|
| Text | ✅ | 直接提取文本 |
| Rich Text (Post) | ✅ | `post.ts` 解析为结构化内容 |
| Images | ✅ | 提取 `image_key`，下载为附件 |
| Audio/Voice | ✅ | 提取 `file_key` → ASR 转录（可配置）|
| Files | ✅ | 提取 `file_key`，下载为附件 |
| Video/Media | ✅ | 同文件处理 |
| Stickers | ✅ | 提取为表情标签 |

### Mention 处理

- 自动检测并剥离 `<at user_id="...">...</at>` 标签
- 区分 `@bot` 和 `@其他用户`
- `@all` / `@_all` 不作为 bot mention
- 支持 mention forward（转发给被 @ 的用户）

---

## 8. 消息发送

### 发送管线：`send.ts` + `outbound.ts`

```
sendMessageFeishu()
    │
    ├── 文本 → client.im.message.create()
    │           msg_type: "text"
    │
    ├── 富文本 → client.im.message.create()
    │           msg_type: "post"
    │
    ├── Markdown Card → client.im.message.create()
    │           msg_type: "interactive"
    │           card template: "markdown"
    │
    ├── 结构化 Card → client.im.message.create()
    │           msg_type: "interactive"
    │
    ├── 媒体 → client.im.message.create()
    │           msg_type: "image" / "file" / "audio" / "media"
    │           先上传到 Feishu，再发送
    │
    └── 回复模式：
        ├── reply: 内联回复（replyToMessageId）
        └── replyInThread: 话题回复（reply_in_thread: true）
```

### 输出能力（Send）

| 类型 | 支持 | 备注 |
|------|------|------|
| Text | ✅ | 自动分块（默认 2000 字符/chunk） |
| Image | ✅ | 支持 URL 和本地文件 |
| File | ✅ | 通用文件附件 |
| Audio | ✅ | 原生 Ogg/Opus；MP3/WAV/M4A 通过 ffmpeg 转码 |
| Video/Media | ✅ | 作为媒体文件发送 |
| Interactive Card | ✅ | 支持 Markdown Card 和结构化 Card |
| Rich Text (Post) | ⚠️ | 有限支持，不含全部 authoring 能力 |
| TTS Voice Note | ✅ | 文本 → 语音 → Opus 音频气泡 |

### 文本分块

- 默认 `textChunkLimit: 2000`（飞书单条消息限制）
- `chunkMode`：`"length"`（按字符数）或 `"newline"`（按换行）
- Markdown 表格自动转换为 ASCII 表格（可选 native）

### 音频转码

飞书原生音频气泡需要 **Ogg/Opus 格式**：
- `.opus` / `.ogg` 直接发送
- MP3/WAV/M4A 通过 `ffmpeg` 转为 48kHz Ogg/Opus
- ffmpeg 不可用时降级为文件附件

---

## 9. 会话与路由

### 会话 ID 构建：`conversation-id.ts`

```
会话 Scope 体系：
┌──────────────────────────────────────────────────────┐
│ DM:    {open_id}                                     │
│                                                       │
│ Group:                                                │
│   scope=group:             {chat_id}                  │
│   scope=group_sender:      {chat_id}:sender:{open_id} │
│   scope=group_topic:       {chat_id}:topic:{thread}   │
│   scope=group_topic_sender:{chat_id}:topic:{t}:sender:{s}
└──────────────────────────────────────────────────────┘
```

**Feishu Topic Group 特殊处理：**
- 原生话题组使用 `thread_id`（`omt_*` 格式）作为 canonical topic session key
- 普通群组的回复线程使用 `root_id`（`om_*` 格式）
- Session Key 格式：`agent:<agentId>:feishu:<scope>:<chatId>[:topic:<t>][:sender:<s>]`

### Agent 路由

通过 `bindings` 将消息路由到不同的 Agent：

```json5
{
  bindings: [
    {
      agentId: "agent-a",
      match: {
        channel: "feishu",
        peer: { kind: "direct", id: "ou_xxx" }
      }
    },
    {
      agentId: "agent-b",
      match: {
        channel: "feishu",
        peer: { kind: "group", id: "oc_zzz" }
      }
    }
  ]
}
```

### ACP 会话绑定

支持通过 `/acp spawn <agent> --thread here` 命令在对话中启动 ACP 会话：
- DM 直接绑定
- 群聊支持 Topic 级别的 ACP 绑定
- Persistent ACP 通过 `bindings` 预配置

---

## 10. Access Control（访问控制）

### DM 访问策略：`dmPolicy`

| 策略 | 行为 |
|------|------|
| `pairing` | 未知用户收到配对码，通过 CLI 审批（默认） |
| `allowlist` | 仅 `allowFrom` 列表中的用户可以对话 |
| `open` | 公开 DM（需 `allowFrom: ["*"]`） |
| `disabled` | 禁用所有 DM |

### Group 访问策略：`groupPolicy`

| 策略 | 行为 |
|------|------|
| `open` | 响应所有群组消息（需 @） |
| `allowlist` | 仅 `groupAllowFrom` 中的群组 |
| `disabled` | 禁用所有群组消息 |

### Per-Group 配置

```json5
{
  groups: {
    "oc_xxx": {
      requireMention: false,     // 不需要 @
      allowFrom: ["ou_user1"],   // 限制群内发送者
      tools: { allow: ["feishu_doc"] },  // 工具白名单
      skills: ["feishu-doc"],    // 技能白名单
      groupSessionScope: "group_sender",  // 会话隔离
    }
  }
}
```

### 实施：`policy.ts`

- `resolveFeishuDmIngressAccess()` — DM 入口控制
- `resolveFeishuGroupConversationIngressAccess()` — 群聊入口控制
- `resolveFeishuGroupSenderActivationIngressAccess()` — 群内发送者激活控制
- `resolveFeishuGroupToolPolicy()` — 群组工具策略

---

## 11. Agent 工具系统

所有工具通过 `api.registerTool()` 注册，以 Tool Name + Action 参数模式暴露给 AI。

### 工具注册入口：`index.ts`

```typescript
registerFull(api) {
  registerFeishuSubagentHooks(api);
  registerFeishuDocTools(api);       // feishu_doc
  registerFeishuChatTools(api);      // feishu_chat
  registerFeishuWikiTools(api);      // feishu_wiki
  registerFeishuDriveTools(api);     // feishu_drive
  registerFeishuPermTools(api);      // feishu_perm
  registerFeishuBitableTools(api);   // feishu_bitable_*
}
```

### 工具详解

#### feishu_chat（聊天操作）

| Action | 功能 | 参数 |
|--------|------|------|
| `info` | 获取群聊信息 | `chat_id` |
| `members` | 获取群成员列表 | `chat_id`, `page_size`, `page_token`, `member_id_type` |
| `member_info` | 获取单个用户信息 | `member_id`, `member_id_type` |

#### feishu_doc（文档操作）

| Action | 功能 | 权限要求 |
|--------|------|---------|
| `read` | 读取文档纯文本 | `docx:document:readonly` |
| `list_blocks` | 获取 Block 结构 | `docx:document:readonly` |
| `get_block` | 获取单个 Block | `docx:document:readonly` |
| `write` | 替换整个文档（Markdown→Blocks） | `docx:document` |
| `append` | 追加 Markdown 到末尾 | `docx:document` |
| `create` | 创建新文档 | `docx:document` |
| `update_block` | 更新单个 Block 文本 | `docx:document` |
| `delete_block` | 删除 Block | `docx:document` |
| `create_table` | 创建表格 Block | `docx:document` |
| `write_table_cells` | 写入表格单元格 | `docx:document` |
| `create_table_with_values` | 一步创建表格+写入 | `docx:document` |
| `upload_image` | 上传图片到文档 | `docx:document`, `drive:drive` |
| `upload_file` | 上传文件附件 | `docx:document`, `drive:drive` |

**Markdown→Blocks 转换链：**
1. 用户写 Markdown
2. 按 `\n\n` 分段
3. 每段识别类型（heading/list/code/quote/paragraph/image）
4. 创建对应的 Block 结构（Text/Heading/Bullet/Ordered/Code/Quote）
5. 批量插入（`insertBlocksInBatches`，batch size = 50）

**表格操作：** `docx-table-ops.ts`
- `insertTableRow` / `insertTableColumn`
- `deleteTableRows` / `deleteTableColumns`
- `mergeTableCells`

#### feishu_wiki（知识库操作）

| Action | 功能 |
|--------|------|
| `spaces` | 列出所有可访问的知识空间 |
| `nodes` | 列出空间/父节点下的节点 |
| `get` | 获取节点详情（包含 `obj_token`） |
| `create` | 创建节点（支持 doc/sheet/bitable 等类型） |
| `move` | 移动节点到目标位置 |
| `rename` | 重命名节点 |

**与 feishu_doc 的关联：** Wiki 节点内容是文档，获取 `obj_token` 后通过 `feishu_doc` 读写。

#### feishu_drive（云盘操作）

| Action | 功能 |
|--------|------|
| `list` | 列出文件夹内容 |
| `info` | 获取文件信息 |
| `create_folder` | 创建文件夹 |
| `move` | 移动文件 |
| `delete` | 删除文件 |

**已知限制：** Bot 没有根目录。需要用户手动创建文件夹并分享给 Bot。

#### feishu_bitable（多维表格操作）

| Action | 功能 |
|--------|------|
| `create_app` | 创建 Bitable |
| `get_meta` | 获取表格元信息 |
| `list_fields` | 列出字段 |
| `create_field` | 创建字段（支持 Text/Number/Select/DateTime 等 20+ 类型） |
| `list_records` | 列出记录（支持筛选/排序/分页） |
| `get_record` | 获取单条记录 |
| `create_record` | 创建记录 |
| `update_record` | 更新记录 |

#### feishu_perm（权限管理）⚠️ 默认关闭

| Action | 功能 |
|--------|------|
| `list` | 列出协作者 |
| `add` | 添加协作者 |
| `remove` | 移除协作者 |

支持权限级别：`view` / `edit` / `full_access`

### 工具配置：`tools-config.ts`

每个工具类别可通过 `channels.feishu.tools.<tool>` 独立开关：

```yaml
tools:
  doc: true      # 文档工具（默认：true）
  chat: true     # 聊天工具（默认：true）
  wiki: true     # 百科工具（默认：true，依赖 doc）
  drive: true    # 云盘工具（默认：true）
  perm: false    # 权限工具（默认：false，敏感操作）
  bitable: true  # 多维表格工具（默认：true）
  scopes: true   # 权限范围诊断（默认：true）
```

### 多账号工具路由：`tool-account.ts`

当存在多个账号时，工具调用通过 `resolveFeishuToolAccount()` 确定使用哪个账号的凭据。

---

## 12. Skills（技能指引）

Skills 是给 AI 的提示词指令，告诉 AI 如何使用对应的工具。位于 `extensions/feishu/skills/`。

| Skill | 路径 | 功能 |
|-------|------|------|
| `feishu-doc` | `skills/feishu-doc/SKILL.md` | 文档操作完整指引 |
| `feishu-wiki` | `skills/feishu-wiki/SKILL.md` | 知识库操作指引 |
| `feishu-drive` | `skills/feishu-drive/SKILL.md` | 云盘操作指引 |
| `feishu-perm` | `skills/feishu-perm/SKILL.md` | 权限管理指引 |

**在 `openclaw.plugin.json` 中声明：**
```json
{
  "skills": ["./skills"]
}
```

---

## 13. 动态 Agent 创建

### 实现：`dynamic-agent.ts`

**触发时机：** 当一个未知用户首次向 Bot 发送 DM 时。

**创建流程：**
```
1. 检查是否已有该用户的 binding → 跳过
2. 检查 maxAgents 限制 → 超限跳过
3. 生成 agentId = "feishu-{open_id}"
4. 创建 workspace 目录（~/.openclaw/workspace-feishu-{open_id}）
5. 创建 agentDir 目录（~/.openclaw/agents/feishu-{open_id}/agent）
6. 更新配置文件：
   - agents.list 添加新 agent
   - bindings 添加 DM 路由
7. 通过 runtime.config.replaceConfigFile() 持久化
```

**配置模板变量：**
- `{agentId}` → `feishu-ou_xxxxxx`
- `{userId}` → `ou_xxxxxx`

---

## 14. 流式卡片输出

### 实现：`streaming-card.ts`

飞书支持通过 **Card Kit Streaming API** 实现渐进式卡片更新：

```
1. POST /open-apis/cardkit/v1/cards              ← 创建卡片（获得 card_id）
2. PATCH /open-apis/cardkit/v1/cards/{card_id}   ← 增量更新文本
3. DELETE /open-apis/cardkit/v1/cards/{card_id}  ← 结束流式（可选）
```

**Token 管理：**
- Token 缓存（按 domain + appId）
- 使用 `tenant_access_token`
- 默认 2 小时有效期

**节流策略：**
- `STREAMING_UPDATE_THROTTLE_MS = 160ms` — 最小更新间隔
- `STREAMING_SIGNIFICANT_DELTA_CHARS = 18` — 至少 18 个新字符才推送更新

**使用方式：**
```json5
{
  channels: {
    feishu: {
      streaming: true,       // 启用流式卡片（默认）
      blockStreaming: true,  // 完成块即时推送
    }
  }
}
```

关闭 `streaming` 则一次性发送完整回复。

---

## 15. 交互式卡片（Presentation）

### 实现：`presentation-card.ts` + `outbound.ts`

**卡片渲染模式：**
| 模式 | 行为 |
|------|------|
| `auto` | 自动检测（包含代码块/表格时用 Card） |
| `raw` | 始终纯文本 |
| `card` | 始终用卡片 |

**卡片能力限制：**
- 最大 Action 数：20
- 每行最多：5 个 Action
- Label 最大：40 字符
- Value 最大：1024 bytes
- Card Template 颜色：12 种可选（blue/green/red/orange/purple/indigo/...）

**结构化卡片：**
```typescript
// 构建卡片 JSON
buildFeishuPresentationCard({ presentation, fallbackText })
// 渲染按钮
mapFeishuButtonType(style) → "primary" | "danger" | "default"
// URL 按钮
resolveFeishuButtonUrl(button) → 仅 http/https
// 交互信封
createFeishuCardInteractionEnvelope()
```

---

## 16. 消息去重与顺序

### 去重：`dedup.ts` + `dedupe-key.ts`

- 使用 `message_id` + namespace 作为去重 key
- 支持持久化存储（SQLite/Redis）
- 启动时 warmup 已有状态

### 顺序处理：`sequential-queue.ts` + `sequential-key.ts`

- 按 `(accountId, chatId)` 生成顺序 key
- 同一 chat 的消息串行处理
- 使用 AbortSignal 支持取消

### 处理声明：`processing-claims.ts`

- 防止同一消息被并发处理
- `tryBeginFeishuMessageProcessing()` / `releaseFeishuMessageProcessing()` 模式

---

## 17. Reactions（表情回应）

### 实现：`reactions.ts`

| 功能 | API |
|------|-----|
| 添加 Reaction | `client.im.messageReaction.create()` |
| 删除 Reaction | `client.im.messageReaction.delete()` |
| 列出 Reactions | `client.im.messageReaction.list()` |

**Reaction 通知：** `monitor.account.ts`
- 监听到 Reaction 事件时，验证是否为 Bot 消息
- `reactionNotifications: "own"`（默认）只在自己消息被回应时通知
- `reactionNotifications: "all"` 所有反应都通知
- `reactionNotifications: "off"` 关闭

**Typing Indicator：**
- `typingIndicator: true`（默认）— 在生成回复期间发送 "Typing" reaction

---

## 18. 安全与密钥管理

### 密钥合约：`secret-contract.ts`

飞书需要以下密钥：
| 密钥 | 用途 | 必需 |
|------|------|------|
| `appId` | API 调用标识 | ✅ |
| `appSecret` | API 调用签名 | ✅ |
| `encryptKey` | Webhook 消息解密 | Webhook 模式 |
| `verificationToken` | Webhook 请求验证 | Webhook 模式 |

### 安全审计：`security-audit.ts`

- `collectFeishuSecurityAuditFindings()` — 检查配置安全性
- Core 通过 `src/plugin-sdk/feishu-security.ts` 的 facade loader 调用

### Webhook 安全校验：`monitor.transport.ts`

```typescript
// HMAC-SHA256 签名验证
function isFeishuWebhookSignatureValid({ headers, rawBody, encryptKey }) {
  const timestamp = headers["x-lark-request-timestamp"];
  const nonce = headers["x-lark-request-nonce"];
  const signature = headers["x-lark-signature"];
  // 计算 HMAC-SHA256(timestamp + nonce + encryptKey, rawBody)
}
```

---

## 19. Setup / 初始化

### 设置向导：`setup-surface.ts`

**两种设置模式：**
1. **QR Code 设置** — 调用飞书 OpenAPI 创建应用，生成 QR 码
2. **手动设置** — 用户在飞书开放平台创建自建应用，粘贴 App ID + App Secret

**设置流程：**
```
openclaw channels login --channel feishu
  ├── 选择账号（default 或命名账号）
  ├── 选择设置方式（QR / 手动）
  ├── 输入凭据
  ├── 验证凭据（probe）
  └── 写入 openclaw.json
```

### 设置适配器：`setup-core.ts`

```typescript
export const feishuSetupAdapter: ChannelSetupAdapter = {
  resolveAccountId: ({ cfg, accountId }) => ...,
  applyAccountConfig: ({ cfg, accountId }) => ...,  // 启用账号
};
```

### 平台配置要求

飞书开放平台 / Lark Developer 需要配置：
- **事件订阅**：`im.message.receive_v1`
- **连接方式**：WebSocket（持久连接）
- **权限范围**：
  - `im:message` — 消息读写
  - `im:message:send_as_bot` — 以 Bot 身份发送
  - `im:chat` — 群聊信息
  - `docx:document` / `docx:document:readonly` — 文档读写
  - `drive:drive` — 云盘
  - `wiki:wiki` — 知识库
  - 等（按需配置）

---

## 20. Doctor / 诊断

### 实现：`doctor.ts`

`feishuDoctor` 作为 channel plugin 的 doctor 钩子，处理：
- 旧格式配置迁移
- 凭据验证
- 状态修复

### 探测：`probe.ts`

`probeFeishu(account)` — 验证账号可用性：
- 尝试获取 `tenant_access_token`
- 获取 Bot 身份信息（open_id, name）

---

## 21. 多账号支持

一个 OpenClaw 实例可以连接多个飞书 Bot：

```json5
{
  channels: {
    feishu: {
      defaultAccount: "main",
      accounts: {
        main: {
          appId: "cli_xxx",
          appSecret: "xxx",
          name: "Primary bot",
          tts: { providers: { openai: { voice: "shimmer" } } }
        },
        backup: {
          appId: "cli_yyy",
          appSecret: "yyy",
          name: "Backup bot",
          enabled: false
        }
      }
    }
  }
}
```

**关键特性：**
- 每个账号独立的凭据、域名、连接模式
- 每个账号独立的工具开关和 TTS 配置
- `defaultAccount` 指定默认出站账号
- 账号可以独立启用/禁用
- 出站路由时可以指定 `accountId`

---

## 22. 依赖与 SDK 版本

### package.json 关键依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| `@larksuiteoapi/node-sdk` | 1.66.0 | 飞书官方 Node SDK |
| `zod` | 4.4.3 | 配置 Schema 验证 |
| `typebox` | 1.1.39 | 工具参数 Schema（替代 JSON Schema） |

### 运行时要求

- Node.js 22.19+（推荐 Node 24）
- 音频转码需要 `ffmpeg`（可选，用于 TTS 语音气泡）

### 对 Core 的依赖

插件通过 `openclaw/plugin-sdk/*` 使用以下关键模块：

| SDK 子路径 | 用途 |
|-----------|------|
| `channel-entry-contract` | `defineBundledChannelEntry` + `loadBundledEntryExportSync` |
| `channel-core` | `createChatChannelPlugin` |
| `channel-contract` | ChannelPlugin 类型定义 |
| `channel-outbound` | 消息发送委托 |
| `channel-config-helpers` | 配置适配器 |
| `channel-pairing` | DM 配对流程 |
| `channel-policy` | 访问策略警告 |
| `channel-status` | 状态探测 |
| `account-resolution` | 账号解析工具 |
| `account-helpers` | 账号描述格式化 |
| `conversation-runtime` | 会话绑定服务 |
| `interactive-runtime` | 交互式卡片渲染 |
| `reply-payload` | 回复负载组装 |
| `text-chunking` | 文本分块 |
| `setup` | 设置适配器类型 |
| `setup-runtime` | 设置向导（Clack 交互） |
| `config-mutation` | 配置文件写入 |
| `extension-shared` | 包版本读取、代理解析 |
| `routing` | Agent 路由 |
| `security-runtime` | 文件/权限检查 |

---

## 23. 迁移清单

如果你要将飞书集成迁移到其他 Agent 项目，以下是需要实现的核心模块清单：

### 必须（最小可用）

- [ ] **SDK 客户端封装**：包装 `@larksuiteoapi/node-sdk`，支持 API Client + WSClient + EventDispatcher
- [ ] **凭据管理**：App ID + App Secret 的配置/解析/安全存储
- [ ] **WebSocket 连接**：建立 WSS 连接，事件监听，自动重连，心跳维持
- [ ] **消息接收/解析**：`im.message.receive_v1` 事件 → 提取 sender/message/chat 信息
- [ ] **消息发送**：`client.im.message.create()` + `client.im.message.reply()` 支持 text/post/interactive
- [ ] **Mention 处理**：检测 @bot，剥离 XML mention 标签
- [ ] **会话路由**：DM vs Group 区分，基础 Session Key 生成
- [ ] **访问控制**：DM Policy（pairing/allowlist/open），Group Policy（open/allowlist/disabled）
- [ ] **消息去重**：基于 message_id 的去重缓存

### 推荐（增强体验）

- [ ] **消息去重持久化**：SQLite/Redis 存储已处理消息 ID
- [ ] **消息顺序处理**：同 Chat 的消息串行队列
- [ ] **富文本解析**：Post/Image/Audio/File/Sticker 类型的提取和转换
- [ ] **媒体下载/上传**：飞书 file_key → 本地文件；本地文件 → 飞书 media
- [ ] **流式输出**：Card Kit Streaming API 实现渐进式文本显示
- [ ] **交互式卡片**：按钮/链接/确认框的 JSON 构建和渲染
- [ ] **表情反应**：添加/删除/列出 Reactions
- [ ] **消息编辑**：编辑已发送的消息
- [ ] **消息引用**：回复特定消息（inline reply + thread reply）
- [ ] **Typing Indicator**：发送 Typing reaction
- [ ] **发送者名称解析**：通过 Feishu API 获取用户显示名
- [ ] **配对流程**：未知用户 DM → 配对码 → 审批

### 可选（完整能力）

- [ ] **文档操作工具**：读/写/创建/追加/表格/图片/文件附件（对应 `feishu_doc`）
- [ ] **知识库工具**：空间列表/节点浏览/创建/移动/重命名（对应 `feishu_wiki`）
- [ ] **云盘工具**：文件列表/信息/删除/移动/创建文件夹（对应 `feishu_drive`）
- [ ] **聊天工具**：群聊信息/成员列表/用户信息（对应 `feishu_chat`）
- [ ] **多维表格工具**：CRUD 操作/字段管理（对应 `feishu_bitable`）
- [ ] **权限管理工具**：协作者增删查（对应 `feishu_perm`）
- [ ] **Webhook 模式**：HTTP Server + 签名验证 + 消息解密
- [ ] **多账号支持**：多个 Bot 凭据共存
- [ ] **动态 Agent 创建**：每个 DM 用户自动创建独立 Agent 实例
- [ ] **Topic Session 隔离**：飞书话题组的独立会话
- [ ] **群组会话范围**：group/group_sender/group_topic/group_topic_sender
- [ ] **TTS 语音气泡**：文本转语音 → Ogg/Opus 转码 → 原生 Audio 消息
- [ ] **音频转录**：飞书语音消息 → ASR → 文本
- [ ] **安全审计**：配置安全检查和密钥泄露检测
- [ ] **设置向导**：QR Code 自动创建 + 手动配置
- [ ] **Doctor 诊断**：配置迁移和健康检查
- [ ] **消息引用历史**：获取线程消息历史
- [ ] **Mention Forward**：消息中 @其他用户 的转发机制
- [ ] **ACP 会话绑定**：文本命令驱动的 ACP 创建和绑定
- [ ] **Comment 评论处理**：文档评论的通知和回复

### 不需要迁移的部分

- OpenClaw 特定的 Plugin SDK 框架（`defineBundledChannelEntry`, `createChatChannelPlugin`）——需替换为你项目自己的插件系统
- PluginRuntime Store 模式——根据你的架构调整
- SecretRef 机制（env/file/exec）——可简化为直接配置或环境变量
- Session Binding Service——根据你的会话管理调整
- Codex/ACP 相关功能——除非你也需要 ACP 协议

---

## 附录：文件索引

### 入口和注册
| 文件 | 用途 |
|------|------|
| `index.ts` | 插件入口，注册所有子系统和工具 |
| `api.ts` | 公共 API 导出 |
| `channel-plugin-api.ts` | 导出 `feishuPlugin` |
| `runtime-api.ts` | 运行时类型定义 + `setFeishuRuntime` |
| `runtime-setter-api.ts` | Runtime setter 导出 |
| `channel-entry.ts` | Channel Entry 定义 |
| `openclaw.plugin.json` | 插件元数据（工具声明、配置 Schema、环境变量等） |
| `package.json` | 包依赖和 openclaw 配置块 |

### 运行时核心
| 文件 | 用途 |
|------|------|
| `src/runtime.ts` | PluginRuntime Store |
| `src/channel.ts` | 核心 ChannelPlugin 定义（~1400 行） |
| `src/channel.runtime.ts` | 运行时实现 barrel |
| `src/client.ts` | Feishu SDK 客户端封装 |
| `src/accounts.ts` | 账号解析和管理 |
| `src/types.ts` | 核心类型定义 |
| `src/config-schema.ts` | Zod 配置 Schema |

### 传输层
| 文件 | 用途 |
|------|------|
| `src/monitor.ts` | 监控入口 |
| `src/monitor.account.ts` | 单账号监控 |
| `src/monitor.transport.ts` | WebSocket + Webhook 传输 |
| `src/monitor.state.ts` | 监控状态管理 |
| `src/monitor.startup.ts` | 启动时 Bot 身份获取 |

### 消息处理
| 文件 | 用途 |
|------|------|
| `src/bot.ts` | 消息处理和 Agent 触发 |
| `src/bot-content.ts` | 消息内容解析 |
| `src/bot-sender-name.ts` | 发送者名称解析 |
| `src/monitor.message-handler.ts` | 消息处理器 |
| `src/reply-dispatcher.ts` | 回复分发器 |
| `src/send.ts` | 消息发送 |
| `src/outbound.ts` | 出站消息适配器 |
| `src/media.ts` | 媒体上传/下载 |
| `src/post.ts` | Rich Text (Post) 解析 |
| `src/mention.ts` | Mention 处理 |
| `src/card-action.ts` | 卡片动作处理 |
| `src/card-interaction.ts` | 卡片交互信封 |
| `src/streaming-card.ts` | 流式卡片 |
| `src/presentation-card.ts` | 交互式卡片渲染 |

### 工具（Agent Tools）
| 文件 | 用途 |
|------|------|
| `src/chat.ts` | 聊天工具 |
| `src/chat-schema.ts` | 聊天工具 Schema |
| `src/docx.ts` | 文档工具 |
| `src/doc-schema.ts` | 文档工具 Schema |
| `src/docx-batch-insert.ts` | 文档批量插入 |
| `src/docx-table-ops.ts` | 文档表格操作 |
| `src/docx-color-text.ts` | 文档彩色文本 |
| `src/docx-types.ts` | 文档 Block 类型 |
| `src/wiki.ts` | 百科工具 |
| `src/wiki-schema.ts` | 百科工具 Schema |
| `src/drive.ts` | 云盘工具 |
| `src/drive-schema.ts` | 云盘工具 Schema |
| `src/bitable.ts` | 多维表格工具 |
| `src/perm.ts` | 权限管理工具 |
| `src/perm-schema.ts` | 权限工具 Schema |
| `src/tool-account.ts` | 工具账号路由 |
| `src/tool-result.ts` | 工具结果格式化 |
| `src/tools-config.ts` | 工具开关配置 |

### 会话和路由
| 文件 | 用途 |
|------|------|
| `src/conversation-id.ts` | 会话 ID 构建 |
| `src/session-conversation.ts` | 会话解析 |
| `src/session-route.ts` | 出站会话路由 |
| `src/dynamic-agent.ts` | 动态 Agent 创建 |
| `src/thread-bindings.ts` | 线程绑定管理 |

### 安全和访问控制
| 文件 | 用途 |
|------|------|
| `src/policy.ts` | 访问控制策略 |
| `src/secret-contract.ts` | 密钥合约 |
| `src/secret-input.ts` | 密钥输入 Schema |
| `src/security-audit.ts` | 安全审计 |
| `src/security-audit-shared.ts` | 安全审计共享类型 |
| `src/approval-auth.ts` | 审批授权 |

### 辅助
| 文件 | 用途 |
|------|------|
| `src/dedup.ts` | 消息去重 |
| `src/dedupe-key.ts` | 去重 Key 生成 |
| `src/sequential-queue.ts` | 顺序队列 |
| `src/sequential-key.ts` | 顺序 Key 生成 |
| `src/processing-claims.ts` | 处理声明 |
| `src/reactions.ts` | 表情回应 |
| `src/pins.ts` | 消息置顶 |
| `src/comment-reaction.ts` | 评论 Reaction |
| `src/comment-target.ts` | 评论目标解析 |
| `src/comment-dispatcher.ts` | 评论分发器 |
| `src/comment-handler.ts` | 评论处理器 |
| `src/targets.ts` | 目标 ID 格式化 |
| `src/setup-core.ts` | 设置核心 |
| `src/setup-surface.ts` | 设置向导界面 |
| `src/doctor.ts` | 诊断和修复 |
| `src/probe.ts` | 账号探测 |
| `src/qr-terminal.ts` | 终端 QR 码显示 |
| `src/directory.ts` | 通讯录（动态） |
| `src/directory.static.ts` | 通讯录（静态缓存） |
| `src/external-keys.ts` | 外部 Key 管理 |
| `src/event-types.ts` | 事件类型定义 |
| `src/async.ts` | 异步辅助（超时竞速、延迟） |
| `src/send-result.ts` | 发送结果格式化 |
| `src/send-target.ts` | 发送目标解析 |
| `src/reasoning-preview.ts` | Reasoning 预览 |
