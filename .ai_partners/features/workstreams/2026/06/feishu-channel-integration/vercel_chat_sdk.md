# 使用 Vercel Chat SDK 接入飞书

## Vercel Chat SDK 是什么

[Vercel Chat SDK](https://chat-sdk.dev) 是一个面向多平台的聊天机器人开发框架，提供统一的 Channel、Thread、Message 抽象。基于该 SDK，开发者可以用同一套业务代码同时支持 Slack、Microsoft Teams、Discord、Telegram、WhatsApp、GitHub、Linear 等多个 IM 与协作平台。

它的核心理念是 **Channel 抽象**：把平台特定的鉴权、消息收发、状态管理统一封装在适配器层之下，业务代码只关心 `Channel`/ `Thread`/ `Message`。新平台的接入由对应的"适配器（adapter）"承担。

## 飞书官方适配器

飞书已经为 Vercel Chat SDK 发布了官方适配器 [@larksuite/vercel-chat-adapter](https://www.npmjs.com/package/@larksuite/vercel-chat-adapter)，让 Chat SDK 上构建的产品可以无缝接入飞书。

适配器支持的能力：

- 原生 cardkit 卡片流式（typewriter，非 post + edit 轮询）
- 完整交互卡片（按钮、链接、菜单、字段、图片）
- 消息编辑 / 删除 / 表情回复
- 历史消息拉取（覆盖飞书 20+ 种原生消息类型）
- 群聊 / 私聊识别 及 @ 提及处理
- WebSocket 长连接传输
- `registerLarkApp` 扫码授权一键创建应用

这个适配器基于飞书 Channel SDK 构建（详见下方[适配器实现原理](https://open.larkoffice.com/document/mcp_open_tools/integrating-agents-with-feishu/vercel-chat-sdk-lark-message-publish#16b76461)章节）。

## 快速开始

### 创建飞书应用

#### 方式一：扫码授权（推荐）

调用 `registerLarkApp` 触发飞书官方扫码流程，生成一次性链接，用户扫码后自动返回 `client_id` / `client_secret`，并预配好本适配器所需的权限与事件订阅。

```TypeScript
import { registerLarkApp } from "@larksuite/vercel-chat-adapter";
import qrcode from "qrcode-terminal"; // pnpm add -D qrcode-terminal

const { client_id, client_secret } = await registerLarkApp({
  onQRCodeReady: ({ url }) => {
    console.log("使用飞书 / Lark 手机端扫码：");
    qrcode.generate(url, { small: true });
  },
  onStatusChange: ({ status }) => console.log("授权状态：", status),
});

console.log("LARK_APP_ID=", client_id);
console.log("LARK_APP_SECRET=", client_secret);

```

只需运行一次。把返回的 `client_id` / `client_secret` 持久化保存，后续启动通过 `LARK_APP_ID` / `LARK_APP_SECRET` 环境变量传入即可。

#### 方式二：开发者后台手动创建

在 [open.larksuite.com/app](https://open.larksuite.com/app)（Lark）或 [open.feishu.cn/app](https://open.feishu.cn/app)（飞书）创建应用，获取 `client_id` / `client_secret`。

### 安装适配器

```Bash
pnpm add @larksuite/vercel-chat-adapter chat @chat-adapter/state-memory
```

### 最小可运行示例

```TypeScript
import { Chat } from "chat";
import { createMemoryState } from "@chat-adapter/state-memory";
import { createLarkAdapter } from "@larksuite/vercel-chat-adapter";

const bot = new Chat({
  userName: "mybot",
  adapters: {
    lark: createLarkAdapter({
      appId: process.env.LARK_APP_ID!,
      appSecret: process.env.LARK_APP_SECRET!,
    }),
  },
  state: createMemoryState(),
});

bot.onNewMention(async (thread, message) => {
  await thread.subscribe();
  await thread.post(`收到：${message.text}`);
});

bot.onDirectMessage(async (thread, message) => {
  await thread.post(`收到私信：${message.text}`);
});

await bot.initialize();

```

`bot.initialize()` 会建立 WebSocket 长连接并保持运行，Node.js 进程会一直运行直到 `bot.shutdown()` 被调用。

### 配置环境变量

如果不显式传 `appId`、`appSecret`，适配器会从以下环境变量自动读取：

- `LARK_APP_ID` —— 飞书 App ID
- `LARK_APP_SECRET` —— 飞书 App Secret

## 适配器实现原理

这个适配器基于 [飞书 Channel SDK](https://github.com/larksuite/node-sdk)（`@larksuiteoapi/node-sdk` 中的 `LarkChannel`）构建。Channel SDK 是飞书提供的高阶 SDK，封装了：

- 事件订阅与签名验证
- WebSocket 长连接与心跳重连
- 20+ 种原生消息类型的归一化（`normalize()` 输出统一格式）
- Cardkit 卡片流式接口
- 历史消息拉取
- 应用凭证管理（自动 token 刷新）

适配器开发者只需要做"接口翻译" —— 把 Channel SDK 的接口映射到 Vercel Chat SDK 的 `Adapter` 接口。下面是核心代码示意（简化版）：

```TypeScript
import { LarkChannel } from "@larksuiteoapi/node-sdk";
import type { Adapter, Message } from "chat";

export const createLarkAdapter = (config?: LarkConfig): Adapter => {
  const channel = new LarkChannel({
    appId: config?.appId ?? process.env.LARK_APP_ID,
    appSecret: config?.appSecret ?? process.env.LARK_APP_SECRET,
    safety: { enabled: false }, // Chat SDK 已处理 dedupe / lock
  });

return {
    name: "lark",

async initialize() {
      // 收消息：Channel SDK 已经归一化，适配器只做格式映射
      channel.on("message", (normalized) => {
        const threadId = encodeThreadId(normalized.chatId, normalized.rootId);
        chat.handleIncomingMessage({
          text: normalized.text,
          threadId,
          author: {
            id: normalized.senderId,
            isMe: normalized.senderId === botIdentity.openId,
          },
          raw: normalized,
        });
      });
      await channel.connect();
    },

async post(threadId, message: Message) {
      const { chatId, rootId } = decodeThreadId(threadId);
      return channel.send({
        chatId,
        content: renderMarkdown(message),
        replyTo: rootId,
      });
    },

async stream(threadId, handler) {
      const { chatId, rootId } = decodeThreadId(threadId);
      // 使用原生 cardkit typewriter，非 post + edit 轮询
      return channel.cardkit.stream({
        chatId,
        replyTo: rootId,
        produce: handler,
      });
    },

async fetchMessages(threadId) {
      const { chatId, rootId } = decodeThreadId(threadId);
      const history = await channel.getHistoricalMessages({ chatId, rootId });
      return history.map(toMessage);
    },
  };
};

```

整个适配器的核心控制流不到百行 —— Channel SDK 已经处理了大部分飞书平台特定的复杂度，适配器只承担"接口翻译"这一核心工作。

## 相关链接

- [@larksuite/vercel-chat-adapter on npm](https://www.npmjs.com/package/@larksuite/vercel-chat-adapter)