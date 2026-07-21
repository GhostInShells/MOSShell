# 接入飞书 Channel SDK

Channel SDK 让你的 Agent 接入飞书，在群聊、单聊、云文档评论中与用户实时对话。
它帮你处理事件接入、消息解析及去重、拦截策略、回复发送、流式输出、媒体上传和卡片交互等通用工作，开发者只需要关注 “用户说了什么、机器人该回复什么、是否还要继续处理”。

## 使用效果

假设你有一个旅游规划 Agent，接入 Channel SDK 后，可以在飞书中调用 Agent 实现以下效果。

-  **流式回复**：Agent 边推理边输出，用户在飞书里看到的卡片实时刷新进度和内容。
- **卡片交互**：在卡片按钮、表单、下拉，用户点击后回调事件直接送到你的 Agent。
- **云文档评论**：用户在飞书云文档里 @ 你的 bot 提问，Agent 收到评论事件后可以直接回复评论，无需切换到群聊。

## Channel SDK 做了什么

如果你需要开发 AI 助手、客服机器人、知识库问答、审批/工单助手、群内自动化协作、卡片按钮交互等场景，Channel 能显著减少接入飞书时的重复工程量。

| 业务目标           | 未接入 SDK 要做的      | 接入 SDK 后要做的  |
| -------------- | ----------------- | --------- |
| **建立与飞书的双向通道** | 申请公网回调地址，或自行实现 websocket 长连、断线重连、心跳检测                                                          | `createLarkChannel()` 一行代码全自动实现                                           |
| **解析飞书事件**     | `im.message.receive_v1` 原始 payload：chat_id/sender/message_type/post 嵌套结构，富文本/图片/文件/at 各种分支解析 | 直接获取 `NormalizedMessage` / `CardActionEvent` / `CommentEvent` 的统一结构化消息和事件 |
| **流式输出**       | 调 `card_update` API、处理限流、转换 markdown 为飞书卡片 schema                                            | `channel.stream()`里的 `ctrl.update()`自动刷新界面                                |
| **机器人回复策略**      | 区分单聊/群聊行为、@机器人策略、群白名单、消息去重、防刷屏策略                                                         | 设置`safety` 及 `policy` 配置参数，SDK 内置完成                                 |

## 接入 SDK

Channel SDK 的工作流程包含以下五个环节：

1. **传输连接**：建立 WebSocket 或 Webhook 通道，并获取机器人身份。
1. **事件转换**：把飞书原始事件转换成统一结构，例如 `NormalizedMessage`。
1. **回复策略**：消息去重、过期过滤、会话串行化、单聊 / 群聊策略拦截。
1. **业务分发**：由开发者实现，通过 `channel.on(eventName, handler)` 注册事件处理器，把消息路由给业务服务或 Agent。
1. **出站渲染**：把 Agent 输出转换为飞书消息、卡片、文件等，并处理分片、重试和降级。

### 多语言 SDK 支持

Channel SDK 支持以下多种编程语言：

| Github 仓库地址   | Github Readme   | 
 | --------------- | --------------- | 
|[NodeJS](https://github.com/larksuite/channel-sdk-node)  | [github readme](https://github.com/larksuite/channel-sdk-node/blob/main/README.zh.md) | 
|[Python](https://github.com/larksuite/channel-sdk-python) | [github readme](https://github.com/larksuite/channel-sdk-python/blob/main/README.zh.md)  | 
|[Go](https://github.com/larksuite/oapi-sdk-go/tree/v3_main/channel) | [github readme](https://github.com/larksuite/oapi-sdk-go/blob/v3_main/doc/channel.zh.md)  | 
|[Java](https://github.com/larksuite/oapi-sdk-java/tree/v2_main) | [github readme](https://github.com/larksuite/oapi-sdk-java/blob/v2_main/CHANNEL.md)  | 

## 能力边界

Channel SDK 不覆盖以下业务或 Agent 框架的能力范畴：
- **Agent runtime**：用什么模型、prompt 如何编写、工具如何调用
- **多用户隔离**：不同 `chatId` 之间的话题与上下文隔离
- **Session / 上下文持久化**：跨多轮对话的记忆存储
- **凭据存储**：拿到 `client_secret` 后的本地或远端存放方式