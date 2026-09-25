---
name: plugin-api-session-event
description: >-
  验证「plugin 注册 api + tool」与「ghost runtime 反向通讯」的完整链路。
  dsh web 内置 /api/events.mux WS 下行流 + plugin 注册的 HTTP 回调, 构成零依赖伪双工。
---

# Plugin API + Session Event

本 skill 验证 dsh 融合的核心命题:**ghost runtime 与 dsh 之间, 用「dsh web 内置 WS
下行 + plugin 注册 HTTP 回调」建立零依赖的双向通讯**, 不引入 std 协议、不 import 第三方包。

## 验证的链路(已跑通)

```
agent 调用 tool 'moss_shell_observe'
  → dsh append tool/call 事件
  → session/event 广播
  → /api/events.mux WebSocket 下行 (dsh web 内置 client-connection 提供)
  → python (websockets 客户端) 收到 tool/call 帧 (含 callId + arguments)
  → python 调 /plugin-api/callback (HTTP, plugin 注册)
  → plugin touch callback.marker ✅
```

判定证据:`callback.marker` 生成, 内容含完整 callId 与 arguments。

## 关键结论(可确信, 源码 + 实验双重验证)

1. **dsh web 自带 session/event 下行流** — `/api/events.mux`(WS)是 `client-connection`
   插件提供的, 广播 `session/event` 帧。**不需要 `sdk-jsonrpc-server`、不需要 std 协议、
   不需要 import ws 库。** python 用标准 `websockets` 库连上即收。

2. **plugin 零第三方依赖** — 本 skill 的 plugin 只用:
   - `node:fs`(touch 文件)
   - `ctx.tools`(注册 tool, `defineTool`)
   - `ctx.webServer`(注册 HTTP 回调 api)
   全部走 dsh 注入服务, 无任何 import 的第三方包。

3. **伪双工拓扑** — 正向(dsh→python)走内置 WS, 反向(python→dsh)走 plugin 注册的
   HTTP。两条通道都是「已存在的东西」, ghost runtime 不需要开自己的对外接口。

4. **tool 注册** — `ctx.tools.register(defineTool({...}))`, 参数/输出 schema 用**纯对象
   描述**(`{ type: 'string', required: true }`), 不是 schemastery 的 `z.string()`。
   tool 的 execute 返回 Promise, `tool/call` 事件会自动 append 到 session 并广播。

5. **每个 skill 自包含环境** — `home/profiles/web/{package.json, cordis.patch.yml,
   cordis.yml, pnpm-workspace.yaml}` + plugin.ts。`node_modules` 由 dsh 启动时
   `healProfilesModuleFallback` 自动软链到全局 dsh 安装, 不需手动建、不需提交。

## 运行方式

```sh
# 常驻脚本: 启动 dsh web + 连 WS 监听 tool/call, ctrl+c 关闭
cd skills/plugin-api-session-event
python3 serve.py
# 打开 http://127.0.0.1:3083, 让 agent 调用 moss_shell_observe
# 观察 serve 输出 + callback.marker 生成
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `serve.py` | 常驻: 起 dsh web, 连 WS 监听 tool/call, 触发回调 |
| `probe.py` | 一次性: 验证 emit → session/event → 反向回调(早期版本) |
| `home/profiles/web/plugin.ts` | plugin: 注册 tool + 回调 api |
| `home/profiles/web/cordis.patch.yml` | 挂 plugin(相对路径 `./plugin.ts`) |
| `home/profiles/web/package.json` | web profile bundles 声明 + `type: module` |

## 踩过的坑

- `defineTool` 的 `parameters`/`output.schema` 必须用纯对象 JSON-schema 描述,
  不能用 schemastery 的 `z.string()`(`unsupported JSON schema: must be a value schema object`)。
- plugin.ts 是 ESM, 需在 `package.json` 加 `type: module`, 否则 Node 24 重解析警告。
- `time.sleep` 在 asyncio 里会阻塞 event loop, 导致 WS collector 无法 connect, 用 `asyncio.sleep`。
