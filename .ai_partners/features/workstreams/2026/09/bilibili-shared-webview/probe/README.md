# 探针 #2:MV3 service worker 能不能稳定持有 WebSocket

一次性的。结论回填到 [../design/2026-09-20_ghost_in_bilibili_body_design.md](../design/2026-09-20_ghost_in_bilibili_body_design.md)
的「未验证声明」,这份探针本身可以扔掉。

## 为什么值得单独跑

整个通讯设计压在一条浏览器行为上:**SW 会不会在 ~30s 空闲后被回收**。它错了,通讯设计
要重来。这里不能靠猜 —— 上一次"隔离世界不受页面 CSP 约束"的直觉就是这么翻车的。

顺带测三条边界事实(内容脚本里 eval / 跨域 fetch / EventSource),各有单独判据。

## 跑

```bash
# 终端 1
.venv/bin/python .ai_partners/features/workstreams/2026/09/bilibili-shared-webview/probe/server.py

# Chrome: chrome://extensions → 开发者模式 → 加载已解压的扩展程序
#         选 .../bilibili-shared-webview/probe/extension
# 然后打开任意 B 站视频页(https://www.bilibili.com/video/BV...)
```

服务端只写 `ws://127.0.0.1:23881`,所有事件打在那个终端里,同时追加到同目录 `probe.log`。

## 请做的动作(共约 12 分钟)

1. 打开一个 B 站视频页,让它待着 —— **什么也别做,至少 10 分钟**。这是核心观测。
2. 中途按一次 **F5 刷新**该 tab。
3. 再**开第二个视频页**(另一个 tab)。
4. 关掉其中一个 tab。
5. 回到终端,Ctrl-C 看汇总。

## 看什么 → 判什么

服务端输出里,`★ 存活 60s / 180s / …` 是心跳里程碑,`-- 断开` 是连接断。

| 观察 | 结论 |
|---|---|
| 一条连接连续存活 ≥ 10 分钟,中间没有 `-- 断开` | **设计成立**:app-level ping(20s)续命有效,建通讯层 |
| 每 ~30s 就 `-- 断开` 一次然后重连 | 消息没能续命。跑第二轮:`--no-app-ping` 对比,再不行就上 `chrome.alarms` |
| 断开后立刻自动重连、且 `<- log` 里出现 `sw boot` | SW 被回收过,但重连逻辑成立 —— 重连必须当常态设计(已按此设计) |
| 刷新 tab 后连接没断 | 内容脚本重跑不影响 SW 的连接 |
| 第二个 tab 只多出 `msg tab=...`、没有第二条 WS | 一条连接复用所有 tab 成立(设计的关键假设) |
| `握手 origin='chrome-extension://...'` | Origin 白名单这条路成立,把值记下来 pin 进配置 |
| `没有 Origin 头` | 白名单不成立,安全方案要换(改用 token) |

内容脚本那三条边界事实会以 `[sw] tab N checks {...}` 的形式出现在日志里:

| 字段 | 期望 | 若不是 |
|---|---|---|
| `eval` | `EvalError: ...`(被 CSP 禁) | 复核设计文档里那条结论 |
| `fetch` | 失败(CORS / blocked) | 内容脚本能直连 node —— 通讯设计可简化,SW 不必然是唯一边界 |
| `eventsource` | `constructed onerror` 或构造即抛 | SSE 方案可用,重新评估通讯选型 |

## 收尾

把上面判据表逐行填上结论,回写设计文档的「未验证声明」,然后 commit 探针 + 结论。
探针代码本身不进 `nodes/` —— 它不是实现。
