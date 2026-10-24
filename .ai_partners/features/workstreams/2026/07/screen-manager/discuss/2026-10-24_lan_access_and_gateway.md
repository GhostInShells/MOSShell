# screen node — 局域网接入、动态网关与认证（命题论证）

> 2026-10-24, 人类工程师 × deepseek-flash, via claude code
> 结论见上级 `FEATURE.md` 的「局域网接入」；本文记论证过程、事实勘察与被剪枝的分支。

## 上下文

命题：让手机 / 平板 / 电视等局域网设备也能看这块屏，并把屏上所有 webview 一起带出去；
路由由 ghost 动态管，准入默认拒绝，所有访问先登录（类似路由器，但要求更严）。

讨论从三个候选出发：① 新建 `lan_gateway` 节点动态代理；② 治理所有节点写死的 host，
让页面跟着浏览器 host 走；③ 网关由 screen 带 `--lan` 时启动。中间还提出过"网关用 LAN IP
作 host、app 节点跟着改"和"路径前缀 `/file_editor`"。

## 事实勘察（2026-10-24，基于当前代码）

**页面端几乎不存在写死 host。** 全量扫 `nodes/**/*.html|js`，唯一的 `127.0.0.1` 出现在
`ghost-in-web` / `ghost-in-bilibili` 浏览器扩展的 `sw.js`，那是扩展直连本地控制端口，
不是被嵌入的页面。webview 页面本身全部是 origin 相对的：

- `nodes/os/file_editor/index.html:2510` — `new WebSocket(location.host + "/ws")`
- `:1258` — `fetch("/renderers")`
- `:2534` — `fetch("/__ver")`

**写死地址只在服务端，共三处消费方式：**

| 位置 | 内容 | 影响 |
|---|---|---|
| `WebViewDeclaration(url=surface.url)` | 上报"我在哪监听" | 消费方据此建 iframe |
| channel 的 `out["url"]`（`file_editor/channel.py:1178`、`terminal/channel.py:670`） | ghost 念给人类的地址 | 局域网下指向 127.0.0.1 |
| `ScreenSurface.add_local()`（`surface.py:368`） | `f"{self.url}/local/{token}"` 绝对地址 | 发到浏览器后无法解析 |

**每个节点都绑 `127.0.0.1` + 临时端口**（`file_editor/main.py:37`、`terminal/main.py:37`
等）；screen 例外，固定 20381，但 `HOST` 仍是 loopback（`screen_manager/main.py:52`）。
`surface.url` 同时用于绑定、宣告和开浏览器，没有区分。

**screen 是 `webview` kind 的消费者**：`bridge.py:81` 把 `declaration.url` 原样塞进 store，
`index.html:965` / `:2764` 直接 `iframe.src = it.url`。

**协议约束**：`WebViewDeclaration` 是 provide 时冻结的静态 meta，运行态只能走 `state`
（pub + snapshot）。路由映射是运行态，因此不能写进 app 的 declaration。

## 存活结论

1. **网关是 screen 的可拆模块**，`--lan` 时组合启动。screen 已是所有 webview 的聚合者，
   路由表天然等于它已 adopt 的集合；但安全边界要保持独立模块，将来可原样拆成节点。
   参照 Home Assistant ingress、code-server：单进程里做边缘是常规做法，但都是独立模块。
2. **上游永远 loopback，网关是唯一对外 listener**；app 节点对 LAN 完全无感知。
   绑定地址与宣告地址分离；网关双 listener（本地 / LAN）。
3. **按 hostname（子域）或独立端口路由**，保留上游根路径与同源隔离。
4. **URL 治理只动服务端**：组件只知道监听在哪，永远不猜将从哪被访问；发给浏览器的引用
   一律 origin 相对；必须绝对时由网关按 `X-Forwarded-Host` 翻译。
5. **路由主键 = 服务 address**，slug 由 title 派生并持久化（端口会变，名字不能变）。
6. **策略三层，deny 优先**：内置默认（default deny、`terminal` 常驻 deny）→ 持久化文件
   （workspace 下，0600）→ 会话覆盖（ghost channel）。放行是安全事件：屏上可见 + 审计。
7. **ghost 自动降级、提权需人确认**（见威胁模型）。
8. **未准入 route 由网关返回静态占位页，不连上游**；screen 保留 item 并标未授权。
9. **全模式登录**，设备凭证优先，密码 Argon2id 哈希；本地注册/重置是 CLI/TUI，
   信任锚是文件系统权限，不依赖 screen 进程存活。
10. **TLS + Host/Origin 白名单为硬要求。**

## 被剪枝的方案

| 被剪方案 | 否决理由 | 去向 |
|---|---|---|
| **路径前缀**（`mosshell.lan/file_editor`） | 页面传输全部是根路径（`/ws`、`/renderers`、`/__ver`），前缀必然 404 / WS 连错；要救只能在网关注入 shim 改写 `fetch`/`WebSocket`，属于偷偷改别人的页面；更根本的是不同节点的页面会被塞进同一 origin，抹掉同源隔离（一个 XSS 打到 terminal） | 子域或独立端口 |
| **让 app 节点知道 LAN**（绑 `0.0.0.0` / 宣告 LAN IP / 网关用 env host 且 app 跟着改） | 网络拓扑被写进每个节点；DHCP 换 IP 全线失效；LAN 上出现 N 个监听面，认证无处收口；本地访问语义也随之改变 | 上游永远 loopback |
| **只让远程登录、本地免登录** | loopback 不是安全边界：同机用户、本地恶意进程、浏览器 rebinding / CSRF 都能打 127.0.0.1；两条路径必然漂移 | 全模式登录 |
| **加密 user/passwd 存 runtime** | 加密可逆，密钥与密文同机等于没存；认证只需要 verifier | Argon2id 哈希 |
| **把注册做成 loopback 网页** | 同样可被 rebinding 打到；还要额外造一次性 token | 本地 CLI/TUI |
| **路由器式默认凭证**（admin/admin） | IoT 行业的经典事故；首次启动必须没有账号 | 本地注册创建 |
| **单共享密码给所有设备** | 无法区分、无法逐台吊销、无审计 | 配对码换长期设备 token |
| **让 ghost 自由改白名单** | ghost 上下文会进 webview 内容、网页、终端输出、知乎文章——全是 prompt injection 载体；放行 terminal 等于给出直达远程 shell 的路 | ghost 降级自由、提权需人确认 |

## 行业参照

- **边缘唯一、上游 loopback**：docker compose 只 publish 边缘；Traefik / Caddy 做入口。
- **发现驱动路由，而非逐条下发**：Traefik provider/labels、Caddy admin API、Envoy xDS。
  路由是期望状态，由发现自动收敛；ghost 只改策略。
- **每服务独立 origin**：ngrok、cloudflared、localtunnel、Tailscale serve 都按子域或端口
  分配，不用路径前缀——原因正是绝对路径与同源隔离。
- **认证交给成熟组件**：Authelia / oauth2-proxy / Pomerium 做 forward-auth；单用户家用
  可以自建小网关，但密码学不要自创。
- **设备身份优于共享口令**：Tailscale / WireGuard 的模型；WebAuthn/passkey 适合
  域名 + TLS 的场景；一次性配对码适合电视/访客浏览器。

## 威胁模型

| 威胁 | 路径 | 对策 |
|---|---|---|
| 同一 Wi-Fi 下的其他设备 | WPA2-PSK 是共享密钥，流量可解密 | TLS；默认拒绝 |
| 访客 / 被入侵的 IoT | 直接访问 LAN 端口 | 认证 + 逐台设备 token + 吊销 |
| DNS rebinding | 恶意网页把域名解析到 LAN IP，借浏览器打本地服务 | Host 白名单；WS 握手校验 Origin；登录页/占位页 `X-Frame-Options: DENY` |
| 本地其他进程 / 用户 | loopback 无认证 | 全模式登录；注册 CLI 的信任锚是文件权限 |
| prompt injection | 恶意内容诱导 ghost 放行 terminal | 提权需本地人类确认；降级可自动 |
| iframe 内容互攻 | 多节点共享 origin | 子域隔离 |
| 静态资源越权 | `/local/*` 能读工作区文件 | 认证覆盖静态路径 |

## 认证落地要点

- 密码：Argon2id（次选 `hashlib.scrypt`）+ 盐 + 参数；绝不存可逆密文。
- 会话：256-bit 随机 token；HttpOnly、SameSite=Lax、TLS 下 Secure；登录轮换；
  空闲 + 绝对过期；常量时间比较；限速 + 指数退避。
- 设备：本地注册后发长期 token；远程设备用一次性配对码换 token；设备列表可吊销。
- 恢复：本地 CLI 可重置密码、吊销全部设备；不能依赖 screen 进程存活。
- 跨子域 SSO：cookie `Domain=.mosshell.lan`。
- WS 认证：cookie，不放 query（日志泄露）。

## 未决

- 子域 vs 端口的默认选择；泛解析 DNS / mDNS 的可用性。
- TLS 供给：Caddy internal CA / mkcert / Tailscale；WebAuthn 依赖域名 + TLS。
- 多屏 slug 冲突；远程 Matrix cell 的 `url=127.0.0.1` 不可达（现有协议边界，本命题不解）。
- LAN 观众是只看还是可操作——屏幕可点击并回流信号，鉴权之外可能需要只读档位。

## 下一步

登记为 `screen-todo.md` 的 W17（网关）、W18（URL 治理）、W19（认证），选型见 O5。
先在现有协议上做 W18 + 子域/端口二选一验证，再合入 W17；认证最后上，但 TLS 一并在
W17 内解决。
