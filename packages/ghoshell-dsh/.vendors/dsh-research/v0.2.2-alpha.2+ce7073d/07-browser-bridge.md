# 07 — 让 ghost 从外面控制 dsh 的浏览器界面（首轮实测）

## 动机

要把 dsh 的**浏览器界面**变成 ghost 能读能写的东西：不靠截图、不靠 playwright，而是把一段代码
投进页面、在页面的客户端 ctx 上执行、把结果拿回来。这一篇记第一次跑通时踩到的所有硬事实——
它们每条都会决定 `dsh_swarm` 的拓扑形状。

## 路径（可重放）

```sh
# 1. 纯脚本起一个有独立 home 的 dsh
.venv/bin/python .playground/dsh_bridge_probe/spawn.py     # DshLauncher, port 3084, home=/tmp/moss-dsh-probe

# 2. 插件放在 profile 里（见 05-plugin-install.md）
$DSH_HOME/profiles/web/moss-bridge/{package.json,host.js,client.js}
# cordis.patch.yml: - insert: [{id: moss-bridge, name: './moss-bridge/host.js'}]

# 3. 页面打开后，从外面投代码
curl -X POST http://127.0.0.1:3084/moss-bridge/exec -H 'content-type: application/json' \
     -d '{"code":"return ctx.get(\"sidebarRight\").isExpanded()"}'

# 读一份真 bundle 的 ABI：
head -c 300 node_modules/@deepseek-ai/dsh-client-ui-theme/lib/client.js
tail -c 300 node_modules/@deepseek-ai/dsh-client-ui-theme/lib/client.js
```

## 结论

### 装法与 ABI

- `[实测]` **一个目录 + 三行 patch 就够，不需要 pnpm / 打包 / 发布。**
- `[实测]` **patch 的 `name` 必须指到文件**（`'./moss-bridge/host.js'`）；写目录（`'./moss-bridge'`）
  会 import 失败 —— `dsh: warning: 1 entry did not activate` + `failed to import`。Node 的 ESM
  不认目录 specifier。client 半边照样被识别，因为 `nearestPackage` 从 `host.js` 往上找到
  `moss-bridge/package.json`。
- `[实测]` **client bundle 的 ABI**（照真 bundle 抄一次就对）：
  ```js
  window.__ModuleLoader__.load({
    id: 'moss-bridge',                       // 必须等于包里声明的 id / 包名
    factory: (require) => {
      var module = { exports: {} }, exports = module.exports
      // ... 代码 ...
      exports.apply = apply                   // cordis 插件面
      exports.inject = [...]
      return module.exports
    },
  })
  ```
  手写 JS 可以直接跑，不需要 tsdown。
- `[实测]` **boot payload 里能看到自己的 row**：`{"id":"moss-bridge","url":"plugins/??moss-bridge/client.js&rev=..."}`。

### 执行杠杆

- `[实测]` `new AsyncFunction('ctx', code)` 在页面里可用；`AsyncFunction` 从
  `Object.getPrototypeOf(async function(){}).constructor` 取。返回结果必须 **JSON 投影**，
  `ctx` 的方法常返回活对象。
- `[实测]` **`ctx.get(name)` 拿到的是真服务**，返回真实运行时状态。首次读到的值：
  `sidebarRight.isExpanded() === false`、`active() === null`、`layout.panelInfo === {}`。
- `[实测]` **`ctx.<service>` 会抛 `cannot get property "x" without inject`**，而
  **`ctx.get('x')` 绕过**。这是 cordis 反射层本来的语义（"without the inject requirement"）。
  **含义：客户端"只能拿声明过的服务"这条边界是可以被绕过的**，安全不能指望它。
- `[实测]` 控制面会自己拒绝无意义动作：`sidebarRight.toggleExpanded()` 抛
  `sidebarRight: no session surface is mounted`（该实例没有选中 session）。
  `layout.toggleSidebar()` 则执行成功，无报错。

### 门禁

- `[实测]` **URL query 会丢。** 带 `?token=...&moss-bridge=1` 打开，token 登录之后
  `location.href` 变成 `http://127.0.0.1:3084/` —— query 全被抹掉。
  所以"用 query 给页面打标"这条路不成立；要打标得走 `webserver/index-inject`（往 index 注入一行）
  或由 host 侧记住"这个 peer 是我的"。
- `[实测]` 自定义路由（`/moss-bridge/*`）不需要鉴权，loopback 上任何进程都能投代码。

### 调试手段

- `[实测]` 客户端插件里的**报错在 host 侧不可见**：`ctx.logger` 只进内存环形缓冲，不落
  stdout/stderr（见 `moss-dolores-ghost-plugin.ts` 顶部那段注释）。所以客户端失败只能靠
  **自建 beacon 路由**往回打点。

## 观点

**桥的最小形状 = 一个包两半 + HTTP 轮询。** host 半边三条路由（exec / pull / result），
client 半边轮询 + eval。不用 WebSocket、不用 typert 生成、不用构建链。

**但"能不能控"和"该不该让它控"是两个问题。** 这一轮证明的是前者。后者的边界在这个形状里
**完全不存在**：同 realm eval，`window`/`document`/`fetch` 全在，`ctx.get` 绕过 inject。
所以治理只能靠外面那一层（谁被允许发 code、发什么、事后如何复核），不能指望沙箱。

**对拓扑的推演（下一篇展开）**：既然"一个 dsh 页面可以被外部投代码驱动"，那么
"一个 dsh 进程 = 一个 agent"这个等式就不再成立 —— 页面、进程、agent、会话四者可以各自独立
寻址。这直接影响 `dsh_swarm` / `agent swarm` 的形状。

**失效条件**：
- 若上游给 client bundle 加了完整性校验（例如必须由 tsdown 产出、或签名），手写 bundle 作废。
- 若 `ctx.get` 在客户端被收紧成需要 inject，"反射绕过"这条消失。
- 若 token 登录不再改写 URL，"query 打标"这条重新可用。
- 若主 shell 将来挂上 CSP `unsafe-eval`，`new AsyncFunction` 直接死。
