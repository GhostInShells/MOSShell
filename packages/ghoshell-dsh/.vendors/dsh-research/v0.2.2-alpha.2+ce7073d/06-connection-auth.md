# 06 — 用 DshConnection 控制一个 dsh：线协议与鉴权

## 动机

自己起的 dsh 要能冷读、能下命令。每次都要重新摸 HTTP 形状和鉴权，这里钉死。

## 路径（可重放）

```sh
cd .vendors/dsh-research/v0.2.2-alpha.2+ce7073d/source
sed -n '1,40p' packages/api/gateway/README.md
sed -n '1,40p' packages/client/connection/README.md

# 我方 facade（信封与鉴权都写在 docstring 里）：
sed -n '1,40p' src/ghoshell_moss/deepseek_harness/client.py
grep -n "_authorize" -A 20 src/ghoshell_moss/deepseek_harness/launcher.py
```

## 结论

- `[读码]` **unary Remote 的线形状**（`src/ghoshell_moss/deepseek_harness/client.py:5-10`）：
  - URL：`POST /api/{namespace}/{method}`（**斜杠**，不是点号）
  - 信封：`{type:'client-request', rpcId, method:'{ns}/{method}', payload:{args:{...}}}`
  - 响应：`{type:'server-response', rpcId, result:{ok:true,value} | {ok:false,error}}`
  - `args` 的键 = Remote 方法的参数名。多数叫 `request`，**`session/list` 叫 `_request`**，
    无参方法（`session/modelCatalog`）为 `{}`。
- `[实测]` `/api/session.list`（点号）→ **404**；`/api/session/list`（斜杠）→ 200，但若信封的
  `type` 不是 `client-request`，返回 `gateway/bad-request` 并明说 `expected "client-request"`。
- `[实测]` **token 只做 bootstrap，消费一次就作废。** 同一个 token 在第二个进程里再换 cookie 会
  **401 unauthorized**；而第一个进程（持有 launcher 的那个）当次能成功。
- `[读码]` 与 `dsh-version-tracking` KD5 一致：*cookie 优先（可持久化），token 仅作 bootstrap*，
  cookie 的签名密钥落盘在 `$DSH_HOME/.credentials.yaml`。
- `[实测]` `launcher.call(path, payload)` 就是裸 `POST {base}{path}`，**不做信封、不换 cookie**；
  它适合调用插件自己 `register()` 的 HTTP 路由（如 dolores 的 `/moss-api/...`），不是 Remote 动词。

## 观点

**控制必须由持有 launcher 的那个进程做。** 理由不是权限习惯，是 token 的语义：它是一次性的
bootstrap 凭据。跨进程"再连一次"在协议上就不成立，除非把 cookie 持久化下来复用。

对 `dsh_swarm` node 的直接含义：**launcher 和 channel 必须在同一进程里**，channel 的每个动词
复用同一个 `DshConnection`。这也正好是现有 node 的形状。

**失效条件**：
- 若上游允许 token 重复交换（或提供长期 token），"必须同进程"这条可以放宽到"cookie 持久化"。
- 若信封从 `payload.args` 改成别的路径，`client.py` 的映射要整体重写；`args_key` 那张表
  （`request` / `_request` / `{}`）是第一顺位要重核的东西。
