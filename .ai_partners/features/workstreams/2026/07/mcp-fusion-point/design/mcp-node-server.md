# 子任务 1 — Node Run As MCP Server（朝外）

> 索引于 FEATURE.md。机制已敲定，实现待开工。

## 定位

node 的核心身份就是 MCP server——进程跑起来是一个说 MCP 的端点，别的功能可有可无
（可以有 channels，可以没有）。**不是把 channels 投影成 MCP tools 导出（无意义）。**

它的意义：给 mesh 外 agent 的流量开一个**进入 mesh 的入口**。外部 agent 作为 MCP client
连上来，调用从端点**回环进 matrix**，变成 mesh 内可感知 / 可审批 / 可路由的事件。
node 是"外部世界进 mesh 的边门"，不是"能力出口"。

## 关键机制

### 启动形态

- **`moss nodes run` 启动，不是 `mcp run`**——node 有 cell 身份 / 生命周期 / 治理，
  `mcp run` 会剥掉 node 身份。MOSS 拥有启动，FastMCP 只拥有 transport serving。
- **糖：`matrix.run(mcp)`**——调用 matrix 的 run 方法（matrix.py:428，同步阻塞入口，
  自动拉事件循环并治理生命周期），serve 一个**已配置好 tools 的 MCP 实例**。tools 在
  matrix 初始化后、run 之前注册（node 作者用闭包捕获 runtime，调用时才用）。**run
  内部不重新注册 tool**——这正是糖区别于样板脚本的地方；否则它退化成
  `if __name__ == "__main__"` 示例，不是可复用的糖。

  ```python
  def main(port: int = 0):                                 # main() 约定，见下
      matrix = Matrix(...)
      mcp = FastMCP("node", transport="streamable-http")
      @mcp.tool()
      async def some_tool(...): ...                        # 注册运行时 tool（闭包捕获 matrix）
      matrix.run(mcp)                                       # matrix 生命周期内 serve，不重新注册
  ```

- **坑（糖要封装的第一个点）**：serve 内部必须用 `run_async`（async），绝不能
  `mcp.run()`——`run()` 内部 `anyio.run()` 会开新事件循环，跟 matrix 的 loop 打架。

### Transport：stateless streamable HTTP

- 协议值：`transport="streamable-http"`。
- stateless 值：`stateless_http=True`（fastmcp>=3.4.3）或 env `FASTMCP_STATELESS_HTTP`。
- 调用：`await mcp.run_async(transport="streamable-http", stateless_http=True)`
  或 `http_app(stateless_http=True)` 挂载。

### node main() 约定

- `main(port: int = 0)`——0 = OS 自动分配，默认调用即跑通。
- 显式 port 被占 → uvicorn startup 抛 OSError → 返回提示："port X occupied，改用 port=0
  自动或另选端口"。
- 模型启动逻辑：默认调用能跑，撞端口返回值提醒显式定义。

### 单进程 ASGI

- 专用 MCP node：uvicorn 同进程同 loop 服务 FastMCP app，matrix 后台任务共存。
- **绝不做 `startup()` 后 `serve()`**——`serve()` 内部重新 `startup()`（uvicorn
  server.py:93），port 0 会重新 bind 一个新 port，announce 的 port 和实际 serve 的
  port 不一致。这是糖要封装的第二个点。
- 正确二选一：
  - 预 bind socket → `serve(sockets=[sock])`（全公开 API，port 稳定，有早连窗口）。
  - `startup()` → 读 bound port → `main_loop()` + `shutdown()`（零窗口，半内部 API）。
- `http_app()` 挂载只用于**复合 ASGI**（一个 web server 挂多个 ASGI 面，如 MCP +
  Reflex GUI 同端口）。纯 MCP node 让 FastMCP 自己拥有 uvicorn。

### 端口 + announce 时序

pub 需要 port，port 在 bind 时确定。推荐两个 task，异常边界是关键：

- **serve task（主体）**：预 bind socket → `startup(sockets=[sock])` → 失败冒泡杀 node
  （否则 ghost 收到死地址）→ `ready.set()` → `main_loop()` + `shutdown()`。
- **announce task（尽力而为）**：等 ready → publish endpoint。失败只记录不 kill serve
  ——ghost 永远可以 pull。

**announce 机制**：node 生命周期 **EVENT** publish 进 **nodes channel**（统一治理），
endpoint 同时写 **cell presence**（pull 面）。**不发 signal**——node 自己发 signal
只能是 buffer signal，滥用难治理。ghost 从 nodes channel 读节点状态 + 收治理过的事件。

pub 内容：`{transport, host, port, node}`。

### 回调进 mesh

外部 agent 调用 MCP tool → 路由进 matrix 网络内协议（如给 ghost 发 signal、对某
channel 的一次调用）。ghost 可感知 / 审批 / 路由。

## 用例

- 给 ghost 发 signal 的 MCP。
- 给第三方 agent 用、调用后回调进网络协议的 MCP。
- claude code 回环：ghost 审批 claude code 调用的 MCP 工具。

## 依赖

- 新增 `[mcp]` extra：`fastmcp>=3.4.3` + `mcp`（stateless 版）+ `uvicorn`。
  修掉 `mcp_hub.py` 里悬空的 `[mcp]` 引用。
- 现状：`fastmcp>=3.1.1` 在 host，`mcp>=1.21.0` 在 dev。`[mcp]` extra 目前不存在。
