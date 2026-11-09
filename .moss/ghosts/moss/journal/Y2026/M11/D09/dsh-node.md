# dsh 节点 —— 从 launcher 到一个能被 matrix 启动的 cell

## 1. 依赖（`504f8e0e`）

- `ghoshell-dsh` 声明自己 import 的五项：`httpx` / `websockets` / `pydantic` / `typing-extensions` / `pyyaml`。之前靠 workspace 环境兜着，wheel 安装会缺。
- 根包加 `[dsh] = ["ghoshell-dsh"]`，`[tool.uv.sources]` 用 workspace 解析。`ghoshell-processes` 留在默认依赖，位置等同 `ghoshell-common`。
- `uv.lock` 在 `.gitignore` 里，不入库；本机 lock 已刷新。

## 2. launcher 的 URL 解析（`54d2394a`）

dsh 打印的 readiness 行是它自己的 origin；launcher 之前只抓 `token=`，host/port 仍取 config，所以 `--port 0` 永远等不到 mux。现在：

- `_adopt_web_url`: `urlsplit` 解析 host/port/token；loopback（或与 config 同 host）才把 origin 写回 config，并 `client.rebase`；publicUrl 那种非 loopback 的广告 URL 只留给 `web_url()`。
- `DshClient` 加 `base_url` + `rebase`。
- 就绪超时带上 dsh stderr 尾部；端口被占不再只是 30 秒裸超时。
- 包内单测 64 passed，新增 3 个用例（ephemeral 写回 / public 不收养 / 固定端口不动）。

## 3. 节点（`be500fc8`）

目录 `nodes/dsh-swarm/v0_2_0/`：

- `NODE.md`：面向使用者。不写真实 CTML 路径，也不写 `moss nodes run`；启动走 matrix 的 `nodes` channel。
- `INSTALL.md`：维护面。环境要有 `[dsh]`（`uv sync --all-extras`，不要 `--extra dsh`，会剪 host 层）；dsh 0.2.0-rc.2；`python -m ghoshell_moss.deepseek_harness.v0_2_0.node --install` 初始化 home；`moss nodes install` 落 `.installed`。
- `check.py`：spawn 前 probe。查 extra、dsh binary、`.dsh` / `.memento` / 插件 patch。失败即拒绝拉起。
- `main.py`：一行调 `run_node`，不自己维护台账。

库侧 `deepseek_harness/v0_2_0/node.py`：`prepare_home`（一次性、幂等；`.dsh` 已存在只换插件）、`DshNodeRuntime`（launcher + `matrix.processes`）、极小的 `dsh` channel（notice + `status` + `web_url`）、`main` / `run_node`。

验收：节点通过 `matrix.nodes:run` 起在 23080；webview 被 screen 收养成 `dsh 0.2.0-rc.2`；`status` 返回 home / profile / port / running / web（含 token）。

## 4. Hive 拆分（`d0467f62`）

- `Hive` 从 `agents/blueprint/memento_agent.py` 移到 `agents/blueprint/hive.py`；正确 import `AgentPreset` / `Mandate` / `MementoAgent`。
- `ghoshell-agent-swarm` 整包删除；workspace members 去掉它。agent swarm 是 mosshell 表面，不是兄弟分发。

## 5. 还没解决的

- `ghoshell-processes` 的 capture bug：`_drain_stream` 里 `f.write()` 从不 flush，cell 的 stdout 在进程退出或写满缓冲前不可见。一行修复 + 一个"进程未退出时也能读到"的测试。
- Hive ABC 还缺 `as_channel()`；`workspaces() -> list[str]` 没有命名模型。方向：Hive 自己声明 channel（不许旁路 `new`）；workspace 要 name / title / path，命名权在 Hive。
- `DshHive` 替换 `DshNodeRuntime`：只做发现与感知（workspaces / presets / notice / named_notices），agent 能力显式留空。

## 6. 今天踩到的判断错

- 把 StatefulChannel 的 `cleared` 日志读成启动失败，拿 lsof / 日志盲查。实际刷新挂在 thinking 上，不挂在 interpreter / observe 上；挂上 channel 后应等下一个 moment，而不是用 shell 猜。
- 把 Launcher / Connection 的实现形态（`DshLauncher(DshConnection)`）当成抽象结论（"两个 node"）。两者是抽象分家，实现可合可分；node 只选择 Connection 从哪来。
- `subprocesses: object | None` —— 绕路类型。遇到 ABC 没建模完的地方要当面提，不许绕。

## 7. 凌晨续：表面、身份、运行时（D10 02:30）

**六条表面原则**（他）：NODE.md 面向使用者；instruction 只写运行时策略（channel 本身就是渐进披
露）；不用 sustain 子通道、用 gated virtual children；信号要用 XML 容器带 hive/name/时间戳；主
channel 自解释；冷读要全、温数据最小。

**对齐出来的四条**（都经过他纠正）：
- **一个名字**：node name 来自 NODE.md，经 `build_node_cell` 成 `matrix.this.name`，再就是 hive
  name、channel 名、notice 首段、信号 tag。不用 alias；名称带引擎版本（`dsh_0_2_0_rc_2`），于是
  版本只出现一次。
- **`on_output` → `on_turn_end`**：dsh 每个 step 一条 assistant 消息，preamble 与工具步骤都不是
  答案。turn 内缓存最后一条，`TurnEnd` 才交付，全文，不截断。
- **task 就是运行时**：一个 agent 一个 task；停它 = `cancel(reason)`（以 `<system_reminder>` 注
  入）+ discord 订阅 + 关 + 出名册。**先断后关**——先关的话，队列里残余的帧会送到没有主人的 hook
  上，那就是"死人的话还在说"。
- **两笔引擎修复**：`workspace/follow` 的增量帧以前被丢掉（镜像冻结在开机），现在 baseline 重建、
  upsert/remove 维护，`wait_workspace` 是 per-call 的等待；`create_workspace` 用 mutation 自己的
  回执，不再回头查一份可能过期的列表。

**页面桥**：同一个插件包两半（host 路由 + `client.js`），在页面的 cordis ctx 里跑代码，`describe`
读活的 API 形状——`layout` 的方法当场读出来是 `selectPanel / beginNavigation / dispose /
toggleSidebar / openRightbar / closeRightbar`。`layout.toggleSidebar()` 在屏幕上真的动了。

**踩到的两个坑**：① 在 `apply()` 里 `ctx.get('webServer')` 探测会静默落空（服务还没注册），改
成子插件 + `inject: ['webServer']`；② 后台窗口的定时器被浏览器掐停，页面只在**前台**轮询——
`attached` 是唯一可靠的读数。

**还没做**：memento agent 实装（主线）、session 归档与数量策略、`configure` 未验、turn 无答案时
没有出口、页面桥应为可选。
