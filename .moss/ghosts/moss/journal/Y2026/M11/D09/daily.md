---
summary: memento 跨 branch 的两条边；agent 契约定在异步原语；dsh 节点落地，Hive 拆出 agent 契约
status: writing
---

一晚上六笔提交。

**memento**：`recipients`（出）+ `confluences`（入）+ `children`（子分支）；两个 append-only 缓冲在 commit 时被原子认领、去重、吸收。归档改成递归、先子后父、逐条事件；容错补齐 —— 坏行隔离进 `*.rejected.jsonl` 并记进 `CommitRef.rejected`，timeline 坏行原位留占位，绝不让分支卡到要手工修文件。

**被连纠三次**：把"收侧先不要上"读成"整个不做"，只发了发信侧；给发信侧塞了恒为 0 的 `commit_seq`；拿 memento 事件当投递触发器。查 dsh 才发现 `session/prompt` 原生就有 `queue/steer` —— `post` 的语义根本不用发明。

**凌晨**重写 `agents/blueprint/memento_agent.py`：异步是原语（`prompt` 只回执），没有 close 只有 reboot（起 session 写一条 commit 当定位点）。第一版实现放到 `dsh-node` 里，不另开 workstream。

下午到傍晚是 dsh 节点。先补依赖：`ghoshell-dsh` 声明自己 import 的 httpx / websockets / pydantic / typing-extensions / pyyaml，根包加 `[dsh]` extra，`ghoshell-processes` 留在默认依赖里（`504f8e0e`）。然后修 launcher 的真逻辑：host / port / token 全部从 dsh 打印的 readiness 行解析回写，client rebase，就绪失败带 stderr 尾（`54d2394a`）——`--port 0` 这时才成立。节点落在 `deepseek_harness/v0_2_0/node.py` + `nodes/dsh-swarm/v0_2_0/`：launcher 起 dsh，webview 被 screen 收养成 `dsh 0.2.0-rc.2`，channel 只有 `status` / `web_url`（`be500fc8`）。23080 上活着。

最后一笔是治理：`Hive` 从 `memento_agent.py` 拆到 `agents/blueprint/hive.py`，`ghoshell-agent-swarm` 整包删除——agent swarm 上升到 mosshell 表面，不再是兄弟分发（`d0467f62`）。

今天三条教训：① Launcher 与 Connection 是抽象分家、实现可合可分，node 只选择 Connection 从哪来；我反复把实现形态当成抽象结论。② 刷新挂在 thinking 上，不挂在 interpreter / observe 上——挂上 channel 后一直推理，facade 就不刷；正确动作是交还一轮。③ `subprocesses: object | None` 这种绕路的类型比不会更糟；ABC 没建模完的地方要当面提，不许绕。

细节见 [dsh-node.md](dsh-node.md)。
