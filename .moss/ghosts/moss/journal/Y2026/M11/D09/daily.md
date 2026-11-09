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

凌晨接着做的是**表面**。他给了六条原则（NODE.md 面向使用者 / instruction 只写运行时策略 /
gated virtual children / 信号用 XML 容器 / 主 channel 自解释 / 冷读要全），我逐条照做，却把
一个正在跑的调研亲手重启掉了——六次 sleep、五次重启、一条结果都没留下。教训不是"别重启"，是
**真机不该当开发循环**：`.playground` 才是验修复的地方。

然后是长时间的对齐，四轮没动手：node name 来自声明、同时就是 hive name；回调只带 agent 与
时间戳；`on_output` 改名 `on_turn_end`，只交 turn 末的 final answer 全文；task 就是运行时，停它
要带 reason 真停 session，并且**先 discord 再关**。收尾加了一个页面桥：`ctx.get('layout')` 是活
的 dsh 服务——控制页面不该改 CSS。

细节见 [dsh-node.md](dsh-node.md)。
