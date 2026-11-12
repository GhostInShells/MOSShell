---
summary: dsh node 表面按六条原则重写，memento agent 与 case 抽象成形，夜里 case 端到端跑通
status: closed
---

凌晨到两点半是 dsh node 的表面。他纠正我把 connection 当 node——探针连的是 :3083 那台"别人家的 dsh"，真正起点是 launcher（持进程、持 home、装我们的 patch）。随后按他给的六条原则重写 surface：NODE.md 面向使用者、instruction 只写运行时策略、agents 走 gated virtual children、信号用 XML 容器、主 channel 自解释、冷读全而温数据最小。workspace 自解释被证伪：`create_workspace` 回的是引擎接受而不是状态落地，refresh 只吃 baseline、把增量帧丢了；补了冷读与 fail-closed。回调面三处错被逐条点破，最后定死：node name 来自 NODE.md 且就是 hive name；回调只带 agent 与时间戳；`on_output` 改名 `on_turn_end`，只交 turn 末 final answer 全文；task 就是运行时，停它要真停 session，先 discord 再关。

下午到夜里是 memento agent 与 case。蓝图第一版被我写成"平行 ABC"，他一句"为何重写 ABC，而不是继承"点破，改为继承 `AgentSession/Agents/Hive`。case 抽象经四轮改写：`draft → bind → autopilot`、Case（死）/CaseAgent（活）、`CaseState` 四值、系统 commit 用 metatype 不带 operation；布局收敛为一个目录一个 hive（`.hive/.dsh` + `.hive/cases`）。case 端到端跑通后，一个自主 agent 反查出两大缺口，chat 异步化与旁路问答重做；插件把 `chatOffRecord` 做成冷会话 + 任意 seq。

夜里重开机验收：我先用 read/bash 读盘、绕过 channel 看答案，被他点破"为什么是用 read 工具看 node"——错的是立场，不是结论；全程改走 CTML 后，`open · idle · 2-7` 与七条 commit 都对上了。autopilot 默认 reboot 落地：没载体就从 last commit 的 `session_ref` 恢复同一 session，`bind` 才是特殊操作。

教训两条：① 探针只验环境，不过度承诺任务；② 真机不该当开发循环，`.playground` 才是验修复的地方。
