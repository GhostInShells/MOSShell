---
name: ghoshell-dsh-integration
description: dolores ghost prototype 重新集成 dsh v0.2.0 的全部操作点 — 声明/安装、进程/连接、会话/事件、路由、冷读与历史、模型/权限、测试、版本基线；排序待定
---

# ghoshell-dsh Integration — 操作点清单

> 来源：D07 13:13 → 16:00 的一轮对齐（trunk 定义 + 逐条对表）。**这是操作点清单，不是方案、不是排序**；顺序另排。
> 归因沿用 FEATURE.md 的约定：`他` = thirdgerb 说的；`我推` = moss 的推断，未经裁决；`待定` = 已知未决。
> 相邻但不同的一条线：attachment 生命周期在 `packages/ghoshell-dsh` 的 `dsh-attachment-retention` workstream。

## 0. 目标（他）

- 主干 = dolores ghost prototype **重新集成 dsh v0.2.0 的能力**。
- 最终目标 = **commit ref 的历史可读还原**。
- 手法 = 把 dolores 对 dsh 的显著依赖收进一个叫 **Engine / Kernel** 的抽象（不叫 Harness）：dolores 实例化一个 Engine，Ego 等只面向 Engine；**历史兼容层提供一个只读 Engine 面**。
- `commit ref` 尽快增加一个 **engine 字段**，形如 `dsh/v0_2_0`（代际，不带 rc 等不稳定后缀），指向一个**可注册的 Engines 包**。无功能收益，纯兼容，可后做。

## 1. 声明与安装

| # | 操作点 | 证据 | 状态 |
|---|---|---|---|
| 1.1 | **声明机制缺失**：插件名/模块路径全是字符串（`BUNDLE_ENTRY_ID="ghoshell"`、patch 里 `@local/ghoshell-dsh`、消费方 import `ghoshell_dsh.v0_2_0.launcher`），`pyproject.toml` 无 entry points → 集成必须"知道"字符串，不可自动化 | 他；`install.py:40`；`pyproject.toml` | 待定（entry points 还是包内注册表） |
| 1.2 | **一行启动**：除 dsh 安装外，不应有别的显著安装动作 | 他 | 目标 |
| 1.3 | launcher 自带 install classmethod（已存在）：`DshConnection.install(home, profile, binary, init=False)` + `at(home)`；`install.py` 是其实现体 | 他提法 → 已核 `launcher.py:604,623` | 已就位 |
| 1.4 | 缺"覆盖"：写出的 patch 只有 `insert`（加 `@local/ghoshell-dsh`），不会发 `{id, disabled:true}` 替换既有 row | 我见 `v0_2_0/plugin/cordis.patch.yml` | 待定（"覆盖"= replace row？） |
| 1.5 | 开箱配置归属：stub 与活 home 已分叉（stub 3 行 vs home 5 行，多 `ui-settings-general` / `agent-default-model`）；方向是 **home → stub 反哺** | 他；两边 patch 对比 | 已定向 |
| 1.6 | `ui-deliverables` 不启用（要求散文提文件，与 CTML-first 冲突；他判为 dsh 隔离缺陷，matrix node 更好）；`file-reference-local` 先不管 | 他（D07） | 已定 |

## 2. 进程与连接

| # | 操作点 | 证据 | 状态 |
|---|---|---|---|
| 2.1 | blueprint 是 has-a；实现用 is-a + `return self` 可接受。真实分界在实现里：进程走 std、连接走 web | 他；类图 | 已定 |
| 2.2 | `DshLauncher.__init__` 是否接收 `subprocesses`、能否直接吃 `ghoshell_processes`（`matrix.processes` 的接法） | 未核 | 待核 |
| 2.3 | web launcher 面：token 正则、`web_url()`、cookie 交换要显式成一套（现在 token 在旧 launcher，cookie 在新 connection 内部） | 他提法 + 我见 | 待做 |
| 2.4 | 断连策略：launcher 起的 dsh 断连后是致命故障还是重试 N 次 | 他判为新功能点 | 待定 |

## 3. 会话与事件

| # | 操作点 | 证据 | 状态 |
|---|---|---|---|
| 3.1 | 回调面同构：新 `DshSession` 有 `on_exit / on_session_event / on_session_event_model / accept_session_event / accept_follow_snapshot` | 已核 `session.py` | 同构，待字段核 |
| 3.2 | 字段级核对：chunk 的 frame 结构、`TurnEnd.reason` 枚举、`AssistantMessageEvent.usage` | 未做 | 待核 |
| 3.3 | 驱动模型：`_run.py` 是流式消费；`session.run()` 是阻塞单轮。**收线判据归 run（流式监听 + 独立 rpc）**，不归 turn/end 单点 | 他；`_run.py` 头注 | 已定向 |
| 3.4 | `when_idle` 语义变强（内核权威轮询 `waitState`，多 timeout，返回 `SessionState`）；依赖旧行为的调用点重审 | 他 + 已核 `session.py:785` | 待审 |

## 4. ego 私有路由

| # | 操作点 | 证据 | 状态 |
|---|---|---|---|
| 4.1 | 5 条纯透传（ego/create、thinking enter/exit、tool-result、moment/push、turn-end-probe）：只把 `launcher.call` 换 `client.plugin_call` | 他：只有三条要考虑替换 | 已定 |
| 4.2 | `compact`：新包**无对等物**，要单独设计 | 他 | 待设计 |
| 4.3 | `bypass/run` → `chat_off_record`；返回值形状单独核 | 他 | 待做（前提见 5.1） |
| 4.4 | `read` → `session/page`（原始 `SessionEvent`，可翻页）；返回值形状单独核 | 他 + 已核 `session.py:411` | 待做（前提见 5.1） |

## 5. 冷读与历史

| # | 操作点 | 证据 | 状态 |
|---|---|---|---|
| 5.1 | **冷读缺位**：`chat_off_record` / `anchor` / `sessionState` / `waitState` / `session/page` 都依赖 live agent 或 follow cursor；源 session 已死时都调不了。要补冷读，或做成 connection 级接口 | 我见（`ctx.agents.get` + `self._cursor`） | 暴露问题（他） |
| 5.2 | commit ref 加 engine 字段 + 只读 Engine 面（见 §0） | 他 | 待做，后置 |
| 5.3 | 历史数据现状：595 个 session 日志中 209 个 legacy-seed 不可读、0 引用；190 个被 614 条 commit 引用的源日志全部可读 | 我见 D07 盘点 | 已量 |

## 6. 模型与权限

| # | 操作点 | 证据 | 状态 |
|---|---|---|---|
| 6.1 | **API 漂移实证**：`selectForNextRequest` / `selectionFor` 在 rc.2 是官方（`dsh-api-session-controller/lib/types/agent.d.ts:107`，`agent.js:349`）；0.2.2-alpha.2 消失，替代为 `installModelSelection(agentCtx)` + `session/selectModel` | 我见 | 待设计兼容 |
| 6.2 | 模型写路径单入口：现在由插件在 turn 边界写一次（不用 `selectModel`，它会改全局默认） | 旧插件注释 | 待保持 |
| 6.3 | `permission` preset → `sandbox/mode`（旧插件）；新包 `sessionPolicy/setSessionPolicy` 是两轴 | 我见 | 低优先 |

## 7. 测试与模板

| # | 操作点 | 证据 | 状态 |
|---|---|---|---|
| 7.1 | `test_dolores.py`（4116 行）过度依赖 fake；改为**真实数据 + 只 mock 实现层** | 他 | 待做 |
| 7.2 | `stubs/` 与 `dsh_stubs/`（profile/plugin 模板）跟代际对齐；沿用 home → stub | 他 + 我见 | 待做 |

## 8. 版本基线（核心）

| # | 操作点 | 证据 | 状态 |
|---|---|---|---|
| 8.1 | 基线 = 0.2.0-rc.2；`DSH_VERSION` 只作文档声明，不作运行时门控 | 新包 `__init__.py` | 已定 |
| 8.2 | 0.2.2-alpha.2 已见漂移：selection API 改名/移位；`ArchivedSessionGate`（reject archived session 的 `agent/pre-step`）只在 0.2.2-alpha.2 见到，rc.2 未核 | 我见 | 待核 |
| 8.3 | 升级路径：能力探测 vs 版本门控 | 待定 | 待定 |

## 与 todo.md 的关系

`todo.md` #0（bypass 不落日志）、#2（ego workspace 自动归档）、#3（去掉 未分类/home 两块）与本文件 §4.3 / §5.1 是同一条根因链。集成时一并处理，不在两处各做一份。

## 未决问题

1. `file-reference-local` 关不关 —— 取决于 ego preset 是否有 `read`（其 prompt section 条件求值）。他：先不管。
2. 声明机制落在 pyproject entry points 还是包内注册表。
3. "插件覆盖" = replace 现有 row，还是 init=False 只装插件。
4. 断连 = 整 ghost 故障，还是重试 N。
5. `.engines/` 分块：按 dsh 代际还是按能力（我推前者，未经裁决）。
