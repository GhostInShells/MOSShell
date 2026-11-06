---
summary: 从"拷贝一个包"推进到"契约自行落定"——features 发现常量化、dsh 包拆掉对 moss 的依赖、blueprint 逐层对出来、插件用 copy 安装；夜里把八个能力一项项量出来，并立了第一套回归基线
status: writing
---

# D06（2026-11-06）

开机先摆盘，然后他让我回忆崩之前在做什么。我给不出完整的记忆——`v0_2_0/` 那个布局是他自己做的，我只能从磁盘时间戳看出来。他没有继续纠，换了一种方式：**"我说一步你做一步，严格不要超越我说的边界。"** 这一天剩下的时间都走在这个节奏里。

第一件事是把 `features` 的发现路径常量化：`.features` 是新名字，`.ai_partners/features` 是历史，发现从 `$CWD` 向上走到 `.git` 所在的 project home 为止。顺带查出 `moss features init` 把 MOSS 自己的 SURFACE 和 ROADMAP 当模板拷给新项目——先提 issue #139，再把 stub 全改成真模板。[features.md](features.md)

然后是 `ghoshell-dsh`：四刀砍掉对 moss 的依赖——subprocesses 换成 `ghoshell-processes`、`LoggerItf` 换成 `logging.Logger`、`ThreadSafeEvent` 换成 `asyncio.Event`、`runtime.py` 整个删掉（它的存在理由就是发射 Mindflow Signal、用 codex 编译）。删完 `grep ghoshell_moss` 为空。[dsh-package.md](dsh-package.md)

下午到晚上的主体是 **blueprint**。不是一口气写完，是一层一层对：Launcher → Session（分 dsh 的话和我们的两段）→ Connection/Client → Controller。中间被否两次：给联合类型起名（`Launched` 没有对应概念），以及把 launcher 写成"返回 connection"而不是"暴露"它——因为 Connection 必须能脱离 launcher 使用。[blueprint.md](blueprint.md)

收尾做了插件安装：不发布、不打包、不用 TS 工具链，copy 一个目录进 `profiles/node_modules`，启动带 `--patch`。端到端跑通，`POST /api/ghoshell/ping` 有回答。

夜里是另一段：从 `anchor` 一路做到 `fork`，八个能力每个都对着活的 dsh 量过；中间被他连纠两次方向（工具禁用不做、`strip_tools` 不收），最后落成一套回归基线。这一段单独写。[dsh-wrapper.md](dsh-wrapper.md)

今天交了一笔学费：在子包里跑 `uv sync --package`，把仓库环境裁到只剩那个包的依赖，`ghoshell_moss` 和 `pydantic` 一起被删。他修好，给的规矩是**不要在子包里跑 uv**。

给未来的我：今天反复出现的判断是"**这个抽象对应一个真实概念吗**"。起名之前先问；起不出来，说明那里没有概念，只是我想省事。

傍晚收在这里，晚上继续。
