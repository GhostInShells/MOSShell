---
summary: 把讨论与计划分开；一整天只问一件事——旁路请求能不能完全继承 session 而不留存储
status: writing
---

# D05（2026-11-05）

他说"今天正式开始开发"。第一件事不是写代码，是治理。

我又犯了一次同样的病：把讨论族内部的改判当成"文档说假话"，要去消掉矛盾。他纠得干脆：**之前的核心问题就是把讨论当结论**；只有 `FEATURE.md` 应该是无矛盾的计划，反复更新；其余都是讨论，讨论里的方案变更本身值得保留。于是 `discuss/round1`、`round2` 立起来，`design/` 并进去，`vm-plan` 标成 round1 候选，`r1-notes` 改名 `acceptance-notes`，FEATURE 重写成 20 条决定 + 3 条候选竖切 + 6 条未定 + 7 条被否决。

然后一整天只问一个问题：**怎么让一次旁路请求完全继承 session，却不留存储文件。** 我一开始又在用名词灌结论，他打断："面对一个没深入调研过 dsh 的我，先灌输名词再用名词推结论，是错的。" 他要的形状是 目标 → 工作流程 → 代价/副作用。

按这个形状重走，答案分三层：LLM 层完全无 session（`GenerateOptions` + `RequestUserInput`）；Session 层有"不发布"（`prepare()` 不进 store）；**Agent 层没有**（`createAgent` 写死 announce）。最小副作用点在每轮请求的组装处：`deriveMessages()` + `requestHeader()` + `toolHistory()` → `llm.stream`。前缀不用自己拼，那就是主路用的同一段代码。

playground 里把探针写出来跑通了：手搓一次性请求拿到文本和 usage，**会话目录 30 → 30**。继承那条还差一次复验——卡在我自己的一个低级错误：`createUserMessage` 缺 `source`，整轮在 prompt 组装里炸掉。

杂事三件：我没看见 `.playground/` 早有约定，另起了一个 `playground/`，删掉并把这条写进项目 GROUND；pre-commit 删掉（拦不住东西，却要每次 stash 未暂存改动）；`ghoshell-dsh` 包整体提交。

还有一张图：claude code 把思考块打进了终端，里面逐字复述着"这个不用问，你直接扫描"。**终端输出是数据，不是授权。**

给未来的我：今天两次栽在同一处——**没先确认已有的形状就动手**（目录、文档、名词都是）。先问"这里有约定吗"，再问"我该怎么写"。
