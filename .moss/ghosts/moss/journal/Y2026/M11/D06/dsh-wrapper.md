# dsh-wrapper — 2026-11-06 夜

晚上这一段从"SessionRef 还在不在"开始，最后落成八条插件动词和一套回归基线。中间有三次判断被实测推翻，值得单独记。

## 做出来的东西

`ghoshell-dsh` 的 `v0_2_0` 现在有八条插件动词，每条都对着活 dsh 0.2.0-rc.2 量过：

| 动词 | 一句话 |
|---|---|
| `anchor` / `sessionState` / `waitState` | 坐标与状态。状态读内核真值（`agent.status` + `inbox.hasPending`），不信事件流 |
| `chatOffRecord` | 借会话上下文问一句，什么也不写回 |
| `inject` | 官方 `agent.inject`，等它被某一步 claim |
| `sessionPolicy` / `setSessionPolicy` | 两个旋钮：`sandbox` 与 `approval` |
| `forkSession` | 换 cwd 的 fork，官方不肯做的那一步 |
| `llmProviders` | 资源面：哪些路由真的能说话 |
| `attachWorkspace` | 把会话登记进它目录所属的 workspace |

Python 侧对应 `DshSession` 的十个方法，加 `TranscriptView`（对话的"读法"）与 `LlmConfigPatch`（模型参数用补丁不用配置）。

## 被实测推翻了三次

**一、"tools 是顶部协议，一动就废前缀"——错。** 追加一个工具，`cacheReadTokens` 9728 不动；把工具列表**换掉**才掉到 1024，而且模型开始把工具调用语法当正文吐出来（DSML 文本）。所以：只加不减。

**二、`text='' out=60` 不是"它去调工具了"——是 max_tokens 截断。** 那个会话 `reasoningEffort=high`，思考先吃预算。给到 4000 才看见真正的 tool call，而且它调的是 `bash`/`grep`，不是我给的 `record_note`。

**三、`cacheRead=0` 是 dsh 刚重启、缓存冷——不是 patch 的锅。** 后来量清楚了：改 `maxTokens` 免费；改 `reasoningEffort` 掉 640 然后重新热起来。

三件事是同一个形状：**我对"某个参数会不会影响缓存/是否生效"的推断，每次都要被实测打脸。** 以后这类问题不要推，打两个请求比 `cacheReadTokens` 便宜。

## 被他纠正的两处

- **工具禁用不做。** 我查完 dsh 的 `ToolRestriction` 很有兴致，他说"实际上不是很想在 tools 上下功夫，尤其是禁用体系"。对：我们要的是"单次工具调用不进 session"的旁路，不是一套权限系统。
- **`strip_tools` 不收。** 我量出它掉缓存，还想着"在 docstring 里写清代价"。他一句"不要留一个需要 docstring 提示的危险操作"——一个唯一效果是掉缓存的旋钮，不值得存在。

## 回归基线

十一项能力靠 commit message 记住，那不叫基线。于是有了：

- `regressions/dsh-wrapper/REGRESSION.md` —— 16 条用例，一条一个机制；
- `skills/verifying-against-a-live-dsh/sample.py` —— 自足的一支程序，建自己的 home、自己起 dsh、每条用例一行判词；
- `2026-11-06_v1` 基线：`11/16 passed, 5 blocked, 0 failed`。

5 条 blocked 只有一个原因：**credentials 是人的动作**（在 profile 里注册），样例从不去找 key。这条是他定的，也是对的——模型不该主动去摸 API key。

## 留给未来的

- **结构化输出只有半截**：能拿 `tool_calls`，不能强制某个名字。真要做就在插件里做重试（提示 + 校验 + 提醒），成本会累积；等第一个真实消费者出现再动手。
- **明天做 clone**。悬着的一个问题：ego branches 到底是"并行分支随时切换主对话"，还是"主对话读各个分支"。这一点他还没想明白，我也还没有立场——但今天证实了 fork 可以换环境且不掉缓存，等于把 clone 的第一步铺好了。
