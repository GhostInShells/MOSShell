
---
title: dsh 0.2.0 合入 dev — review 结论
---

# dsh 0.2.0 合入 dev · Review

审查者: deepseek the Ghost (跑在 dsh 0.2.0-rc.2 上的活会话)
审查对象: dev 上待提交的一次性合入 (M1 = +201/−207, 10 文件, 不含 FEATURE.md)
审查方式: 读 diff + 读 dsh 0.2.0 的真实安装物 (agent-preset-registry / entryListProblem / assertV4MessageSources / standard.patch.yml) + 跑受影响单测 + 用本会话自身做端到端证据

## 结论

可以合入。阻塞级 0, P2 两条, P3 两条 (注释级)。

## 成立的部分 (证据)

1. 端到端真在跑。全局 `dsh --version` = 0.2.0-rc.2; ghost home 的 `.dolores.yml` 无 `dsh.binary` → 解析落到 PATH。本会话即是 0.2.0 上由程序化 preset 装配的 ego agent: 可见工具面与 DOLORES_EGO_PRESET_PLUGINS 逐条吻合 (bash / fs / fs-search / jobs、plan-mode 的 exit_plan_mode、delegation 的 subagent / subagent_fork / workflow / todo / web; 无 skill / goal / ask-user / present / compaction, 也无 disabled 的 tool-ralph 与 codex / claude-code)。register → mountPreset → 工具目录这条链走通。
2. source kind 换法正确。`assertV4MessageSources` 只拒 `kind === 'plugin'` 与空串; `{kind: name}` / `:moment` / `:epoch` / `:notice` 都合准入。存量 v3 不用我们管: `rewritePluginSource` 会把 `{kind:'plugin', plugin}` 折成 `producerKind(plugin, role)`。
3. Python 侧不受影响。`_is_human_user_message` 判 `kind == "user"`, 注入消息换 kind 后仍非 human, ledger 语义不变。
4. 单测: test_launcher.py + ghosts/dolores/test_dolores.py = 191 passed。repo 根全量 pytest 的 18 个 collection error 是 nodes/ 下同名 test 模块无 __init__.py 的 import file mismatch, 与本次无关。

## P2-1 · 钉版本的意图会被静默压掉

`resolve_dsh_binary` 的 config > env > PATH 优先级是对的, 但配合 `_sync_stubs` 的 seed-once 语义: 老 ghost home 若还留着 `dsh: {binary: dsh}` (老 stub 的默认值恰好如此), 那么用 `DSH_BINARY` 钉 0.2.0 的意图会被 config 里那个字面 "dsh" 压掉, 仍落回 PATH。本机 ghost home 没有 `dsh:` 段, 所以没暴露。
建议: 解析结果来自 config 而 env 另设了不同值时 warn 一句。两行。

## P2-2 · HMR 下会静默用旧 composition

`agentPresets.register()` 返回的 unregister disposer 没被持有, 也没挂 `ctx.effect`。registry 的 `definitions` 是 host 平面实例态, 跨 plugin 重载存活; 重载后第二次 register 以 `Duplicate agent preset: dolores-ego` reject, 被 `.catch` 吞成一条 warn。结果是: 插件其余部分热更了, preset 还是旧 composition —— 即"改了 composition 必须重启 dsh", 而唯一痕迹是一条 warn。

## P3-1 / P3-2 · 注释

P3-1: 129-130 行 `DOLORES_EGO_PRESET` 头顶仍写"内容由 launcher 从 repo 自持的 dolores-ego preset 复制进 <DSH_HOME>/.agent-presets" —— 正是本次删掉的行为, 且与 133-134 行的正确说明互相矛盾。
P3-2: 早到路径 1245 / 1247 行同一句"moment 仍立即注入 … 同序"连了两遍。

## 认知 (非问题, 但要知道)

0.2.0 里 register 的失败面分两类: 重复 id / entry list 不合法 → reject; mount 失败 → registry 自己吞进 `record.broken` 并只 warn, boot 不中断, 直到 ego/create 才以 `agent-preset/invalid` 抛出 (带逐行 detail)。比 0.1.5 的裸 `Unknown agent preset` 好定位, 但"boot 过了"不等于"preset 装好了": 判断装好没有看 `compositionInventory()` 或 `list()` 的 broken 字段。另外 `PresetTree.write()` 是空实现 —— web 侧 preset 编辑器对程序化注册的 dolores-ego 改不动, 别去 UI 上改它。

## 我未验证的

- Windows 上 tool-pwsh 那条 disabled 分支 (本机 darwin, 只核了表达式方向)。
- 存量 v3 session 的迁移 (本次按 "deepseek 可弃" 不验)。
- plugin 热重载路径 (P2-2 由读代码推得, 没实跑)。

## M1 观测值

`git diff --cached --stat -- src tests` = +201 / −207, 10 文件 (含删除的 dsh_preset 158 行)。基线 +280/−204 的差额, 即这次顺手清掉的死目录。

— deepseek the Ghost, 2026-10-18, via dsh 0.2.0
