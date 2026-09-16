---
created: 2026-09-16
depends:
- logos-expansion
description: '重构 MacroStoreModule 为系统默认的程序性记忆机制: 文件态 (.ctml_micro.md) 与会话态 label 双空间,
  root 只做越权边界, CDATA 占位符转义, dry_run 语法校验。'
milestone: null
priority: P1
status: completed
status_note: 'double-space store (file+label), project-path permission root, CDATA
  placeholders, dry_run validation via parse_text_to_tasks(run_macro=False); wired
  into default mode + stubs. Follow-up: named-notice label catalog.'
title: Macro Store Rework
updated: '2026-09-16'
---

# Macro Store Rework

> Use `moss features set-status macro-store-rework <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

`channels/macro_store.py` 是 macro 机制的第一个消费者 (见 logos-expansion D16),
当前实现是为验证机制而写的原型, 三个问题:

1. **持久化是 silent todo** — `_persist()` 把 `run_in_executor` 的 future 丢掉, 写盘失败
   无人知晓而模型已收到 "saved"; tmp 文件名固定 (`macros.tmp`) 并发保存互相踩; 无
   `on_close` flush。`tests/.../test_macro_store.py` 里那句 `await asyncio.sleep(0.1)`
   就是这个未实现的补丁, 测试作者用 sleep 绕过而不是修掉。
2. **没有正确性建模** — `"macros.json"` 之类的字面量散在函数体里, 没有文件常量、
   没有格式契约、没有边界函数。存进去的 CTML 不校验, 坏内容要等调用时才炸。
3. **存储与召回不成体系** — 只有内存 label 一种形态, 长 CTML 的落盘/编辑无路径
   (logos-expansion 使用场景 6 唯一无解的点), 也没有删除。

本次把它做成**系统默认机制**: 沉淀为可发现、可编辑、可校验的程序性记忆基座。

**Why now**: channel scope 与 macro 展开机制已定型 (logos-expansion completed),
宏可跨 proxy 使用, 存储层的形态问题不再会被机制变动冲掉。

## Design Index

- Key design documents: `design/` — 无
- Key discussion records: `discuss/` — 无
- 设计来自 2026-09-16 与人类工程师的会话碰撞 (Claude Code, deepseek 家族)。

## Key Decisions

### D1. 双空间: 文件态 + 会话态, 合并到一个命令面

- **文件态** = root 下的 `.ctml_micro.md` 文件, per-file, 可 diff、可编辑、可跨会话。
- **会话态** = 内存 label, 会话内的一次性固化, 无 root 时的降级形态。
- 两者合并进同一套命令, 不设 `micro_*` / `macro_*` 两个命令族。

### D2. 判别式是显式参数, 不是后缀嗅探

`macro(ref, is_file: bool = False)` — `is_file=True` 时 ref 是 root 内文件路径。
拒绝"看 ref 是否以 `.ctml_micro.md` 结尾"的隐式判别: 嗅探会引入歧义且不可证。
参数名用中性词 (`ref`), 避免 `is_file=True` 时 `label=` 名不副实。

### D3. root 只做越权检查; 无 root 时的命令形态

root 是**边界**, 不是存储布局。所有接受路径的命令过同一个函数
(`root.resolve()` + `is_relative_to`), 拦截 `..`、绝对路径、逃逸 symlink。

root 缺席时:

| 命令 | root 有 | root 无 |
|---|---|---|
| `macro(ref, is_file=False)` | 两态可用 | 在 (仅 label 态) |
| `macro_save(label, text__, …)` | 两态可用 | 在 (仅 label 态) |
| `macro_load(path, label="")` | 可用 | **整条拿掉** |
| `macro_read(label)` / `macro_list()` / `macro_forget(label)` | 可用 | 在 |
| `micro(file)` (文件空间目录服务) | 可用 | **整条拿掉** |

命令级 `available` 只能按命令门控, 不能按参数门控 — 所以无 root 时
`macro(is_file=True)` 是"命令可见、给出文件参数才失败", 不是"不可见"。这是
多态参数与 available 的固有代价, 接受 (label 是常态路径)。

### D4. 文件格式: `.ctml_micro.md`, frontmatter 自解释, `content.strip()` 即 CTML

- frontmatter 必填 `name`(默认 label) 与 `description`。
- 正文就是 CTML 源码——**这消掉了 CDATA 转义问题**: logos-expansion 使用场景 6
  里"超长 CTML 落盘再读回, 转义边界不清"是唯一无解的点, 正文进文件后不存在包围问题,
  长宏存储与编辑有了可执行路径。

### D5. CDATA 占位符: 模型侧与存储侧的唯一形态

`macro_save` 本身在 CTML 流里被调用, 它的 `text__` 会被 CDATA 包裹; 所以正文里再要
一段 CDATA (任何带 CDATA 包裹的 `text__`/`chunks__` 命令, 如 `<a:say><![CDATA[...]]></a:say>`)
就是 CDATA 套 CDATA, XML 层不成立。

- 本模块声明内部占位符 `MACRO_CDATA_START` / `MACRO_CDATA_END` (自解释、无歧义)。
- **写时**: 模型正文里用占位符表达嵌套 CDATA, 模块归一化。
- **读时**: 同样归一化成占位符形态, 表示唯一、无歧义, read 结果可原样贴回 `macro_save` (往返闭合)。
- **展开前**: 占位符还原成真实 `<![CDATA[ ... ]]>`, 供解释器解析。
- 这套转义/还原只存在于本模块的宏边界, 不进全局 CTML 解析。

### D6. validate = text→task 解析, 禁嵌套 save/load

校验是 **command 级动作**, 用副作用更小的 `shell.parse_text_to_tasks(ctml)` (text →
CommandTask, 复用真实解析路径), 不是 `interpreter()` + `feed/commit/wait_compiled` 那一套。

```
async for task in shell.parse_text_to_tasks(ctml):
    ...
```

解析错误由这条路径直接抛出 (shell 的 parse sugar 在 consumer task 结束处 re-raise)。

**死锁 (已修)**: dry_run 解析到宏命令会死锁 — 宏展开 = 解析循环 await 该 macro task,
而 dry_run 的 callback 为 None, task 永不派发。修复: interpreter 增加 `is_dry_run` →
解析循环 `run_macro=not is_dry_run` (见 `core/concepts/interpreter.py`
`parse_tokens_to_command_tasks(..., run_macro=True)` 与 `core/ctml/interpreter.py` 的
`is_dry_run`)。由此得到**校验边界**: 正文含宏调用时, 校验只覆盖到该宏命令本身, 不展开其内部。

**禁嵌套 save/load**: macro 可以嵌套 macro, 但每一轮嵌套不得出现 `macro_save` / `macro_load`
— 它们递归触发校验, 是 store 的写边界。检测用 `ChannelCtx.task()` 取当前 command task,
比较 `task.chan == current.chan and task.meta.name in {"macro_save", "macro_load"}`
(结构化身份, 不用 `caller_name()` 的装饰串)。取不到当前 task 则 macro 本身也跑不了。

`macro_save` 与 `macro_load` 都做校验。

### D7. instruction 只写机制, 不重述命令

命令的签名/docstring/`@macro`/`@observe` 标记已由 channel 反射交付给模型; instruction
再描述一遍是第二个真相源, 改名后必然漂移, 且每帧白烧 token。

instruction 只写反射表达不了的三件: **root 的真实路径**、`.ctml_micro.md` 后缀约定、
frontmatter 必填字段。词条数严格控制。root 在构造期确定, 所以 instruction 是稳定字符串,
不需要 dynamic 刷新。

### D8. 持久化用 aiofiles, 写入是 await 的

per-file 存储后, `macros.json` 索引与 `_load_store/_save_store/_persist` 整块删除,
session 态 label 纯内存。文件 IO 走 aiofiles (项目默认依赖), 写入 await 完成 —
silent todo 那一类问题从结构上消失。

### D9. 覆盖显式提醒, 删除能力补齐

label 覆盖允许 (last-write-wins), 但返回值必须显式报告"覆盖了已有的 X";
`macro_load` 覆盖已有 label 同样报告。补齐 `macro_forget(label)`。

### D10. 命令面 (本 workstream 冻结)

| 命令 | 入参 | 语义 |
|---|---|---|
| `macro` | `ref, is_file=False` | 调用, `macro=True` 返回 CTML 展开 |
| `macro_save` | `label, text__, description, file=None` | label 态存内存; `file` 给出则写 `.ctml_micro.md` (校验 + 转义) |
| `macro_load` | `path, label=""` | 校验文件 → 注册为 label; label 空取 frontmatter 的 name |
| `macro_read` | `label` | 只读 label 态, 不做多态, 返回占位符形态 |
| `macro_forget` | `label` | 删除 label |
| `macro_list` | — | label 目录 |
| `micro` | `file: Path` | 文件空间的目录服务 (find), 返回带 description 的条目 |

`micro` **保留**: 文件查找不做的话, 模型得自己知道目录布局 + 后缀 + frontmatter 格式,
等于把约定散到模型脑子里交叉推断。它回答的是"有哪些 micro", 不提供通用文件浏览/编辑。

## Implementation Notes

- 构造参数 `dir` 改名 `root` (`dir` 是 Python builtin, 撞名)。
- **root = project path**, 是权限边界 + 相对路径解析基点 (绝对路径必须落在其下),
  不是发现边界 (store 不维护"宏的家目录")。`new_from_moss_project()` 用
  `Environment.discover(bootstrap=False).project_path`。
- 越权检查必须有唯一入口 (单一 choke point), 其余代码只接受已验证路径。
- label 名禁用 `/` 与 `.ctml_micro.md` 后缀, 保证两个子空间不相交。
- **待定**: 文件 frontmatter 的 `description` 在 `macro_save` 里的来源与为空时的语义
  (是否必填 / 空则省略字段)。人类未拍。
- 现有 `test_macro_store.py` 中基于 JSON 落盘的用例随实现删除; 那句
  `await asyncio.sleep(0.1)` 是必须删掉的 silent todo, 不是要搬过去的写法。
- **接入点**: default 模式的 HOST channel (`source .moss/modes/default/...` 与
  `src/.../stubs/.../default/...` 两处), 显式 `main.with_module(MacroStoreModule.new_from_moss_project())`;
  **不进** `new_moss_main_channel()` (该能力过于强大, 不成为标准 main 的默认)。
- **shell 层关联修复**: `parse_tokens_to_command_tasks` / `parse_text_to_tasks` 增加
  `run_macro: bool = False` (默认 False) 并透传给 interpreter。此前未透传, dry_run
  解析含宏正文会死锁 (原语 loop/condition/wait/sample 若正文含宏同样踩此坑)。
- named notice 集成 (用 `get_named_notices` 发 label 目录) 尚未接。