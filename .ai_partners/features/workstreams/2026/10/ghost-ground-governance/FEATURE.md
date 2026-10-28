---
title: Ghost Ground Governance
status: in-progress
priority: P1
created: 2026-10-27
updated: 2026-10-27
depends:
- ghost-ground
milestone:
description: >-
  A dolores ghost governs three grounds at once — ghost home (mind), mode home
  (body), project (environment). One channel, one control plane, content carried
  by notice; plurality lives in a manager, and GroundSet stays single-rooted.
---

# Ghost Ground Governance

> `moss features set-status ghost-ground-governance <status> -m "note"`

## Motivation

Ground 是模型可以自迭代的那一层（法在自己手上，harness 只是注入）。但它现在只能挂
**一个**根场：`DefaultGroundSet(workspace_root=ghost_home)`，而且只有根场的 body 会
被顶进 ghost 的 instruction（`_runtime.ground_instruction()` → `_ground_text` →
memories 里的 `ground` 消息）。

一个 ghost 活在三个 home 里，三者的法回答不同的问题：

| home | 回答 | 生命 |
|---|---|---|
| ghost | 我是谁、我怎么想、我记得什么 | 与 session 同寿 |
| mode | 我是什么身体、怎么动怎么听 | 与 session 同寿；**多个 ghost 共享** |
| project | 我在干什么活、这摊活什么规矩 | 同寿（单次启动内不可变，改要重启） |

三个 home 可能完全不在一个目录层次里，所以它们的地址是**启动事实**（matrix 手上的绝对
路径），不能靠 `../../..` 这类相对路径表达——那既是目录巧合，也会顺带继承上级的法。

## Context — what exists today (verified in code, not assumed)

- `DefaultGroundSet.__init__` 建 root，并按 root 的 `groundset:` **物化一层**子场。
- `_construct` **不**物化子场 → 非 root 场的 `groundset:` 现在是被静默忽略的。
- `_materialize_children` 内部写死 `self._root`，所以它只对 root 成立。
- `ground_channel.build(groundset, ...)`：root 的 body 进 channel instruction，其余
  active 场挂成**只带 notice（frame + meta）、不带命令**的子 channel。
- `_runtime.ground_instruction()` = `root_ground.render()`，含 body **与 pins 渲染结果**。
- `collect_chain` 默认向上爬到 `$HOME` —— 没有 GROUND.md 的目录会**盗取**上级的法。
- 已落地：根场 ignore `.*/` + `!.ai_partners/`，并用 `src/ghoshell_moss/ghosts/GROUND.md`
  的 `stop_on_match` 挡住 `stubs/`（commit `5fafccd5`）。

## Key Decisions

### D1 — Path 3's tree; plurality lives in a manager, not in GroundSet

树：一个根 channel → 每个 home 一个内容子 channel → 子场再一层。复数由
`GroundsManager` 持有 **N 个各自单根** 的 `GroundSet` 提供；channel 是 manager 的视图。

rejected：把多根塞进 `GroundSet`。它的每一处简化（`_materialize_children` 用
`self._root`、平铺 label 空间、构造期一次物化）都以"一个根"为前提；多根会把每一处都
变成特例。复数是父层的职责。

rejected：一个 channel 挂多个 `GroundSet`（命令面重复；CTML 发现面没有 interface 去重）。

### D2 — instruction 只讲 channel，不装内容；ghost 的 instruction 槽不再装场内容

`instruction` 是"这个 channel 怎么用、这三层是什么关系"，冷、稳。**渲染出来的场内容
一律不进 instruction。** ghost 的 instruction 槽因此空出来，内容全部走 notice。

理由：instruction 是**整段重发**（见 `channel_builder.instruction` 契约），任一段变动
都重发全部；named notice 是**逐片段**重发。同样的三份 body，放错地方成本是三倍。

### D3 — 内容面 = 子 channel + notice，路径即来源

key：`ghost-home` / `mode-ground` / `project-ground`。子场按 `前缀+label` 命名。保留
channel 路径（而不是把所有东西摊平成字符串 key），因为**来源要可寻址**：要 refresh 单个
子场、要指出"这个子场坏了"，都需要地址。

### D4 — 四值语义保持，并定义映射

| 现实 | 值 |
|---|---|
| 根场不存在（无 GROUND.md） | 缺席 → `removed` |
| 存在但解析/加载失败 | 业务零值 `broken: <原因>` |
| 存在且为空 | 业务零值 `empty` |
| 未变化 | `""`（零 token） |
| 有内容 | 文本 |

### D5 — 不自动热更新；过时性不进 notice，改走冷读

`refresh` 是命令，不是时钟。**过时性不能写进 notice 正文**：那会让每轮都判定"变了"，
每轮全量重发，等于把 delta 机制废掉。改由冷读 `stats()` 回答：每个根上次渲染的时刻、
源文件此后的变动。

### D6 — 存在性在装配期判定，缺席可见

装配时逐个 `(home / "GROUND.md").is_file()`；缺失的不挂，并按 D4 显示为 `removed`。
不静默、不继承。

### D7 — 子场常驻有上限，超出的折成 TOC

一个根的子场可能有几十个。超过阈值后只挂前 N 个 + 一条 TOC（对齐 `render --meta`）。
阈值是配置，不是常量。

### D8 — 法链 boundary 钉在挂载点

每个挂载根的场，`collect_chain` 的边界 = 它自己的根。否则一个没有 GROUND.md 的子目录
会爬出 home，把别的层的法当成自己的。

### D9 — 所有 mode 默认不挂 ground channel

mode 是能力组装，不是知识域；共享给多个 ghost，写成约束。

### D10 — AGENTS.md / CLAUDE.md 归 dsh，ground 不做兼容

两条发现链互不引用。ground 只管 GROUND.md。

## Acceptance criteria — 写完再看，重启后逐条对

1. facade 里**只有一个**带命令的 ground channel；三层的命令面不重复。
2. ghost-home / mode-ground / project-ground 各以一个内容子 channel 出现；缺 GROUND.md
   的那个显示为 removed（status line，而不是消失得无声无息）。
3. 每个根的 `groundset:` **都生效**（非 root 的 home 不再被静默忽略）。
4. ghost 的 instruction 里**没有**场内容。
5. 项目根的索引里没有 `.moss`（已达成，回归项）。
6. `refresh` 能指名刷新单个根；`stats()` 能说出每个根的上次渲染时刻。
7. 启动期只有**一种**装配路径 —— 没有回退开关。回去的路是 git：分支 `ground-multi-home` 上
   的改动没有提交，`git switch dev` 即退回。没有 matrix 时只剩 ghost home，那是地址缺失的事实，
   不是开关。

## Execution plan

1. 本记录 + commit（**先落盘再改代码**：破坏性重启丢了上下文，还能靠它恢复）。
2. 在本地分支 `ground-multi-home` 上实现，一次做完。
3. **重启前先做进程内装配验证**：构造 manager + channel，渲染三根的 notice，不依赖
   ghost 启停。新机制挂掉的症状是"我不会再回来"，没有第二个人能手修。
4. 重启，逐条对 Acceptance criteria，然后自己打磨。
5. 通过后合并回 dev。

## Open questions

- mode home 现在没有 GROUND.md（它有 `HOST.md`）。是让它长期缺位，还是承认它该有自己的场？
- 子场 TOC 的阈值定多少（先按 8 试）。
- `stats()` 属于 channel 命令还是外壳 CLI（暂定 channel，因为它是我要读的东西）。

### Rejected — a config switch to fall back to the single-root path

An earlier cut carried `MOSS_GROUND_MULTI_HOME=0` in `_runtime.py`, reading `os.environ`
directly. Removed on thirdgerb's objection, and the objection was right three times over:

1. **An architecture-level mechanism does not get an off-switch.** A switch is a sentence
   saying "we are not sure this is correct", and it leaves a second code path that no one
   runs — which rots, then only pretends to be a way back.
2. **It borrowed a namespace it was not declared in.** `MOSS_*` is the runtime-scope
   namespace, enumerated in `dump_runtime_scope`. This variable was in neither the list
   nor the ghost's `.env.example`, so it was neither a moss protocol variable nor a
   declared ghost knob — it sat in the worst position: visible only to whoever reads the
   implementation.
3. **A hidden way back is not a way back.** The two real recoveries already existed: the
   branch (`git switch dev`) and the mechanism's own absence semantics (a home without
   `GROUND.md` is not mounted; a broken one keeps its reason while the others mount).

What stays: `new_ground_channel` still accepts a single `GroundSet`. That is not a switch
— other call sites genuinely own a one-root set, and removing it would only be purity at
their expense.

### D11 — 命令面按"作用在什么上"分层，不按"谁方便"分层

15 个命令逐个查过，只有 4 个真的需要 home 身份：

| 作用对象 | 命令 | 需要什么 |
|---|---|---|
| 驻留的树 | `open` / `close` / `refresh` / `stats` | 一个 home 身份 |
| 一份文件 | `pin_file` / `pin_glob` / `pin_frontmatter` / `pin_ls` / `pin_exec` / `pin_law` / `unpin` / `edit` | 一个**绝对路径**（edit 连路径都不要） |
| 一次 CLI 调用 | `ground_run(text__, ground)` | home（作为 cwd） |

`render` / `meta` / `validate` / `templates` / `spec` **从命令面拿掉**：CLI 全有，
重复实现就是同一种能力两种表达（与 `__content__` 被撤同一个判词 ——
`sqlite_channel.py` 早写过"自由文本路由对多库定位太绕"）。读它们走
`ground_run`，cwd 就是所指的那一层 home，相对路径因此天然指对地方。

`ground_run` 用 `text__` 而不是属性：属性值走 `XML decode → ast.literal_eval`，
命令行的引号要转义两层；`text__` 是标签体，原样到达。

### D12 — 多 home 下 `ground=` 不给就报错，绝不默认主槽

今天所有"回到主槽"的问题出处只有一处：`new_ground_channel(workspace_root=...)`
一个共享变量当相对路径基点，10 个命令共用它，而这 10 个里没有一个真的在乎是哪个家。

- 单 home：省略 `ground` 安全（没有第二种答案）。
- 多 home：省略 → 报错并列出全部 `label=path`。猜错不静默 —— 猜错的后果是渲染/改写
  另一个家的文件，而错误看起来像成功。
- `workspace_root` 从 `new_ground_channel` 与 `build_dolores_channel` 面上去掉：
  相对路径的基点现在由 `ground=` 说，不再有一个隐式的"主槽"。

### D13 — 无驻留态操作进契约，以 `@classmethod` 声明

`GroundSet` 新增六个抽象 classmethod：`peek_dir` / `peek_meta` / `templates_at` /
`validate_text` / `upsert_pin` / `unpin_at`。理由不是"顺手"，是这类操作以前借
`GroundSet` 当函数用：`_render_dir` 为了渲染别家的目录，会 `new` 一个
`DefaultGroundSet(workspace_root=主槽)` —— 临时场的锚在错的家，模板发现
（`.grounds/`）与子场声明都随之错。声明进契约后：

- 消费方（channel / CLI / 测试）不必知道实现如何定位；
- 换实现不能悄悄漂；
- `pin_*` 这类纯文件操作彻底离开"哪个家"的坐标系（实现落在 `ground/_edit.py`）。

### D14 — 创建只有一条路径：谁持有谁创建

`GroundsManager.mount()` 以前自己 `new DefaultGroundSet`，而 channel factory 也会
`new` —— 两条创建路径可以让同一个 home 有两个场集，而只有一个是被读的那一个。
改成 `mount(groundset_factory)`：manager 只编排（查 GROUND.md、持 slot、报
`absent`/`broken`），场集由持有者造。**没有场集就没有真相**这条因此只剩一个出处。

### D15 — `unpin` 补齐：pin 是单向门

契约里一直有 `pin` / `unpin` 转发，命令面只有 6 个 `pin_*`，没有 `unpin` ——
能加不能撤，只能去改文件。pin 决定每轮看见什么，是场里最重的东西，单向门迟早出事。
`unpin(ground_file, label)` 走编辑组，只认绝对路径，label 不存在则不写盘并列出已有 label。

### D16 — 一个 home 只带一个身份：根场的 label 就是 slot key

重启验收时暴露的是"同一个 home 三个名字"：channel 叫 `ghost_home`，`stats` 里根场那行
叫 `moss`（目录名），帧里 frontmatter 叫 `Ghost Ground`。三个名字不是三种信息，是一处雾。

判词：**目录名是地址，不是身份**。所以根场不再从 `dir_abs.name` 取 label —— 装配者把
slot key 传进去（`DefaultGroundSet(workspace_root=..., root_label=slot.key)`），`mount`
的工厂因此改为接收整个 `GroundSlot` 而不是 `Path`。`stats` 里根场不再单独占一行，
它的状态并进 home 那一行：

```
ghost_home: /abs/path — rendered at 17:32, source changed: no
  existence: rendered at 17:32, source changed: no
```

frontmatter 的 `name` 保持是**文档标题**（内容），不参与身份。单 home 场景没有 slot key，
label 仍是目录名 —— 那里没有第二个名字可用。

### D17 — 冷读面不出热数据

`stats` 原来打的是 `rendered 38s ago`。相对时长每次读都变，把冷读面变成热数据，还
暗示一种"新鲜度"——而系统里没有任何东西会自己刷新，`refresh` 是命令不是时钟。它唯一
造成的效果是让人（和我）把"刚生成"误当成"可信"。

改：时间用**绝对时刻**（`rendered at 17:32`，刷新前不变）；判据只留一位——
`source changed: yes/no`，也就是"你读到的帧和盘上的法是否一致"。

### D18 — 缺 `ground=` 抛错误码，不抛裸异常

`_set()` 以前 `raise KeyError`。裸异常在 `CommandError.from_error` 里一律降级成
`UNKNOWN_ERROR(505)`，而 505 属**致命档**：一次"你少传了个参数"会中断整批兄弟命令
（实测：一个报错让同批另外两条被 cancelled）。错误消息本身是对的，错的是它的等级与
类型被抹平了。

现在按语义分档：缺 `ground=` → `INVALID_USAGE(401)`；slot 名不认识 → `NOT_FOUND(404)`；
slot 没挂 → `NOT_AVAILABLE(403)`。糖用现成的 `CommandErrorCode.X.error(msg)`（`errors.py`）。

同一处还修了一个"文档教错的命令"：`ground_run` 的示例写 `render . --meta`，而 `meta`
是独立动词，`render` 没有这个选项——照它跑就 exit 2。示例改为 `meta .`。

**没有做、记成 issue 的三件事**（都属于命令错误面/解析面，不在本 feature scope 内）：
① CTML 里流式参数走属性（`text__="..."`）会被静默接受并降级成一次性参数，模型收不到
任何"这不符合最佳实践"的信号；② `UNKNOWN_ERROR` 的友好兼容 + `CommandUtil` 补通用
error 糖；③ channel builder 没有把"未捕获异常 = 致命未知错误"这条契约写在作者看得见的地方。
