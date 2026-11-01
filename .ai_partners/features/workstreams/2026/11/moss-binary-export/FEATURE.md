---
title: Moss Binary Export
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P2
created: 2026-11-01
updated: 2026-11-02
depends: []
milestone:
description: >-
  把一次调研后的复杂 MOSS 能力导出成一支焊死场景的可执行文件, 让只有 bash 的外部 agent
  不需要理解 MOSS 就能调用. 是历史经验沉淀与信息压缩的产物形态; 是否要做、做多深尚未定.
---

# Moss Binary Export

> Use `moss features set-status moss-binary-export <status> -m "note"` to update state.

状态: **draft** — 动机与形状已收敛, 边界仍有未决项, 实现为零。
本文件是 D31 夜 (2026-10-31 18:30 – D01 00:2x) 讨论的沉淀, 尚未有 `design/` 与 `discuss/` 目录。

D02 01:0x 补入一段: **启动器与发现链** (KD13–KD15)。结论是产物零样板、入口收归 workspace
顶层启动器; 附带一份同进程探针实测 (见 Measured Facts), 它把"0.5s 重在哪"重新定位到了
import + bootstrap, 而不是总线入网。

## Motivation

**目标**: ghost 治理一个 agent 集群 — 集群帮自己修装手脚, 接近 RSI 命题. 优先目标是
dsh (TS 版本). 见关联 workstream `dsh-agent-swarm`.

**真问题**: 给一支外部 agent 装 MOSS 能力时, 要同时懂三样 —

1. MOSS 自己的架构与 API;
2. 对方 harness 的架构与 API (而且还得是 python 侧接法);
3. 把两者接起来的集成。

每换一个 harness, 2 和 3 重来一遍。集群里 N 个 harness 就是 N 次集成。

**binary 删掉的是 2 和 3。** 它对目标侧的唯一假设是"你能跑一条 shell 命令"。
所以它不是"更便宜的通道", 是**把需要理解对方的东西, 从"它的架构"降到"它有没有 bash"**。
接口应落在目标能提供的最低通用基质上: bash < python < 任何 harness 自己的插件 API。

**判据 (thirdgerb)**: 不接受"ghost 控制 agent 集群, 但要二次开发才能建立能力传递"。
注: 这个机制不减少要写的代码 — 每份 plate 的语义仍要人/ghost 写; 它减掉的是**打包**
(CLI、参数解析、帮助文本、环境发现、bootstrap)。判据是: 打包写一次, 新能力只写语义。

**元价值**: 一次调研之后, 把复杂的 MOSS 能力导出成 skill 里的 binary — 本身就是历史经验的
沉淀与信息压缩。写过的 plate 留在项目里成为资产。

## Key Decisions

<!-- append-only; 编号只增不改 -->

- **KD1 产物两层, 两种寿命。** 源 (`plate`) 留在项目里积累、可读、可提交; 导出物一次性、
  焊死场景、不进 git。
- **KD2 焊死场景。** 导出时把 project id / ghost name / mode name / network / scope 烧进
  产物, 运行时**全部对上**才允许执行。比较方式用**排序后 hash** (一个短字符串), 而不是逐
  字段比对 — 好处: 不认识的字段也能校验; 失败时能报"你的 scene 是 ab12…, 这支是 cd34…"。
  `sys.executable` 也焊 — 放弃可分发, 接受。副作用之一: "目标有没有 python"这个问题消失,
  它之所以有 python, 是因为我们给了绝对路径。
- **KD3 导出物只有三行。** docstring + `Binary(name, scene=...)` + `binary.main()`。
  读这份源码的人只读得到这些。
- **KD4 `--help` 是源文件 `__all__` 的反射。** 头注释取 plate 的模块 docstring; 函数体抹掉
  (只渲染签名 + docstring); 并给出调用约定: 写 `async def main()`, import 已禁用。
- **KD5 `__all__` 就是边界。** 沙箱拒绝 import 并给明确失败; **反射面与执行面出自同一份
  `__all__`**, 所以"告诉你能用什么"和"只让你用这些"不可能漂移。(它是摩擦型边界, 不是监狱 —
  sandbox.py 自己写明 "not a security boundary"; 本设计威胁模型是防事故, 不是防恶意。)
- **KD6 编译 + 反射, 不走 AST。** 能拿到真签名与真 docstring。代价是一条纪律:
  **plate 顶层必须无副作用** — 否则 `--help` 变成执行。
- **KD7 全走 `async def main()`。** 一个进程只能起一次 Matrix (`_instance` 单例,
  `Matrix.new` 遇已有实例直接抛), 所以 Binary 自己吃完整个生命周期, 内部
  `matrix.run(caller_main())`。"只能起一次"与"一支进程一次调用"正好抵消。
- **KD8 Project / Matrix 不是分叉, 是包含关系。** wire ⊂ project ⊂ matrix, 上一层能造出
  下层能造的一切。所以这不是边界取舍, 是**延迟取舍**: 走 matrix 只慢 0.4s, 不会少功能。
  **先硬编码 matrix 那一支**; Project / wire 是之后的优化, 只改分支不改形状。
- **KD9 记账不进产物。** payload 自带 name / issuer / priority / description, 监听者订阅同一
  键即可重建全部账目, 且比发送方自报可信。topic 砍掉 (代价写明: 门上暂时没人)。
  最小值得做的不是账本, 是**回执** — 一次发送是否成立, 只有接收方能证明。
- **KD10 channel 面不做生成器。** channel 是开放面 (任意命令、任意参数、可能长跑或改状态),
  反射出来就是一支可执行面, 威力太大; 生成目标是 topic / signal (封闭词表, 带优先级与速率)。
  旁证: 总线不认识器官 — schema 里出现 session/turn/tool 就是越界。
- **KD11 身份由产物名字给出, 不由工具给出。** 实测血证: 探针脚本发来的消息以
  `<binary name="note">` 到达, 那是运输工具的名字, 发送者身份丢了。历史旁证: 已删除的
  memento agent 由文件 stem 推 owner (`calc/main`)。
- **KD12 调用方看不到实现。** plate 由 ghost / 人类写; 调用方连"有个 Matrix"都不该知道。
  函数体不是"没必要暴露", 是必须抹掉。
- **KD13 启动器是基础设施, 不是过渡件。** 发现链是
  `launcher → import ghoshell_moss → Environment.discover() → workspace → gateway / 总线`。
  第一环一断, 产物就只能写死 gateway 地址 (thirdgerb 的反例: 用任意 python 跑一支纯代码
  产物, 除了硬编码地址没有别的发现途径)。所以 workspace 必须自带一个启动器,
  "用哪个 python、环境在哪" 这份知识**只存在它一处**。
- **KD14 产物零样板: 不含 shebang, 不含发现逻辑。** 产物是语义本身 —
  `Project.discover().bootstrap().container.force_fetch(Gateway).rpc_call(api, body)`;
  入口统一走 workspace 根上的顶层启动器 `./moss bin <name>`。依据: 把布局约定
  (向上找 `.moss` 之类) 写进每个产物 = N 份独立演化的约定副本; 约定一改全部**静默**作废 —
  它们报的是"找不到文件", 不是"约定变了"。
- **KD15 判据: 产物的字节数应与它做的事成正比, 与它所在的环境无关。** 样板每复制一份,
  就是把一份知识从库里搬进产物; 搬 N 份之后改一次要追 N 处。这是"信息无法有效压缩"的
  实际账单, 也是 KD14 的量化形式。

## Measured Facts

今晚的实测, 别再靠推断:

| 层级 | 成本 | 拿到什么 |
|---|---|---|
| wire (裸 zenoh: open + put) | **0.010–0.039s** | 发布订阅, 无 cell、无 bootstrap |
| project (`discover()` + `bootstrap()`) | 0.135s (含 import 共 0.48s) | IoC 契约 (34 个, lazy generator) |
| matrix (`new` + `__aenter__`) | new 0.121 + enter **0.521** + exit 0.008 (含 import 共 0.85s) | 一个 cell: provide_channel / node 生命周期 / presence |

- **跨进程验证**: A 进程订阅、B 进程裸 zenoh 发布 → A 收到。
- **跨 agent 验证**: 一支跑在 `works/` 的 dsh agent 用探针脚本 `moss-signal` 发来消息, 成功
  进入 ghost 输入。
- 用 MOSS 自己的模型拼 payload 的探针: **端到端 0.995s** (wire 只占 40ms, 其余是 import)。
  → **便宜的只是 wire, 不在 binary。**
- **渲染面的硬事实**: 只有 `Message` 的信封与正文到达模型; `Signal` 的 priority / issuer /
  description **不渲染**。所以身份与任何必须送达的元信息, 只能走 Message 的 tag/name。
- `Matrix.new` 从 `sys.argv[0]` 推 node 名: stdin 与 `python -c` 会被 pydantic 正则
  (`^[a-zA-Z0-9_.-]+$`) 挡掉, 必须是真实文件; node 身份用传入的 name, 文件名不会漏到网上。
- 每起一次 matrix, 线上多一个 cell (`node__moss_probe_matrix__wkp8pz` 出现又关闭)。
- dsh 沙箱: 读不拦、写 EPERM、边界是 session 的 cwd; `ps` 也被挡。
- `MOSS_*` / `DSH_*` 环境变量全继承到 dsh 的 bash (MOSS_WORKSPACE / MOSS_PROJECT_DIR /
  MOSS_NETWORK_SCOPE / MOSS_MODE_NAME / MOSS_GHOST_NAME…)。
- 声明的 signal 词表只有七个: `aside / cell_event / command / input / interrupt / knock /
  notify` (`.moss/src/MOSS/manifests/signals/default.py` re-export)。
- 实测: `aside` 缓冲一个 moment 后到达、不打断; `input` 立即占用注意力。
  → 七个名字**都到**, 台阶决定的是代价, 不是可达性。
- `add_signal` 的实现在 zenoh_session.py 是**裸 put**, 且 docstring 明写"本方法不做限频 —
  连续高频调用会打满发布通道… 限频应在 Mindflow 的 signal ingestion 层实现, 而非 transport
  层"。**传输层被明确设计成不治理** — 这是"单向通道必须在存在前定边界"的代码级依据。

### D02 00:48 探针 — 440ms 到底在哪 (实测)

同进程, 单位 ms: `Project.discover() → bootstrap() → Matrix.new → async with →
service_operator() → get_services_by_kind('webview')`

| 环节 | 耗时 |
|---|---|
| import ghoshell_moss | 238.4 |
| Project.discover | 9.1 |
| bootstrap | 139.0 |
| Matrix.new | 63.0 |
| `matrix.__aenter__` | **9.1** |
| service_operator | 0.3 |
| get_services_by_kind | 1.3 |
| http 往返 (本地节点页) | **0.4–0.6** (首次 0.9–15.5) |

- 四个节点被同一个 `kind=webview` 发现: terminal / artifacts / file_editor /
  **host 自己 (dsh)** —— 发现面统一, host 也是节点。
- **与上表 0.521s 的 enter 冲突 (差 57 倍)。** 两个都是实测, 互不覆盖, 标为**待复现**;
  它决定 gateway 是"必需"还是"锦上添花"。
- **"重"不在总线上**: 238 + 139 + 63 = 440ms 全是 python 侧准备; 真正入网 9ms、
  找一个服务 1.3ms、一次往返 0.5ms。
- dsh web: query token + 裸 GET → **401**。token 登录后 query 被抹掉, 凭证只能走 header。
- 环境值 (实测): `MOSS_PROJECT_DIR` = 仓库根; `MOSS_WORKSPACE` = `.moss` 目录 ——
  要拼解释器只能拼前者。
- shebang 实测: 相对路径 → `bad interpreter`; 本机无 `/bin/python3` (只有 `/usr/bin/python3`);
  `#!/usr/bin/env python3` 可用; `#!/bin/sh` + 多行 polyglot (`""":" … ":"""`) 可用,
  且参数原样透传 (实测 `argv = ['./q.py', 'a', 'b']`)。

## Exploration paths

死路与教训, 按代价从大到小:

- **"binary 每次 0.85s、必须长驻 daemon" — 错。** 发信号根本不需要 Matrix; 裸 put 10–40ms。
  教训: 把一条具体路径的测量说成了一个类别的性质。
- **"把 channel 面反射成 binary" — 错。** 开放面, 威力太大, 反射还会膨胀。
- **"先建记账/票/超时/计数" — 先砍。** 撤销与归因在落点做更对; 发送方自报不可信。
- **探针脚本用 env 查 workspace — 错。** 后果是同一个文件谁跑就连到谁的世界 — 正是几轮前
  自己刚警告过的"静默连错地方"。正解: 生成时烧死路径 (或从 `__file__` 上溯找 `.moss`),
  运行时校验其存在; env 只能当最后兜底且要校验一致。
- **把运输工具名写进信封 — 错。** 见 KD11。
- **我最初把 `__all__` 说成"提示不是边界" — 错。** 它是边界; 它只是不是监狱。
- **"用 `$MOSS_PROJECT_DIR` 拼解释器" — 错。** 变量在普通 shell / cron / 另一个 agent
  直接 exec 时不在, 而那恰恰是最需要桥的时刻。`${VAR:?}` 会大声失败, `${VAR:-default}`
  会在变量为空时悄悄用默认值 —— env 型解析的专有失败模式: 它不报错, 它指到错的地方。
- **"产物自定位, 向上找 `.moss`" — 错 (技术可行, 但会腐烂)。** 实测跑通: 没有任何 MOSS
  环境变量、cwd 也不在 workspace 里, 它仍靠 `$0` 找到了解释器。但它把一份布局副本写进了
  每个产物, 约定一改全部静默作废。剪于 KD14。
- **"产物写死绝对路径的 shebang" — 半错。** 它能跑, 但让"workspace 搬家"直接变成
  "所有产物全废"; 而且它把 KD2 的焊死从 scene 扩大到了布局。

历史旁证 (值得读): `143633b5` 删掉的 memento agent —
`agents/memento_agents/calc.agent.py` 的文件即声明 (docstring = instruction, imports = 范围),
模型只拿一个通用 `sandbox_exec`, 而 `moss memento agent parse` 输出 narrative + verbatim
source + 工具说明 + **sha 行**。启发: **sha 就是身份与校验的最小件, 可以替代票表** —
对端拿到一行 sha、跑一遍、对一下, 不需要任何登记处。撤销也还在: 文件变了 sha 就变。

## Methods

```
moss binaries list                                  # plate 的模块 docstring 即说明
moss binaries export <name> [--path <target>]       # 生成一支焊死场景的可执行物
```

plate 的写法:

```python
"""给调用方看的 docstring"""

from ghoshell_moss.core.blueprint.matrix import Matrix

__all__ = ('send_to_ghost',)

async def send_to_ghost(text: str):
    """注释面, 给调用方看"""
    matrix = Matrix.discover()      # 进程单例, Binary 已经启动好了
    matrix.send_signal_to_ghost(...)
```

调用方 `--help` 看到的是同一份文件、函数体换成 `pass`、import 消失, 加一段方言说明
("写一个 `async def main()`, import 已被禁用", 后面跟可用名字)。调用方写:

```python
async def main():
    await send_to_ghost('hello')
```

**三个东西是同一份声明**: `--help` 展示的骨架、沙箱里可调的名字、调用方要填的东西 —
不是三份, 是一份的三种投影。

`moss binaries` 还兼任教学 — 教调用方自己写一支: `moss codex` 看 Matrix / Project 怎么用,
`moss manifests` 看环境声明、尤其 IoC contracts, 然后就能导出能力。

## Implementation Notes

**已知 bug (先修这个):**
- `core/codex/sandbox.py::_reflect_local_detail` 渲染函数时打印 `def {name}{sig}`,
  **不带 `async`**。本设计所有 op 都是 `async def`, 调用方必须 await — help 在关键处说谎。
  修法: 按 `inspect.iscoroutinefunction(value)` 加 `async ` 前缀; 类分支的
  `public methods` 那一行也应标出哪些是协程。
- draft `core/blueprint/binary.py` 里: `BinaryMeta.find()` 返回 `read_text()` (str, 应为
  `model_validate_json`); `clear()` 用 `rmdir()` 删非空目录 (里面有 binary.json / ops.jsonl)。

**最小编译流水 (据 KD7/KD8):**

```
Binary 校验 scene hash (不一致 → 拒绝, 报两边的 hash)
  → Matrix.new(name, description, category='binary', persist=False, singleton=False)
  → matrix.run(...):
        compile plate 源码成沙箱, 注入活的 matrix / project / session / container
        snapshot_base()
        aexec(调用方的 code —— 必须定义 async def main())
  → 退出
```

**可迁移的设计原则 (这几条不止用于 binary):**
- **被省略的信息, 模型会用常识补上。** 所以只能省略那些"默认值恰好正确"的部分 — 省掉
  `async`, 它不会问, 它会写错。
- **压缩的目标不是短, 是方差为零。** 展示的每一行, 要么显然相关, 要么被明确标成无关。
  抹掉函数体优于展示函数体, 不是因为短, 是把一整片方差删掉了。
- **声明面 / 实现面分开**, 且声明面同时是反射面与执行面。
- **单向、可抢注意力的通道, 必须在存在之前把边界定死** — 请求/响应可以宽, 因为它有往返去
  纠正。

## Open Questions

1. **Project / Matrix 的降级时机。** 现在决定硬编码 matrix (KD8); 什么时候降到 project 或
   wire, 未定。判据候选: 按 plate 实际命名的对象推断, 而不是声明一个 level 字段。
2. **stdlib 的暴露方式。** 默认不做。三条候选: 只放函数 (最干净, 最不能干活) / 标准库也进
   `__all__` (能用, 帮助变吵) / `#imports#` 标记切"帮助的渲染"而非切编译 (最便宜, 但引入
   位置相关的约定)。等第一支真需要的 plate 出现再定。
3. **资产位置。** `.moss/runtime/` 现在混着 `cells/ locks/ logs/ sessions/ media/` (运行态
   废弃物), 而 plate 是要积累、要提交的资产。要不要把源挪到 `.moss/binaries/` 或项目级
   `binaries/`, `runtime/binaries/` 只留导出物的临时落点。
4. **并发。** N 支并发 = N 个 cell + N 条 zenoh 连接。未量。
5. **回执。** 对称的另一个键 + 阻塞订阅即可, 不需要新机制; 但阻塞的那一端变成长活进程 —
   谁启动、谁看管、死了谁收尸。所以回执只适合"这条命令等一条答复", 不适合"一直在线"。
6. **瞬时节点能否不写 cell runtime info** (thirdgerb 记得可以, 未核实)。
7. **`--help` 的头注释语言与方言说明的措辞** — 尚未写。
8. **顶层启动器由谁生成、何时刷新。** 候选: `moss init` 时生成一次 / 每次 `moss` 启动时重写 /
   由 workspace bootstrap 维护。它自己怎么找到解释器 (生成时写死绝对路径? 相对 `.venv`?) —
   它是唯一允许持有这份知识的地方 (KD13), 所以这个问题只能在这一个文件里回答。
9. **`moss bin` 的参数与 stdio 约定。** 实参怎么传 (argv / env / 一个 json 文件)、结果怎么回
   (stdout 文本 / 结构化 json / 退出码语义)。回执与错误也要在这里一次定死, 否则每支产物
   自己发明一套。
10. **`/moss/runtime/bin/python` 这层桥还需不需要。** 如果采用顶层启动器 (KD14), 产物不再
    直接引用解释器, 这层中间人可能多余; 反过来, 若有人直接 exec 产物 (绕过 `moss bin`),
    它又是唯一能让产物跑起来的兜底。
11. **plate 在顶层启动器下还要不要沙箱。** KD5 的 `__all__` 边界与 `async def main()` 约定,
    原本是靠"编译 + 反射"实现的; 若改成"由 `moss bin` 在已 bootstrap 的进程里 aexec",
    边界的技术手段会变, 但**反射面与执行面同源**这条不能变。
