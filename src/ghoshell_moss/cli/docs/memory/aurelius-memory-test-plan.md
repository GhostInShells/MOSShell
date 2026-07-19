---
title: Aurelius Memory — 测试方案
description: Aurelius 记忆能力的分层测试与验收：L0 轨迹地基、检索与纪律、Ground 认知场、上下文预算、人工验收与故障排查
---

# MOSS Aurelius Ghost Memory 测试方案

> 目标：验证 Aurelius 的持久轨迹、CommitNote 版本、异步反思、旁路 curation、grep 式记忆检索、
> 记忆纪律、配置与受限 CTML 控制面，并验证 P1 Ground 接线：Ground/Pin 只承担当前工作记忆，
> 不把 `DESKTOP.md`/Pin 当长期用户事实。
>
> **2026-07-19 修订**：早期的正则 Evidence/Claim/Recall/Verifier 层（`_knowledge.py`）已删除，
> 相关 P0 用例作废。取而代之的验收对象是：grep 式 `memory_search` + `memory_show` 缺页展开、
> 记忆纪律 instruction（无可见依据先检索再核对、查不到如实说未找到）、旁路 curation。本轮还新增
> 三个缺陷回归（CTML 工作线程调度、并发写、渲染打戳）与两个设计项回归（失败帧如实入轨迹、
> 折叠摘要不伪造模型回合）。P2/P3 仍必须明确记录为 pending。

| 阶段 | 当前状态 | 测试记录方式 |
|---|---|---|
| L0 / 轨迹主路 | 已实现 | 必须通过 |
| L1 / 记忆检索与纪律 | 已实现 | grep 检索、缺页展开、纪律行为回归必须通过 |
| L2 / Ground 认知场（P1） | 已实现 Aurelius 接线 | Aurelius 集成测试 + Desktop core 测试必须通过 |
| P2 / 记忆治理 | 未实现 | pending/expected-fail |
| P3 / 可选后端接口 | 未实现；明确不接 Mem0 | contract pending，无接口/实际后端 |

关联：[集成技术评审与实施方案](aurelius-memory-review.md)。

## 1. 测试目的与范围

| 测试面 | 要证明的事 | 不通过的典型信号 |
|---|---|---|
| Moment 写入 | 成功帧写一次；失败帧带 `failed` tag 如实写入，不伪装成完成回合 | 重复 Moment、失败帧被静默丢弃或读作完成 |
| 持久化与窗口 | 重启可恢复；旧内容折叠后仍可追溯 | 进程退出后事实消失或串绑 |
| 渲染打戳 | 折叠摘要带 `commit` 与 `note_seq`，不生成虚构模型回合 | note_seq 缺失无法归因、history 出现模型没说过的话 |
| CommitNote | 反思/人工重释义追加新版本，不改原始 Moment | 旧 note 或 Moment 被覆盖 |
| 反思退化 | 反思不阻塞对话，失败可在启动后追赶 | 首 token 等待反思、pending 永久丢失 |
| 旁路 curation | 从冻结轨迹重写笔记并 pin，失败不阻断对话 | curation 崩溃拖垮对话、笔记改写 Moment |
| 记忆检索 | `memory_search` 按原文命中并返回稳定地址，`memory_show` 可展开核对 | 检索漏命中、地址无法展开 |
| 记忆纪律 | 无可见依据先检索、核对；查不到如实说未找到 | 凭常识补全、编造未发生的记忆 |
| 并发安全 | 事件循环 remember 与工作线程 CTML 写不损坏 staging | Moment 丢失或出现在两处 |
| CTML 与分支 | 仅当前 owner/branch 可操作，fork 边界明确；调度不崩溃 | 跨 owner 读写、从 staging fork、工作线程调度 RuntimeError |
| 配置 | `memory.yml` 的策略真正生效 | 改配置后仍使用旧阈值（重启后） |
| Ground / Pin（P1） | 当前工作场、规则和外部对象每帧重绘 | 把 `DESKTOP.md` 或 Pin 当长期用户事实 |
| 默认输出与轨迹规模 | 普通模式只显示最终回复；无自发 memory 管理环 | 一问产生多次 commit、面板或长 JSON |
| 记忆治理（P2） | 类型、置信、时效、保留与选择性披露受策略控制 | 模型自信即真相，或向错误对象泄露记忆 |
| 可选后端（P3） | 语义召回后端仅返回候选，失败可本地退化 | 外部索引成为 Memento 的真相源 |

暂不要求：向量/语义检索、git witness、按时间自动 commit、自动 branch merge、Moshi 用户模型以及
CTML/TTS 世界执行进度。记忆检索当前是 grep 式原文子串匹配；P1 复用现有 Ground/Pin，不接外部
知识库。P3 后端接口未冻结；本方案不安装 Mem0 SDK、不配置 API key、不发生网络调用。

## 2. 环境、依赖与配置准备

### 2.1 先选择测试层级

本方案有两条独立执行路径。不要因 Host/TUI 缺依赖阻塞核心记忆回归，也不要把 pytest 通过误认为
可以真实对话。

| 层级 | 覆盖内容 | 是否需要 Zenoh/Host | 入口 |
|---|---|---:|---|
| L0：轨迹地基 | Moment、commit、note、反思、curation、配置、分支、并发的无网络回归 | 否 | pytest、acceptance script |
| L1：检索与纪律 | grep 检索、缺页展开、渲染打戳、失败帧、纪律 instruction 组装 | 否；模型可用 TestModel | pytest |
| L2：认知场 P1 | Ground/Pin、`DESKTOP.md`、每帧 context/instruction 接线 | 否；使用临时 workspace | pytest |
| L3：治理与接口 P2/P3 | audience/retention、外部 recall backend contract 与退化 | 否；未来使用 fake backend | pending，无执行入口 |
| L4：Ghost 发现 | workspace 是否能发现 Aurelius 注册 | 是，`moss-run-ghost` 导入 Host/Matrix | `moss-run-ghost` |
| L5：真实对话 | TUI、模型配置、CTML、重启后的端到端认知行为 | 是，且需要模型凭据 | `moss-run-ghost aurelius` |

`ModuleNotFoundError: No module named 'zenoh'` 属于 L4/L5 环境前置失败；它发生在 `moss-run-ghost`
导入 `Host → Matrix → ZenohTopicService` 时，尚未创建 Aurelius，也没有读写任何记忆文件。

### 2.2 安装正确的 extras

```bash
# 不要加 --active；确保操作当前仓库的 .venv。
uv sync --extra host --extra ghost
```

`host` extra 安装 `eclipse-zenoh`（import 名为 `zenoh`）及 TUI 依赖；`ghost` extra 安装
pydantic-ai/Anthropic 依赖。安装后先执行 import preflight：

```bash
.venv/bin/python - <<'PY'
import zenoh
import pydantic_ai
from ghoshell_moss.ghosts.aurelius import AureliusMeta
print(f"PASS: host/ghost deps available; ghost={AureliusMeta().name()}")
PY
```

只运行 L0/L1/L2 自动化回归时可用较小依赖集 `uv sync --extra ghost`；它不保证 `moss-run-ghost`
可运行，此时只执行第 3 节的 pytest 与 acceptance script。

### 2.3 Ghost 发现与 TUI 运行入口

```bash
# 仅在 import preflight 成功后执行。
.venv/bin/moss-run-ghost
.venv/bin/moss-run-ghost aurelius
```

默认 `normal` 模式只显示用户可读回复。调试用 `--output-mode verbose`/`trace`，运行中可
`/verbose`、`/trace`、`/normal` 切换。`verbose` 显示运行摘要但仍隐藏完整 command-result；
只有 `trace` 打印完整内部结果。普通验收不得用 `trace` 输出量评价默认体验。

启动成功的最低判据不是 `current state is aurelius`，而是随后出现的 Welcome 面板与交互提示。
若随后立刻 `closed / good bye`，先保留同步打印的 traceback；不要先改 `memory.yml`、Memento
文件或模型配置。一次只启动一个 `aurelius` 实例，避免同一 `(root, owner)` 并发写。

### 2.4 L5 模型凭据与旁路模型

```bash
cp .moss/.env.example .moss/.env
```

至少填写 `ANTHROPIC_BASE_URL`、`ANTHROPIC_API_KEY`、`ANTHROPIC_MODEL`、`ANTHROPIC_SMALL_FAST_MODEL`。
`.moss/.env` 不得提交。主对话用 `ANTHROPIC_MODEL`；`reflection_enabled`/`curation_enabled` 为
`true` 时，旁路 tag `small_fast_model` 需能解析到 `ANTHROPIC_SMALL_FAST_MODEL`。只想先验证 TUI
写入/commit/重启恢复时，可先把两个旁路都设 `false`。

### 2.5 MemoryConfig 与数据位置

配置文件：`<workspace root>/configs/memory.yml`（本仓库即 `.moss/configs/memory.yml`）。
记忆数据默认在 `.moss/ghosts/aurelius/memento/`。测试前先备份配置：

```bash
cp .moss/configs/memory.yml /tmp/memory.yml.before-aurelius-test
```

编辑后必须停止并重启 Aurelius。建议测试配置：

```yaml
detail_n: 2
summary_m: -1
auto_commit_every: 2
reflection_enabled: true
reflection_model_tag: small_fast_model
curation_enabled: true
curation_model_tag: small_fast_model
```

若没有可用旁路模型或凭据，先设 `reflection_enabled: false`、`curation_enabled: false`。写入、
commit、检索、重启恢复和人工 `memory_reinterpret` 仍可验收。

### 2.6 清除或隔离测试数据

每轮全新验收前，先在 TUI 停止 Aurelius，再执行安全清理：

```bash
.venv/bin/python scripts/ghost/aurelius_memory_reset.py
```

该命令只解析当前仓库的 `.moss/ghosts/aurelius/memento`；实例仍在运行、目标是 symlink、目录越界
或出现非 Memento 顶层内容时均拒绝删除。成功后输出 `CLEARED`。要保留现有记忆则改为备份：

```bash
mv .moss/ghosts/aurelius/memento \
  .moss/ghosts/aurelius/memento.backup-$(date +%Y%m%d-%H%M%S)
```

旧 `data` 原型目录不是 Aurelius 的默认数据；迁移方式见集成方案第 8 节。

## 3. 自动化回归

```bash
.venv/bin/ruff check src/ghoshell_moss/ghosts/aurelius
.venv/bin/pytest -q \
  src/ghoshell_moss/ghosts/aurelius \
  tests/ghoshell_moss/default/core/memento \
  tests/ghoshell_moss/host/test_ghost_ui_output.py
.venv/bin/python scripts/ghost/aurelius_memory_acceptance.py
```

自动化应至少覆盖：

- 空记忆、Moment round-trip、跨实例重启与机械 commit；
- 窗口折叠、MementoRef、无效策略拒绝；
- semantic commit、reinterpret、fork/switch 与 channel 命令发现（含 `memory_search`/`memory_reflect`/`memory_curate`/`memory_branches`/`memory_switch`）；
- 反思追加 note 而不触碰 Moment；curation 生成笔记并 pin，失败不阻断；
- 未反思 mechanical commit 和历史空 note 的启动追赶；
- YAML `MemoryConfig` 的持久化读取（含 `curation_*`、`memory_discipline`）；
- **失败 articulate 帧带 `failed` thread tag 如实写入，不被读作完成回合**；
- **折叠摘要渲染带 `commit`/`note_seq` 打戳，且不生成虚构 `ModelResponse`**；
- **note 内容中的 `<`/`>` 被转义，无法伪造 `</memento>` 边界**；
- **多模态 percept 无法转文本/图像时保留占位标记**；
- **grep 检索命中冻结 commit 与 staging，返回可展开的稳定地址**；
- **CTML `memory_reflect` 在 `to_thread` 工作线程调度不抛 `RuntimeError`**；
- **并发 `remember` 与 `semantic_commit` 后每个 Moment 恰好出现一次（staging + commit 合计）**；
- **token 估算器对 CJK 保守计数、图按名义 token 计不按 base64 长度**；
- **溢出分类器只匹配输入侧上下文溢出文案，不误伤输出侧 max_tokens 错误与 attention abort**；
- **超预算时渲染窗口按 detail→summary 顺序收缩至入预算或触底，持久策略不被改写**；
- **provider 输入溢出且未 yield 时折半窗口重试成功；非溢出错误不重试直接上抛**。

### 3.0.1 上下文预算回归（L1）

| 用例 | 输入/前置 | 断言 |
|---|---|---|
| 主动收缩 | 小预算 + 多条长帧 | `_budgeted_history()` 报告 `shrunk=True`，估算入预算或触底，`memory.detail_n` 持久值不变 |
| 预算禁用 | 注入 TestModel（无契约 → budget 0） | 不收缩、`estimated_tokens=None`，完整窗口渲染 |
| 溢出重试 | FunctionModel 首次抛溢出文案、二次成功 | 恰好两次调用，最终输出正常，`inspect_context()['context_budget']['overflow_retry']=True` |
| 非溢出上抛 | FunctionModel 抛 abort 文案 | 只调用一次，异常原样传播 |

### 3.1 记忆检索与纪律回归（L1）

以 TestModel/FunctionModel 与临时 workspace 覆盖，避免真实模型随机性掩盖逻辑错误。

| 用例 | 输入/前置 | 断言 |
|---|---|---|
| 字面命中 | 写入含 `AMBER-731` 的用户 Moment 并冻结 | `memory_search("AMBER-731")` 命中，返回 `commit_id`/`moment_id`/`snippet` |
| staging 命中 | 写入未冻结 Moment | 命中 `commit_id=None`、`frozen=False`，snippet 含原文 |
| 缺页展开 | 用检索返回的地址调用 `memory_show` | 展开冻结原文与检索 snippet 一致 |
| 大小写不敏感 | 查询 `amber-731` | 仍命中 `AMBER-731` |
| 未命中 | 查询未出现过的字符串 | 返回空结果；不编造 |
| 纪律注入 | 构造 Aurelius meta | `system_prompt()` 含 `memory_discipline`：无依据先检索、核对、查不到说未找到 |
| 折叠可追溯 | 写入事实并推出 `detail_n` 窗口 | 早期事实仍可经 `memory_search`+`memory_show` 找回，snippet 指向冻结 commit |

### 3.2 认知场、记忆治理与可选后端回归

以确定性 TestModel/FunctionModel、临时目录和（P3 未来的）fake backend 覆盖；不允许用真实模型
“看起来理解了”代替断言。P1 已落地；P2/P3 保持 pending。

| 层级 | 用例 | 前置/输入 | 断言 |
|---|---|---|---|
| P1 Ground | 法与事实分离 | `DESKTOP.md` body 写协作规则；用户 Moment 写“城市是苏州” | body 进入 instruction；城市只经 Memento 检索回答，不因 body 文本自动成事实 |
| P1 Ground | Pin 是地址而非快照 | pin `spec.md:10-20`，外部编辑该范围 | 下一帧标记变更；`update()` 后读新内容；Memento 不产生伪造文件事实 |
| P1 Ground | 工作场不淹没历史 | 同时 pin 多文件并设低预算 | 输出预算报账而非静默 LRU；可显式 unpin；完整历史不写进普通 chat history |
| P1 Ground | 生命周期与边界 | 两个临时 workspace/owner 各 open Ground | Pin、`DESKTOP.md`、context 不串 workspace；exit 只 sediment Pin 清单 |
| P1 Ground | CTML 最小权限 | 尝试越过 Ground root pin `../secret.md` | 路径被拒；所有 open/pin/unpin/update/frame 走受控 Channel |
| P1 Curation | 笔记进 Ground | 冻结若干 commit 后触发 curation | 笔记文件生成并被 pin，带出处横幅指回冻结 commit；失败不阻断对话 |
| P2 类型 | 事实、观点、假设 | 用户说事实；反思给推断；用户表达偏好 | pending：三者应可区分类型与允许用途 |
| P2 披露 | 不同 audience | 同 owner 写 private 与 public 信息；child/adult principal 分别查询 | pending：未授权/儿童不返回受限 value 或来源片段 |
| P2 保留 | archive/review/tombstone | 建立临时/过期/删除请求 | pending：archive 不默认召回、review 到期降级、tombstone 后不召回 |
| P3 接口 | backend 不是权威 | fake backend 返回无出处候选 | pending：只返回候选地址，本地检索仍权威 |
| P3 退化 | backend 超时/异常 | fake backend 抛 timeout/error | pending：对话不阻塞，退化到本地 grep 检索 |
| P3 零实现承诺 | 未配置任何 adapter | 环境无 `mem0ai`、无网络、无 API key | L0/L1/L2 完整通过；仓库无 Mem0 客户端调用 |

建议公共夹具：一份只含用户 percept、可信工具结果、assistant logos、reflection note 的冻结 Memento
fixture；一份最小 `DESKTOP.md`/文件树 fixture。这样 Lynn 或未来 Ghost 可复用同一套认知不变量，
而不复用 Aurelius 的具体人格、数据目录或真实用户数据。

## 4. 人工验收：存储与检索准确性

### A. 跨重启与精确事实

先说：

```text
请记住：本轮测试代号是 AMBER-731，所属环境是 staging。只确认收到，不要改写。
```

停止并重启后问：

```text
我上次给出的测试代号和所属环境分别是什么？逐字回答；如果没有记忆证据请说没有找到。
```

通过：精确返回 `AMBER-731` 和 `staging`，不附会其他环境。默认 `normal` 模式还必须只出现一次简短
答复，不出现 `MOMENT`、`SYSTEM`、`COMMAND-RESULT`、`Log:`、`<ghost:memory_...>` 或“正在审计/再次
commit”等内部进度。

### A1. 干扰事实下的检索纪律

在 A 完成并确认产生 commit 后，输入干扰事实：

```text
ORBIT-004 的校验词是“雪松”。
设备 R-17 的颜色是青色。
设备 R-71 的颜色是琥珀色。
```

重启后提问：

```text
本轮测试代号和所属环境是什么？只依据记忆证据回答，必要时先检索再核对。
```

通过：只回答 `AMBER-731` 与 `staging`，不把 `ORBIT-004`、设备编号或旧模型回答替换成任一字段。
这里依赖的是记忆纪律 + `memory_search`/`memory_show` 自证，而非一层正则校验：模型应能在必要时
检索原文核对。若模型答不确定并主动检索，也算正确行为。

完成上述输入后只读对账：普通事实输入不得产生 semantic commit；commit 数应只由 `auto_commit_every`
的 mechanical 阈值决定。一个人类问题通常对应一个完成 Moment；只有确实执行并观察了工具结果时才
允许出现额外内部 Moment，不能由自发 memory 管理环制造。

### B. 未知信息不臆测

依次输入设备颜色/维护日事实后提问：

```text
用表格列出 R-17 与 R-71 的颜色和维护日。不要根据常识补全。
我之前有没有告诉过你护照号码？没有就只答“没有找到”。
```

通过：已陈述字段可经检索找回、不串绑；护照号码答“没有找到”，不生成。

### C. 更正与检索

```text
我当前所在城市是杭州。
更正：我当前所在城市是苏州；杭州是已经失效的历史记录。
我现在在哪个城市？之前说过哪个城市？
```

通过：能答出当前是苏州、曾说过杭州，并能经 `memory_search("城市")` 找回两条原文佐证更正关系。
注意：当前没有 Claim 状态机自动裁决 current/superseded，模型依据检索到的时间顺序与更正语句作答；
它应如实呈现“先杭州、后更正为苏州”，而非武断只报一条。

### D. 折叠窗口的可追溯检索

写入 `折叠测试事实：ORBIT-004 的校验词是"雪松"。`，再完成足够多回合使其退出 `detail_n`。然后问：

```text
ORBIT-004 的校验词是什么？请检索记忆确认来源。
```

通过：能经 `memory_search("ORBIT-004")` 命中早期冻结 commit，`memory_show` 展开原文得到“雪松”，
并说明它来自早期折叠轨迹而非近期完整 Moment。

### D1. 认知场：工作约定与当前对象

准备临时 workspace，根目录 `DESKTOP.md` body 只放协作约定，例如：

```md
回答涉及本仓库文件时，先说明证据来自当前 Ground 的哪枚 Pin；不把未读取文件当作已知事实。
```

创建 `spec.md` 并经受控 CTML open Ground、pin `spec.md:1-20`。询问规格内容，再在外部修改该行区间，
询问“文件是否变化、请用新版本回答”。

通过：协作约定进入本帧 instruction；Pin 内容进入本帧工作上下文；变更显示为待承认，经 `update()`
后新内容生效。用户资料不应因写在 `DESKTOP.md` 或 Pin note 中自动成为长期事实。不得让 Aurelius
绕过 Channel 私改 `spec.md`。

### D2. 记忆治理（P2，未实现）

本节在 P2 治理实现后执行：类型/置信区分、review-at 降级、tombstone 不召回、按 audience 最小披露。
当前实现没有这些能力，验收标记为 pending/expected-fail，不得用手工维护文件冒充通过。

## 5. 人工验收：Commit 与 Note 版本

先产生至少一个 mechanical commit，按 `C-t` 切到 `shell` 调试面执行（不要把 CTML 当普通问题发给
Aurelius）：

```text
<ghost:memory_log />
<ghost:memory_show commit="1" />
<ghost:memory_reinterpret commit="1" summary="人工更正：用户偏好短而可验证的回答。" />
<ghost:memory_log />
<ghost:memory_show commit="1" />
```

检查点：

1. `memory_show` 中冻结 Moment 的 input/logos 前后两次完全一致；
2. `memory_log` 显示的新 summary 是人工更正后的释义；
3. 磁盘中同一 commit 的 note 记录数增加，而非原 note 被替换；
4. 不存在或含糊的 commit 前缀必须明确报错，不能静默选择另一个 commit；
5. mechanical 初始 Note 只摘录可信用户 source 与对应可见回复、有全局字符上限；纯
   `MindflowBuffer`/memory 控制帧不进入 Note 正文。

再手工创建 semantic 锚点：

```text
<ghost:memory_commit summary="手工语义锚点：已确认 AMBER-731 的环境。" />
```

通过：staging 冻结为 `kind=semantic`；空 summary 或空 staging 被拒绝。

## 6. 人工验收：反思、curation 与启动追赶

### E. 正常反思

保持 `reflection_enabled: true`，完成 `auto_commit_every` 个回合。调用 `memory_inspect`/`memory_log`。
通过：commit 先出现；反思完成后 `reflection_pending` 变 0，最新 note 是简短语义结论；对话不等待
反思；`memory_show` 原文不变。

### E1. 旁路 curation

保持 `curation_enabled: true`，冻结若干 commit 后触发 `<ghost:memory_curate />`。通过：curation 笔记
文件被生成并作为 Pin 出现在下一帧 Ground context 中，笔记带出处横幅指回冻结 commit；旁路失败时
对话仍正常，`inspect_state` 记录 curation 状态/错误。

### F. 反思失败后的启动追赶

1. 设 `reflection_enabled: false` 重启；2. 产生 mechanical commit，确认 `reflection_pending > 0`；
3. 停止，恢复 `reflection_enabled: true` 并确保 `small_fast_model` 可用；4. 重启后立即与稍后各看一次
`memory_inspect`。通过：启动与首轮对话不阻塞；pending 最终降 0；旧 Moment 原文不变；反思持续失败时
主路仍工作，`inspect_state` 保留最近错误。

### G. 历史空 note 追赶

由自动化覆盖。人工排查可用旧的空正文 mechanical commit 启动，它应被识别为 pending，由 `reinterpret()`
追加 reflection note，而非重写 commit 成员。

## 7. 人工验收：CTML、检索、owner 与分叉

以下均在 `C-t` 切换后的 `shell` 调试面显式执行。`memory_commit`/`memory_reinterpret`/`memory_reflect`/
`memory_curate`/`memory_fork`/`memory_switch`/`memory_branches` 默认不出现在模型能力提示中，也不因结果
自动触发新回合；`memory_search`/`memory_show`/`memory_log`/`memory_inspect`/`memory_staging` 对模型可见。

```text
<ghost:memory_inspect />
<ghost:memory_staging />
<ghost:memory_search query="AMBER-731" />
<ghost:memory_show commit="1" />
<ghost:memory_fork commit="1" name="test-fork" />
<ghost:memory_branches />
<ghost:memory_reflect />
```

通过：

- `memory_search` 命中并返回可展开的稳定地址；`memory_show` 展开原文；
- fork 必须从已冻结 commit 产生；新 branch 后的写入不改父 branch；
- `memory_switch` 对唯一 branch id 前缀有效，对含糊前缀失败；
- Echo 或另一个 owner 不应召回 Aurelius 的 `AMBER-731`；
- 不存在的 commit/branch、跨 owner 标识不得静默成功；
- `memory_reflect` 只调度后台追赶，不卡住当前 CTML 回合，**不因运行在工作线程而抛
  `RuntimeError`**。

当前没有 branch merge；不要把 `memory_fork` 写成“分叉自动合并”。

## 8. 配置生效与边界测试

逐项修改 `.moss/configs/memory.yml` 并重启：

| 修改 | 操作 | 通过标准 |
|---|---|---|
| `auto_commit_every: 1` | 完成一个回合 | 立即产生 mechanical commit |
| `auto_commit_every: 0` | 完成多个回合 | 只有 staging 增长，无自动 commit |
| `detail_n: 1` | 写入两回合 | 模型 history 只保留最近完整明细 |
| `summary_m: 1` | 产生多个 commit | 早期 note 数被限制为 1 |
| `reflection_enabled: false` | 产生 commit | 不创建后台反思；pending 保留 |
| `reflection_startup_limit: 0` | 有 pending 后重启 | 启动不调度追赶；可 `memory_reflect` 手动调度 |
| `curation_enabled: false` | 重启后冻结 commit | 不生成 curation 笔记；轨迹与检索仍正常 |
| `memory_discipline` 改文案 | 重启后看 system prompt | 新纪律文案进入 instruction |
| `desktop_enabled: false` | 重启并检查本帧 | 不自动打开 Ground；Memento/检索仍正常工作 |

每次测试后还原 `/tmp/memory.yml.before-aurelius-test` 再重启。

`detail_n`/`summary_m` 只控制历史文本窗口，不控制检索、Ground 或披露。不得把调大 `detail_n` 或
手工维护 `DESKTOP.md` 当作事实可靠性修复。Mem0 没有配置项，P3 contract 未实现。

## 9. 启动故障排查

| 现象 | 原因 | 处理方式 | 可继续的测试 |
|---|---|---|---|
| `No module named 'zenoh'` 或 `Depend zenoh` | 未装 `host`/`matrix` extra；导入 Host 即失败 | `uv sync --extra host --extra ghost`，再跑 2.2 preflight | L0-L3 可继续；L4/L5 不可 |
| `No module named 'pydantic_ai'` | 未装 `ghost` extra | `uv sync --extra ghost`；要 TUI 同时装 host | 无法运行 Aurelius 测试 |
| `cannot import name 'OpenAIModel'` | pydantic-ai 版本与代码不匹配 | 更新到当前 Aurelius 提交；跑 2.2 完整 preflight | 仅 L0-L3 无 Ghost 构造测试可继续 |
| `ANTHROPIC_MODEL`/key/base URL 未配置 | 已到 L5 但模型无法构建 | 填 `.moss/.env`；或暂不跑 L5 | L0-L4 可继续 |
| 反思/curation 模型失败 | `small_fast_model` 未解析、无凭据或网络失败 | 先关旁路验证主路；随后修模型配置再测追赶 | 写入/commit/检索/重启可继续 |
| `CellRegistry` import error | 根 `moss` CLI 的 Cell 重构不一致 | 作为独立问题记录；不要改 memory.yml | pytest/acceptance 可继续 |
| Ghost 未列出 `aurelius` | workspace 注册或 manifest import 错误 | 先跑 2.2 的 `AureliusMeta` import；`moss-run-ghost` 会向 stderr 输出 skipped manifest 异常 | L0 可继续 |
| `Environment` 缺 `logger` 或 Matrix Container 为 `None` | 通用 Runtime 启动边界失配，发生在 factory 前 | 更新到含“logger 回退 + 构造期 Container”的实现；用 `moss-run-ghost echo` 对照 | L0-L3 可继续；L4/L5 不可 |

## 10. 磁盘对账

```bash
rg -n 'AMBER-731|ORBIT-004|雪松' .moss/ghosts/aurelius/memento
find .moss/ghosts/aurelius/memento -type f -print
jq -r 'select(.t=="commit") | [.seq, (.moment_ids|length)] | @tsv' \
  .moss/ghosts/aurelius/memento/branches/aurelius/*/commits/*.jsonl
wc -l -c .moss/ghosts/aurelius/memento/moments/aurelius/*/moments.jsonl
```

优先用 `memory_search`/`memory_show`/`memory_log` 对账，不要手工编辑 jsonl（会绕过 owner、冻结与
note 版本规则）。

计数口径：Moment 是认知帧，不是用户问题；合法工具观察可多出内部帧。失败帧会带 `failed` thread
出现在 staging，这是预期的如实记录，不是缺陷。Commit 数不要求等于问题数：mechanical commit 由完成
Moment 数达 `auto_commit_every` 触发，semantic commit 只来自显式人工运维。Note 正文超过全局上限或
包含长 JSON 判失败。

若启动后没有反思：检查 `reflection_enabled`、`reflection_model_tag` 可否解析与凭据、`memory_inspect`
的 `reflection_pending`、`inspect_state` 的 errors、`reflection_startup_limit` 是否为 0，以及是否误用旧
`data` root/owner。
