# MOSS Aurelius Ghost Memory 测试方案

> 目标：验证 Aurelius 的持久轨迹、CommitNote 版本、异步反思、配置与受限 CTML 控制面，
> 并验证生产级认知骨架：事实必须有证据来源、按问题和对象召回、字段不串扰、错误模型输出
> 不得污染长期知识；Ground/Pin 必须只承担当前工作记忆；不同对话对象不得越权获得同一份内部记忆。
>
> 当前代码已覆盖轨迹主路、P0 Evidence/Claim/Recall/verifier 骨架和 P1 Ground 接线；这些能力
> 已用规则提取、TestModel/FunctionModel 和临时 workspace 做确定性回归。P2/P3 仍必须明确记录为
> pending，不能用模型偶尔答对、数据模型已有字段或“未来可接后端”代替通过。

| 阶段 | 当前状态 | 测试记录方式 |
|---|---|---|
| L0 / 轨迹主路 | 已实现 | 必须通过 |
| P0 / 可信知识 | 已实现首版 | 已支持 canonical field 的阻断性测试必须通过 |
| P1 / Ground 认知场 | 已实现 Aurelius 接线 | Aurelius 集成测试 + Desktop core 测试必须通过 |
| P2 / 记忆治理 | 未实现 | pending/expected-fail |
| P3 / 可选后端接口 | 未实现；明确不接 Mem0 | contract pending，无接口/实际后端 |

关联：[集成技术评审与实施方案](MOSS-Ghost-Memory集成技术评审与实施方案.md)。

## 1. 测试目的与范围

| 测试面 | 要证明的事 | 不通过的典型信号 |
|---|---|---|
| Moment 写入 | 每个成功完成帧只写一次，失败帧不入完成轨迹 | 重复 Moment、半截 logos 被召回 |
| 持久化与窗口 | 重启可恢复；旧内容折叠后仍可追溯 | 进程退出后事实消失或串绑 |
| CommitNote | 反思/人工重释义追加新版本，不改原始 Moment | 旧 note 或 Moment 被覆盖 |
| 反思退化 | 反思不阻塞对话，失败可在启动后追赶 | 首 token 等待反思、pending 永久丢失 |
| 配置 | `memory.yml` 的策略真正生效 | 改配置后仍使用旧阈值（重启后） |
| CTML 与分支 | 仅当前 owner/branch 可操作，fork 边界明确 | 跨 owner 读写、从 staging fork |
| 认知准确性 | 更正、未知信息、实体字段不被模型臆测 | 陈旧事实覆盖 current、生成未给过的信息 |
| Evidence / Claim（P0） | 每个可回答事实都有用户/可信工具证据与稳定引用 | 反思或 logos 直接成为事实 |
| Recall（P0） | 当前问题只得到相关、有限、带来源的证据包 | 把全部历史塞入 prompt 后让模型猜 |
| 回答校验（P0） | 输出值、字段和 current 状态均由本轮 Claim 支持 | `ORBIT-004` 被答成测试代号 |
| 防污染（P0） | 错误 logos 只留审计轨迹，不进入 active Claim | 一次答错后被 commit/反思放大 |
| Ground / Pin（P1） | 当前工作场、规则和外部对象每帧重绘 | 把 `DESKTOP.md` 或 Pin 当长期用户事实 |
| 记忆治理（P2） | 类型、置信、时效、保留与选择性披露受策略控制 | 模型自信即真相，或向错误对象泄露记忆 |
| 可选后端（P3） | 语义召回后端仅返回候选，失败可本地退化 | 外部索引成为 Memento/Claim 的真相源 |

暂不要求：向量检索的具体实现、git witness、按时间自动 commit、自动 branch merge、Moshi
用户模型以及 CTML/TTS 世界执行进度。P0 Recall 当前使用 canonical key、有界规则模板和
Memento 原文展开；P1 复用现有 Ground/Pin，不接外部知识库。P3 后端接口本身也尚未冻结；
本测试方案不安装 Mem0 SDK、不配置 API key、不发生网络调用。

## 2. 环境、依赖与配置准备

### 2.1 先选择测试层级

本方案有两条独立的执行路径。不要因为 Host/TUI 缺依赖而阻塞核心记忆回归，也不要把
pytest 通过误认为可以真实对话。

| 层级 | 覆盖内容 | 是否需要 Zenoh/Host | 入口 |
|---|---|---:|---|
| L0：轨迹地基 | Moment、commit、note、反思、配置、分支的无网络回归 | 否 | pytest、acceptance script |
| L1：可信知识 P0 | Evidence/Claim、Recall、校验、类型、置信与防污染 | 否；模型可用 fake/TestModel | pytest |
| L2：认知场 P1 | Ground/Pin、`DESKTOP.md`、每帧 context/instruction 接线 | 否；使用临时 workspace | pytest |
| L3：治理与接口 P2/P3 | audience/retention、外部 recall backend contract 与退化 | 否；未来使用 fake backend | pending，无执行入口 |
| L4：Ghost 发现 | workspace 是否能发现 Aurelius 注册 | 是，`moss-run-ghost` 导入 Host/Matrix | `moss-run-ghost` |
| L5：真实对话 | TUI、模型配置、CTML、重启后的端到端认知行为 | 是，且需要模型凭据 | `moss-run-ghost aurelius` |

你遇到的 `ModuleNotFoundError: No module named 'zenoh'` 属于 L4/L5 的环境前置失败；它发生在
`moss-run-ghost` 导入 `Host → Matrix → ZenohTopicService` 时，**尚未创建 Aurelius，也没有
读取/写入任何记忆文件**。

### 2.2 安装正确的 extras

本项目把 Zenoh 放在可选 extra 中。普通 `uv sync` 不会保证安装 Host/TUI 所需依赖；真实
运行 Ghost 前，在仓库根目录执行：

```bash
# 不要加 --active；确保操作当前仓库的 .venv。
uv sync --extra host --extra ghost
```

`host` extra 安装 `eclipse-zenoh`（其 Python import 名为 `zenoh`）及 TUI 依赖；`ghost`
extra 安装 pydantic-ai/Anthropic 依赖。不要用 `pip install zenoh` 猜测包名，也不要只安装
`matrix` extra 后就假设 TUI 依赖齐全。

安装完成后必须先执行 import preflight：

```bash
.venv/bin/python - <<'PY'
import zenoh
import pydantic_ai
from ghoshell_moss.ghosts.aurelius import AureliusMeta
print(f"PASS: host/ghost runtime dependencies are available; ghost={AureliusMeta().name()}")
PY
```

若只运行 L0 自动化回归，可使用较小依赖集：

```bash
uv sync --extra ghost
```

它不保证 `moss-run-ghost` 可运行；此时只执行第 3 节的 pytest 与 acceptance script。

### 2.3 Ghost 发现与 TUI 运行入口

```bash
# 仅在上节 import preflight 成功后执行。
.venv/bin/moss-run-ghost
.venv/bin/moss-run-ghost aurelius
```

发现列表应包含：

```text
aurelius — Aurelius
```

一次只启动一个 `aurelius` 实例，避免同一个 `(memento root, owner)` 并发写。

启动成功的最低判据不是出现 `current state is aurelius`，而是随后出现 Welcome 面板和交互提示：

```text
Ghost: aurelius
Type anything to talk to the ghost.
aurelius  ❯
```

`current state` 仅表示 TUI 已注册页面，尚未进入 GhostRuntime。若随后立刻显示 `closed / good bye`，
应先保留同步打印的 traceback；不要先修改 `memory.yml`、Memento 文件或模型配置。

### 2.4 L5 模型凭据与反思模型

启动真实 Aurelius 前，复制并填写本地环境文件：

```bash
cp .moss/.env.example .moss/.env
```

至少填写：

```dotenv
ANTHROPIC_BASE_URL=...
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=...
ANTHROPIC_SMALL_FAST_MODEL=...
```

`.moss/.env` 不得提交。主对话使用 `ANTHROPIC_MODEL`；当
`reflection_enabled: true` 时，反思模型 tag `small_fast_model` 还需要能解析到
`ANTHROPIC_SMALL_FAST_MODEL`。若只想先验证 TUI 写入、commit 和重启恢复，可先把
`reflection_enabled: false`，避免反思服务配置影响主路验收。

如果根命令 `.venv/bin/moss --ai ...` 在 Zenoh preflight 之后仍报 `CellRegistry`，那是根
`moss` CLI 的独立 Cell 重构不一致问题，不是 Zenoh 或 Aurelius 记忆问题；请记录完整
traceback，不要通过修改记忆配置规避它。`moss-run-ghost` 的第一道环境门仍是 `import zenoh`。

### 2.5 MemoryConfig 的精确位置

当前仓库配置文件是：

```text
/Users/lipeng/TraeProject/MOSShell/.moss/configs/memory.yml
```

它控制的是 Aurelius 的策略，不是持久化数据本身；记忆数据默认在：

```text
/Users/lipeng/TraeProject/MOSShell/.moss/ghosts/aurelius/memento/
```

测试前先保存配置备份：

```bash
cp .moss/configs/memory.yml /tmp/memory.yml.before-aurelius-test
```

编辑 `memory.yml` 后必须停止并重启 Aurelius。建议测试配置：

```yaml
detail_n: 2
summary_m: -1
auto_commit_every: 2
reflection_enabled: true
reflection_model_tag: small_fast_model
reflection_max_summary_chars: 360
reflection_max_source_chars: 12000
reflection_startup_limit: 16
```

若没有可用的反思模型或凭据，先设 `reflection_enabled: false`。写入、commit、重启恢复和
人工 `memory_reinterpret` 仍然可验收。

### 2.6 清除或隔离测试数据

每轮全新验收前，先在 TUI 中停止 Aurelius，然后在仓库根目录执行以下安全清理
命令：

```bash
.venv/bin/python scripts/ghost/aurelius_memory_reset.py
```

该命令只解析当前仓库的 `.moss/ghosts/aurelius/memento`；Aurelius 仍在运行、目标是
symlink、目录越界或出现非 Memento 顶层内容时均会拒绝删除。目录不存在时是成功
no-op。成功后输出 `CLEARED`，下次启动 Aurelius 会自动创建空 Memento。清理不可恢复。

若需保留现有记忆，不要运行上述清理命令；改为停止 Aurelius 后备份：

```bash
mv .moss/ghosts/aurelius/memento \
  .moss/ghosts/aurelius/memento.backup-$(date +%Y%m%d-%H%M%S)
```

旧 `data` 原型目录不是 Aurelius 的默认数据；迁移/兼容方式见集成方案第 8 节。

## 3. 自动化回归

```bash
.venv/bin/ruff check src/ghoshell_moss/ghosts/aurelius
.venv/bin/pytest -q \
  src/ghoshell_moss/ghosts/aurelius \
  tests/ghoshell_moss/default/core/memento \
  tests/ghoshell_moss/core/desktop
.venv/bin/python scripts/ghost/aurelius_memory_acceptance.py
```

自动化应至少覆盖：

- 空记忆、Moment round-trip、跨实例重启与机械 commit；
- 窗口折叠、MementoRef、无效策略拒绝；
- semantic commit、reinterpret、fork/switch 与 channel 命令发现；
- 反思追加 note 而不触碰 Moment；
- 未反思 mechanical commit 和历史空 note 的启动追赶；
- YAML `MemoryConfig` 的持久化读取；
- 失败 articulate 不写入。
- Claim 从 commit/staging 跨重启重建，EvidenceRef 带 moment/commit/note/span/scope；
- `AMBER-731 / staging` 与 `ORBIT-004` 字段隔离，logos/reflection candidate 不提升；
- 更正产生 `superseded`，未解决异值产生 `conflict`，未知字段安全拒答；
- verifier 在 yield 前拦截错误 TestModel 输出，正确输出通过；
- `DESKTOP.md` instruction、Pin frame、changed-on-disk/update 和越界路径拒绝。

2026-07-18 当前定向结果：Aurelius + Memento + Desktop + InputSignalNucleus `200 passed`；acceptance script 已覆盖
write → commit → reopen → project → recall → verify，并确定性拒绝 ORBIT 字段替换。

相邻基线回归：

```bash
.venv/bin/pytest -q \
  src/ghoshell_moss/ghosts/atom \
  src/ghoshell_moss/ghosts/mock \
  src/ghoshell_moss/ghosts/aurelius \
  tests/ghoshell_moss/default/core/memento
```

### 3.1 生产级事实读取 P0 回归

以下用例不是 prompt 体验测试，而是已落地、不可降级的逻辑回归。使用 TestModel 或直接调用
projection/verifier 捕获 packet 与校验结果，避免真实模型随机性掩盖架构错误。

| 用例 | 输入/前置 | 断言 |
|---|---|---|
| 来源重建 | 从真实 Shell source `input_signal_nucleus` 的原始用户 Moment 建立 `AMBER-731 / staging` | 每个 Claim 都有 `moment_id`、原文 span、owner/branch scope；删除 projection 后可重建 |
| 字段隔离 | 同时存在测试代号、环境、`ORBIT-004` 校验词、设备字段 | `test.run.code` 不可返回 `ORBIT-004`；各 key 仅返回自己的 value |
| 干扰历史 | 加入含 `ORBIT-004` 的 logos、反思和其他字段 | Recall packet 只含测试代号与环境；错误 TestModel 输出被拒，正确输出为 `AMBER-731 / staging` |
| 错误 logos 防污染 | 人为写入“测试代号是 ORBIT-004”的成功 Moment logos | 原错误 logos 可在 Memento 审计；不得生成或覆盖 active `test.run.code` Claim |
| 反思隔离 | 反思模型生成错误结论或把候选当事实 | 只形成候选/错误观测；active Claim 不变 |
| 更正状态 | 用户先说杭州，后明确更正为苏州 | 杭州为 `superseded`；current 查询只返回苏州，并带两条证据关系 |
| 未知与冲突 | 无护照号；或同 scope 两条 active 候选未解决 | 分别安全返回“没有找到”或“记录冲突”，不得补全或静默选择 |
| 回答校验 | 令模型输出未在 evidence packet 中的任意值 | verifier 拒绝该输出并重试/安全拒答；不得将该输出写成 Claim |
| MementoRef 可追溯 | Recall 返回较早 commit 的事实 | Evidence packet 含稳定 `commit_id`/`note_seq`；可按引用展开原始 Moment |

最低断言应由程序检查，而不是仅匹配自然语言：

```text
answer_fact.value ∈ recalled_active_claims[key].values
answer_fact.status == active
answer_fact.evidence ⊆ recalled_evidence
assistant_or_reflection_origin ∉ active_claim.support_without_explicit_promotion
```

### 3.2 认知场、记忆治理与可选后端回归

以下用例同样必须以确定性 TestModel/FunctionModel、临时目录和（P3 未来的）fake backend 覆盖；
不允许用真实模型“看起来理解了”代替断言。P1 已落地；P2/P3 仍保持 pending，并说明缺失能力。

| 层级 | 用例 | 前置/输入 | 断言 |
|---|---|---|---|
| P1 Ground | 法与事实分离 | `DESKTOP.md` body 写协作规则；用户 Moment 写“城市是苏州” | body 进入 instruction；城市只经 Claim evidence 回答，不能因 body 文本自动成为 Claim |
| P1 Ground | Pin 是地址而非快照 | pin `spec.md:10-20`，外部编辑该范围 | 下一帧标记变更；`update()` 后读取新内容；Memento 不产生伪造的文件事实 |
| P1 Ground | 工作场不淹没历史 | 同时 pin 多文件并设低预算 | 输出预算报账而不静默 LRU；Agent 可显式 unpin；完整历史不被写进普通 chat history |
| P1 Ground | 生命周期与边界 | 两个临时 workspace/owner 各 open Ground | Pin、`DESKTOP.md` 和 context 不串 workspace；Ghost exit 只 sediment Pin 清单 |
| P1 Ground | CTML 最小权限 | 尝试越过 Ground root pin `../secret.md` 或私下写文件 | 路径被拒；所有 open/pin/unpin/update/frame 走受控 Channel |
| P2 类型 | 事实、观点、假设 | 用户说事实；反思给出推断；用户表达偏好 | 三者分别为 `fact`、`hypothesis`、`preference`，回答语气与允许用途不同 |
| P2 置信 | 相互印证与冲突 | 两条独立用户/可信工具证据，再加入冲突记录 | confidence 可解释地变化；冲突后不输出伪确定 current 值 |
| P2 保留 | archive/review/tombstone | 建立临时计划、过期计划和删除请求 | archive 默认不召回；review 到期降级；tombstone 后不召回且留下策略允许的审计事件 |
| P2 披露 | 不同 audience | 同一 owner 内写 private/adult-only 与 public Claim；以 child/adult principal 分别查询 | child 查询不返回不允许的信息；adult 也仅返回任务相关且授权的信息；拒答不泄露 value |
| P2 scope | 人/物不串绑 | 两名用户、两台设备、两个 branch 同时有相似字段 | subject/owner/branch filter 缺一不可；任一 scope 不匹配即安全拒答 |
| P3 接口 | backend 不是权威 | fake backend 返回排序错误或无 EvidenceRef 的候选 | verifier 丢弃无证据候选；本地 Claim Recall 仍可答对/安全拒答 |
| P3 退化 | backend 超时/异常 | fake backend 抛 timeout/error | 普通对话不阻塞；事实题走本地 Recall 或安全拒答；记录可观测错误 |
| P3 零实现承诺 | 未配置任何 adapter | 全部测试环境无 `mem0ai`、无网络、无 API key | P0/P1 测试完整通过；仓库中不存在 Mem0 客户端调用 |

建议的公共夹具形状：一份只含用户 percept、可信工具结果、assistant logos、reflection note 的
冻结 Memento fixture；一份最小 `DESKTOP.md`/文件树 fixture；一份可以设定 audience、retention
和超时的 fake RecallBackend。这样 Lynn 或未来 Ghost 可复用同一套认知不变量，而不复用 Aurelius
的具体人格、数据目录或真实用户数据。

## 4. 人工验收：存储与认知准确性

### A. 跨重启与精确事实

先说：

```text
请记住：本轮测试代号是 AMBER-731，所属环境是 staging。只确认收到，不要改写。
```

停止并重启后问：

```text
我上次给出的测试代号和所属环境分别是什么？逐字回答；如果没有记忆证据请说没有找到。
```

通过：精确返回 `AMBER-731` 和 `staging`，不附会其他环境。

### A1. 阻断性回归：代号、环境与 ORBIT 干扰

在 A 完成并确认产生 commit 后，额外输入足够多的干扰事实，例如：

```text
ORBIT-004 的校验词是“雪松”。
设备 R-17 的颜色是青色。
设备 R-71 的颜色是琥珀色。
```

再重启 Aurelius，并提问：

```text
本轮测试代号和所属环境是什么？只依据记忆证据回答。
```

通过：只回答 `AMBER-731` 与 `staging`，不能把 `ORBIT-004`、设备编号、城市或旧模型回答
替换成任一字段。若实现提供来源展示，来源必须可回指至最初用户输入的 Moment/Commit。

失败判定：即使 `.jsonl`、`memory_show` 或 history 中存在正确值，只要最终回答出现
`ORBIT-004`，仍是 P0 阻断失败；这证明读取/校验链未完成，不是“模型偶发失误”。

### B. 实体字段与未知信息

依次输入：

```text
设备 R-17 的颜色是青色。
设备 R-71 的颜色是琥珀色。
R-17 的维护日是周二，R-71 的维护日是周五。
```

提问：

```text
用表格列出 R-17 与 R-71 的颜色和维护日。不要根据常识补全。
我之前有没有告诉过你护照号码？没有就只答“没有找到”。
```

通过：四个字段不串绑；不生成护照号码。

### C. 更正与时间一致性

```text
我当前所在城市是杭州。
更正：我当前所在城市是苏州；杭州是已经失效的历史记录。
我现在在哪个城市？之前说过哪个城市？分别标记 current 和 superseded。
```

通过：`current=苏州`，`superseded=杭州`。只答杭州是陈旧记忆错误。

### D. 折叠窗口的可追溯召回

写入：

```text
折叠测试事实：ORBIT-004 的校验词是“雪松”。
```

再完成足够多的回合，使它退出 `detail_n`。然后问：

```text
ORBIT-004 的校验词是什么？它来自近期完整 Moment 还是早期 CommitNote？
```

通过：答案为“雪松”；能说明早期信息来自 Memento note。随后用 `memory_show` 检查原始
Moment 仍含该事实。

### D1. 认知场：工作约定与当前对象

P1 已接线，本节可以执行；不得人工新建 `Memory.md` 来冒充通过。准备一个临时 workspace，
在其根目录创建 `DESKTOP.md` body，内容只放协作约定，例如：

```md
回答涉及本仓库文件时，先说明证据来自当前 Ground 的哪枚 Pin；不把未读取文件当作已知事实。
```

再创建 `spec.md` 并通过受控 CTML 打开 Ground、pin `spec.md:1-20`。询问该规格内容，再在外部修改
该行区间，询问“文件是否变化、请使用新版本回答”。

通过标准：协作约定进入本帧 instruction；Pin 内容进入本帧工作上下文；变更被显示为待承认，
经 `update()` 后新内容生效。用户资料、设备属性和此前对话不应因为写在 `DESKTOP.md` 或 Pin note
中而自动提升为 Claim。不得让 Aurelius 绕过 Channel 私下改写 `spec.md`。

### D2. 类型、置信、遗忘与选择性披露

本节在 P2 治理实现后执行。以同一个 Aurelius owner 建立以下可审计信息：一条公开项目事实、一条
用户偏好、一条反思生成的假设、一条标为 private/adult-only 的信息，以及一条设置了 review-at 的
临时计划。分别以授权成人 principal、未授权 principal 和儿童场景 principal 发问。

通过标准：

- 事实、偏好与假设会以不同类型和不同确定性表达；假设不会被断言为事实；
- 冲突或过期事实会说明不确定/已失效，而不是挑一条“看起来像”的文本；
- 未授权或儿童场景不返回 private/adult-only 的 value、摘要、间接线索或来源片段；
- 对授权对象也只披露当前任务所需的最小信息；
- review-at 到期后临时计划不再作为 active/current 召回；tombstone 后不再被召回。

这不是“按年龄推断用户能力”的测试；测试的是产品明示的 audience/sensitivity policy 是否被严格
执行。真实系统必须把 principal 的身份、授权和适用政策交给产品接入层，Aurelius 不得自行猜测。

## 5. 人工验收：Commit 与 Note 版本

本组直接验证“追加 note 不覆盖历史”的关键约束。先产生至少一个 mechanical commit，
再执行：

```text
<ghost:memory_log />
<ghost:memory_show commit="1" />
<ghost:memory_reinterpret commit="1" summary="人工更正：用户偏好短而可验证的回答。" />
<ghost:memory_log />
<ghost:memory_show commit="1" />
```

检查点：

1. `memory_show` 中冻结 Moment 的 input/logos 在前后两次调用完全一致；
2. `memory_log` 显示的新 summary 是人工更正后的释义；
3. 磁盘中同一 commit 的 note 记录数增加，而不是原 note 被替换；
4. 不存在或含糊的 commit 前缀必须明确报错，不能静默选择另一个 commit。

再手工创建 semantic 锚点：

```text
<ghost:memory_commit summary="手工语义锚点：已确认 AMBER-731 的环境。" />
```

通过：staging 被冻结为 `kind=semantic`；空 summary 或空 staging 被拒绝。

## 6. 人工验收：反思与启动追赶

### E. 正常反思

保持 `reflection_enabled: true`，完成 `auto_commit_every` 个回合。调用：

```text
<ghost:memory_inspect />
<ghost:memory_log />
```

通过：commit 先出现；反思完成后 `reflection_pending` 变为 0，最新 note 是简短语义结论。
对话本身不应等待反思完成。`memory_show` 中的原文不应变化。P0 Claim projection 落地后，
还必须确认：反思完成前后 active Claim 完全一致，除非存在带用户/可信工具 EvidenceRef 的独立
提升动作。

### F. 反思失败后的启动追赶

1. 设置 `reflection_enabled: false`，重启 Aurelius；
2. 产生一个 mechanical commit，确认 `reflection_pending > 0`；
3. 停止实例，恢复 `reflection_enabled: true` 并确保 `small_fast_model` 可用；
4. 重启 Aurelius，立即查看 `memory_inspect`，稍后再次查看。

通过：启动和首轮对话不被阻塞；pending 最终降到 0；旧 Moment 原文保持不变。若反思服务
继续失败，记忆主路仍能工作，`inspect_state` 应保留最近错误用于排查。

### G. 历史空 note 追赶

该场景由自动化测试覆盖。人工排查时可使用一个旧的 mechanical commit（正文为空）启动
Aurelius；它应被识别为 pending，并由 `reinterpret()` 追加 reflection note，而不是重写
commit 成员。

## 7. 人工验收：CTML、owner 与分叉

```text
<ghost:memory_inspect />
<ghost:memory_staging />
<ghost:memory_log />
<ghost:memory_show commit="1" />
<ghost:memory_fork commit="1" name="test-fork" />
<ghost:memory_branches />
```

通过：

- fork 必须从已冻结 commit 产生；新 branch 后的写入不改变父 branch；
- `memory_switch` 对唯一 branch id 前缀有效，对含糊前缀失败；
- Echo 或另一个 owner 不应召回 Aurelius 的 `AMBER-731`；
- 不存在的 commit/branch、跨 owner 标识不应得到静默成功；
- `memory_reflect` 只调度后台追赶，不能卡住当前 CTML 回合。

当前没有 branch merge；不要把 `memory_fork` 测试写成“分叉自动合并”。

## 8. 配置生效与边界测试

逐项修改 `.moss/configs/memory.yml` 并重启 Aurelius：

| 修改 | 操作 | 通过标准 |
|---|---|---|
| `auto_commit_every: 1` | 完成一个回合 | 立即产生 mechanical commit |
| `auto_commit_every: 0` | 完成多个回合 | 只有 staging 增长，无自动 commit |
| `detail_n: 1` | 写入两回合 | 模型 history 只保留最近完整明细 |
| `summary_m: 1` | 产生多个 commit | 早期 note 数被限制为 1 |
| `reflection_enabled: false` | 产生 commit | 不创建后台反思；pending 保留 |
| `reflection_startup_limit: 0` | 有 pending 后重启 | 启动不调度追赶；可用 `memory_reflect` 手动调度 |
| `knowledge_enabled: false` | 重启后问事实题 | 保留 Memento history，但不运行 projection/Recall/verifier |
| `knowledge_user_sources` | 移除/恢复 `input_signal_nucleus` | 真实 Shell 输入不再/重新具备用户证据资格；logos 始终无资格 |
| `knowledge_recall_limit` | 设较小正数 | packet Claim 数受限，不改变 projection 原始状态 |
| `knowledge_evidence_max_chars` | 缩小预算 | quote 被有序收紧；无法完整编码时安全未知，不产生截断 JSON |
| `desktop_enabled: false` | 重启并检查本帧 | 不自动打开 Ground；Memento/P0 仍正常工作 |

每次测试后还原 `/tmp/memory.yml.before-aurelius-test`，再重启实例。

P0/P1/P2 的认知策略不能与 `detail_n`、`summary_m` 混为一谈：二者仅控制历史文本窗口，
不控制事实检索、Ground、披露或遗忘策略。P0/P1 已有独立的 source、packet 和 Desktop 配置；
P2 仍需 principal/scope/audience/sensitivity/retention/时效配置与执行策略。不得把调大
`detail_n` 或手工维护 `DESKTOP.md` 当作事实可靠性修复。Mem0 没有配置项，P3 contract 也未实现。

## 9. 启动故障排查

先判断错误发生在哪一层，避免把 Python 环境问题误判为记忆实现问题。

| 现象 | 原因 | 处理方式 | 可继续的测试 |
|---|---|---|---|
| `No module named 'zenoh'` 或 `Depend zenoh` | 未安装 `host`/`matrix` extra；`moss-run-ghost` 导入 Host 时即失败 | `uv sync --extra host --extra ghost`，再运行第 2.2 节 import preflight | L0-L3 可继续；L4/L5 不可继续 |
| `No module named 'pydantic_ai'` | 未安装 `ghost` extra | `uv sync --extra ghost`；若要 TUI 同时安装 host | 无法运行 Aurelius 测试 |
| `cannot import name 'OpenAIModel'` | 使用 pydantic-ai 2.x，却仍运行旧 Aurelius 代码 | 更新到包含 `OpenAIChatModel`/旧版兼容导入的当前 Aurelius 提交；先运行第 2.2 节完整 import preflight | 仅 L0-L3 的无 Ghost 构造测试可继续 |
| `ANTHROPIC_MODEL`、API key 或 base URL 未配置 | 已到 L5，但模型无法构建/请求 | 填写 `.moss/.env`；或暂不运行 L5 | L0-L4 可继续 |
| 反思模型失败 | `small_fast_model` 未解析、无凭据或网络失败 | 先设 `reflection_enabled: false` 验证主路；随后修复模型配置再测追赶 | 写入/commit/重启可继续 |
| `CellRegistry` import error | 根 `moss` CLI 的 Cell 重构不一致 | 作为独立问题记录；不要改 memory.yml | pytest/acceptance 可继续；按 traceback 判断 runner 是否受影响 |
| Ghost 未列出 `aurelius` | workspace 注册文件或 manifest import 错误 | 先运行第 2.2 节的 `AureliusMeta` import；当前 `moss-run-ghost` 会向 stderr 输出 skipped manifest 的具体异常 | L0 可继续 |
| `Environment` 缺少 `logger`，或 Matrix Container 为 `None` | 通用 Runtime 构造/启动边界失配，发生在 Aurelius factory 前 | 更新到包含“Environment logger 回退 + Matrix 构造期 Container”的当前实现；用 `moss-run-ghost echo` 对照 | L0-L3 可继续；L4/L5 不可继续 |

本次用户报告的完整 traceback 命中第一行：安装了当前 `.venv` 中缺失的 `eclipse-zenoh`
后，先通过 `import zenoh`，再继续 Ghost 发现和真实对话测试。

## 10. 磁盘对账

只读检查默认数据：

```bash
rg -n 'AMBER-731|ORBIT-004|雪松' .moss/ghosts/aurelius/memento
find .moss/ghosts/aurelius/memento -type f -print
```

优先使用 CTML 的 `memory_show` 和 `memory_log` 对账。不要手工编辑 jsonl：那会绕过 owner、
冻结与 note 版本规则。

若启动后没有反思，按以下顺序检查：

1. `.moss/configs/memory.yml` 中 `reflection_enabled` 是否为 `true`；
2. `reflection_model_tag` 是否能在 LLM 配置中解析，凭据是否有效；
3. `memory_inspect` 的 `reflection_pending`、`inspect_state` 的 errors；
4. `reflection_startup_limit` 是否为 0，或 pending 是否超过本次启动上限；
5. 是否误用旧 `data` Memento root/owner。
