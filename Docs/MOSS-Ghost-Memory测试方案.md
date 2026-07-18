# MOSS Aurelius Ghost Memory 测试方案

> 目标：验证 Aurelius 的持久记忆、CommitNote 版本、异步反思、配置与受限 CTML 控制面，
> 并证明它不会把推断、失败帧或别人的记忆伪装为事实。

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

暂不验收：向量检索、git witness、按时间自动 commit、自动 branch merge、Desktop、Moshi
用户模型以及 CTML/TTS 世界执行进度。

## 2. 环境、依赖与配置准备

### 2.1 先选择测试层级

本方案有两条独立的执行路径。不要因为 Host/TUI 缺依赖而阻塞核心记忆回归，也不要把
pytest 通过误认为可以真实对话。

| 层级 | 覆盖内容 | 是否需要 Zenoh/Host | 入口 |
|---|---|---:|---|
| L0：核心记忆 | Moment、commit、note、反思、配置、分支的无网络回归 | 否 | pytest、acceptance script |
| L1：Ghost 发现 | workspace 是否能发现 Aurelius 注册 | 是，`moss-run-ghost` 导入 Host/Matrix | `moss-run-ghost` |
| L2：真实对话 | TUI、模型配置、CTML、重启后的端到端记忆 | 是，且需要模型凭据 | `moss-run-ghost aurelius` |

你遇到的 `ModuleNotFoundError: No module named 'zenoh'` 属于 L1/L2 的环境前置失败；它发生在
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
print("PASS: host/ghost runtime dependencies are available")
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

### 2.4 L2 模型凭据与反思模型

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

### 2.6 隔离测试数据

不要删除现有用户记忆。先停止 Aurelius，再备份：

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
  tests/ghoshell_moss/default/core/memento
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

相邻基线回归：

```bash
.venv/bin/pytest -q \
  src/ghoshell_moss/ghosts/atom \
  src/ghoshell_moss/ghosts/mock \
  src/ghoshell_moss/ghosts/aurelius \
  tests/ghoshell_moss/default/core/memento
```

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
对话本身不应等待反思完成。`memory_show` 中的原文不应变化。

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

每次测试后还原 `/tmp/memory.yml.before-aurelius-test`，再重启实例。

## 9. 启动故障排查

先判断错误发生在哪一层，避免把 Python 环境问题误判为记忆实现问题。

| 现象 | 原因 | 处理方式 | 可继续的测试 |
|---|---|---|---|
| `No module named 'zenoh'` 或 `Depend zenoh` | 未安装 `host`/`matrix` extra；`moss-run-ghost` 导入 Host 时即失败 | `uv sync --extra host --extra ghost`，再运行第 2.2 节 import preflight | L0 可继续；L1/L2 不可继续 |
| `No module named 'pydantic_ai'` | 未安装 `ghost` extra | `uv sync --extra ghost`；若要 TUI 同时安装 host | 无法运行 Aurelius 测试 |
| `ANTHROPIC_MODEL`、API key 或 base URL 未配置 | 已到 L2，但模型无法构建/请求 | 填写 `.moss/.env`；或暂不运行 L2 | L0/L1 可继续 |
| 反思模型失败 | `small_fast_model` 未解析、无凭据或网络失败 | 先设 `reflection_enabled: false` 验证主路；随后修复模型配置再测追赶 | 写入/commit/重启可继续 |
| `CellRegistry` import error | 根 `moss` CLI 的 Cell 重构不一致 | 作为独立问题记录；不要改 memory.yml | pytest/acceptance 可继续；按 traceback 判断 runner 是否受影响 |
| TUI 已启动但 Ghost 未列出 `aurelius` | workspace 注册文件或导入错误 | 检查 `.moss/src/MOSS/ghosts/aurelius.py` 和 `ghoshell_moss.ghosts.aurelius` import | L0 可继续 |

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
