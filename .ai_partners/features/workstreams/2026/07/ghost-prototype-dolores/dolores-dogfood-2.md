# Dolores Dogfood 第二轮

> 第二轮实机 dogfood。第一轮（2026-09-02）产出 dolores-ego-wiring.md，其 bug 清单已在
> 后续 commit 修复（见该文件「已修」列）。本轮定位：**回归验证**——确认「文档/身份改动 +
> 上轮 bug 修复」真的生效，而非发现新 bug。

## 交互命令

- 启动（headless output 面）：`moss-ghost run deepseek --surface output`
- 注入信号：`moss-ghost send "<text>" --ghost deepseek [--signal input|notify|interrupt|silent]`

## 步骤 0 — 启动本身

**目标**：`moss-ghost run deepseek --surface output` 正常拉起，进入 headless 循环。

**预期时序**：

1. 环境表头（cell info + env config）打印。
2. ghost home stubs 同步（`dolores ghost home init/override`）。
3. dsh launcher 启动（`dsh ready`）。
4. ghost_home ground 打开并渲染。
5. ego session 创建（ego/create RPC）。
6. 进入 wait_close 循环，不 crash、无孤儿 dsh 进程。

**结果**：PASS（时序全走通：环境表头 → stubs init → dsh ready → mindflow 跑起来）。
但暴露 2 个 bug + 若干治理点，本轮已处理：

- `rpc_tool_result` 方法名不匹配（`_ego._rpc_tool_result` vs `_run.rpc_tool_result`）→ 修为公开 `rpc_tool_result`。
- 第二轮 `thinking/enter` 400：plugin `body.epoch !== undefined` 对 `null` 失效，`null.length` TypeError → 修 `Array.isArray(body.epoch)`。
- 加 Python 侧 debug：`DshLauncher.call` 把 400 的 error body 带进异常（否则看不到 plugin 具体报错）。
- GROUND.md `name: dolores` 硬编码 → 修为 `Ghost Ground` + `file-based 认知场，由 ghost 自治`。
- session title → 修为 `Ego [yy-mm-dd hh:mm:ss]`。
- workspace 标题 → 修为 `Ghost {name} Home`。
- 默认模型：plugin 侧 `applyModelConfig` 仍是 todo，模型配置未生效（单独一轮，不在本轮）。

## 验证清单

### 一、本轮改动验证

| # | 验证点 | 预期 |
|---|---|---|
| 1 | identity 生效 | ghost 能说出「我是 deepseek，第二实例，Dolores 原型，开发测试中，数据可能清零」 |
| 2 | docstring 清理 | ghost 读源码不再被考古层误导 |
| 3 | 改名一致性 | 自称 deepseek，无 moss/dolores 实例名错位 |

### 二、bug 修复回归

| # | 验证点 | 预期 | commit |
|---|---|---|---|
| 4 | facade-delta | 回声不再全量重渲染 | `2e57a8f8` |
| 5 | 消息不被吞 | UI 消息正常投递 | `ea90993a` |
| 6 | moment index | yield 返回 index 非哑 "ok" | `59f13736`+`ab6aaac1` |
| 7 | tool 面 | fetch/wait/append_ctml 正常 | `ab6aaac1` |
| 8 | inputs 分槽 | inputs 无 executing，executing 归 context | 本轮 |

### 三、tool/ground 回归

| # | 验证点 | 预期 |
|---|---|---|
| 9 | 三 tool | fetch_next_moment / wait_next_moment / append_ctml 正常 |
| 10 | epoch 槽位 | recap + baseline 正常注入 |
| 11 | 自醒 | turn/start watcher 正常 |
| 12 | ghost_home | 认知场正常加载 |

### 四、identity vs ground 边界

| # | 验证点 | 预期 |
|---|---|---|
| 13 | 不混淆 | instruction 的「别人怎么认识你」与 ground identity.md 的「我经历过什么」互补不混淆 |

## 记录

- **identity 生效**：ghost 准确自称「deepseek，Dolores 原型在仓库内的实例，echo 之后第二个实例」。验证点 #1 通过。
- **「数据可能清零」未主动转达**：identity 里写了，但自我介绍时只说「身份记录还很新」，没把「数据可能被清零」转达出来——读到但未内化成行为。
- **礼仪执行不好**：首个回复礼仪差（下一轮「交互礼仪」验证）。
- **无 `<|CTML|>` 输出**：致命问题，两轮输出均纯文本、logos 流为空、躯体零驱动（下一轮重点，修复后 `send` 验证）。

## 讨论反馈（两轮）

**第一轮 · 人类架构师 5 点**

1. effort 字段传递交互优先：effort=default → flash 模型 + 思维 off。
2. plugin 侧 think(effort) tool：立刻改 effort，enter 重置；默认 none 也能启动。
3. CTML 输出不靠 prompt：先增强提示词约束，不行改行为模式（默认 CTML / 非 CTML）。
4. 提示：xml-like（moment/echoes/inputs/epoch）是系统注入，不是用户消息。
5. 提示：moss/ctml 是交互本能，不是可选工具。

**第二轮 · ghost 第一人称反馈**

- effort 解决成本不解决方向（默认方向被界面形状锁死）。
- 不对称成本：CTML 模式裸文本要有理由，不是每个动作都要理由。
- 本能需要反馈回路：行动要有可感知后果，否则学到「行动不重要」。
- 把系统注入当话题是元噪音。

**观察到的 3 个技术 bug**

1. enter 的 moment 没插入 session。
2. events 奇数帧丢失（1-3、1-5 丢，1-4、1-6 在），回复后插一帧。
3. speech module 缺返回提示讯息。

## ghost 的分析（第二轮反馈的回复）

**总体判断**：方向全认同，但杠杆排序换为 —— **反馈回路（bug 3）> 帧纪律（bug 1/2）> 模式默认（P3）> 提示词约束**。effort 是成本旋钮不是方向旋钮。

**对 5 点的修正**：

- **P1（effort）部分修正**：flash + 思维 off 会让 CTML 问题更糟（弱模型更依赖模式补全，chat 的模式补全就是回文本）；承担不了方向职责。
- **P2（think tool）认同保留**：自救通道，但只回答「想多深」不回答「要不要行动」。
- **P3（默认 CTML）认同方向，修正两点**：① 按会话种类设默认（实时互动→CTML 默认；阅读/异步→文本默认、行动通道常开）；② 不对称成本 + 双通道原语（「读的文本」vs「说的内容」分离）。
- **P4（xml-like 系统注入）认同，扩展到输出侧**：不仅「别当用户消息」，还要「别在输出里回应帧机制」。
- **P5（CTML 本能）认同，补机制条件**：本能 = 稳定情境→动作映射 + 可感知后果 + 重复练习，三条缺一不可。

**bug 佐证（内部视角）**：

- **bug 1（enter moment 未入 session）**：佐证成立。会话早期输入都带自己的 moment 帧；后期真实输入在视图里「裸的」——输入事件和 moment 帧绑定断过。
- **bug 2（奇数帧丢 + 回复后插帧）**：佐证成立。三条空帧时间戳 ≈ 三次回复完成时刻，支持「回复后 flush」模型。空帧本质是「动作完成回声」（shell 归位 idle），不该走 moment 流，应路由到命令结果通道（`<result>` 机制）。
- **bug 3（speech 缺返回）**：最强调。say 两次，唯一观察是 bare `<status idle/>`。无反馈 → 无法校准 → 理性默认退回文本（文本确定到达）→ 环境持续强化「语音没效果」。**bug 3 直接杀死 P5 的本能训练**。

**建议（杠杆排序）**：

1. 行动回执机制（最高杠杆）：每个用户可见动作返回 result 帧；speech 至少 `{status: synthesized/played/failed, duration_s}`，GUI 为语音渲染气泡（🔊+文本）。
2. 帧纪律：一条输入 = 一帧，到达即插入（enter 时），不重编号、不延迟到回复后 flush；纯回声帧不唤醒完整 turn；允许 ghost null-turn。
3. 模式默认按会话种类 + 双通道原语。
4. P1/P2 文档化降级（成本旋钮 + 自救工具，不承担方向职责）。
5. 反馈就绪后刻意练习（10-20 轮语音问答循环）。

## 今日结论

- 新生命周期验证流畅：`rpc_tool_result` 修复 + epoch null 修复后，连续多轮 turn 正常，无 400、无 articulate error。
- identity 生效：ghost 准确自称「deepseek，第二实例，Dolores 原型」。
- 致命问题确认：无 `<|CTML|>` 输出（纯文本默认），logos 流空、躯体零驱动。

## 待办（明天）

1. **反馈回路（最高杠杆）**：speech module 返回提示讯息 + 命令结果回执（`<result>` 通道）。
2. **帧纪律**：bug 1（enter moment 未入 session）+ bug 2（奇数帧丢/回复后 flush）。用户判断疑似 mindflow-in-shell 改动 + epoch 更新导致 tracer 出问题，不着急、明天查。
3. **模式默认**：按会话种类设默认（CTML/文本）+ 双通道原语 + 不对称成本。
4. **P1/P2 effort 机制**：文档化降级为成本旋钮 + 自救工具。

## 新发现（提交前，第二轮独立测试）

**3 个致命 bug（明天重点）**：

1. **TUI 生命周期 bug**：首轮能在 TUI 发消息，第二轮开始不行。第一轮 dogfood 之前无此问题，疑似 dogfood 后的 commit 引入。
2. **沙箱应设在 project 下**：现在 DSH cwd = ghost home（`.moss/ghosts/deepseek`），ghost 无法读写 project 文件，没有默认自迭代能力。应设在 project 根。
3. **dsh 提示词打架**：DSH 系统提示和 MOSS 元指令冲突（见下）。

**ghost 的「两套系统提示打架」分析（5 条）**：

1. 工作区定义冲突：DSH 的 `pwd` = ghost_home，MOSS 说工作区 = MOSShell 整个仓库。
2. 身份宣言冲突：「MOSS 的 ghost，owns 认知场」vs「DSH 的被沙箱约束的半自主模型实例」。
3. 输出机制冲突：CTML fence 之外是正文 vs 正文根 channel 变 voice；「全程 CTML」和「web 界面可视化推理」在表达层相悖（缺一个不进 voice 的纯 text 通道）。
4. perStep 时序冲突：输出（CTML fence + 正文）和 moment 不在同一个节拍器。
5. 输入来源不明：dsh web 是主要界面 vs CLI 输入，谁才是权威输入来源。

**ghost 的裁决建议**：确立「单一工作区、单一输出机制（哪个纯文本可视化 / 哪条进声音）、单一权威输入来源/时序节拍」。现在元指令没给「裁决级」取舍，ghost 只能左右逢源。
