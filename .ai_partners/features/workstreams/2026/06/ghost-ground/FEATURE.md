---
created: 2026-06-10
depends:
- momento-mori
description: 'Ground 是 Ghost 的认知场 — context_messages 上可 pin 的表面 + 场 (目录) 的开合管理. 以
  frontmatter+body 的 L0 文件 (GROUND.md) 为载体, pins 作为 frontmatter 的一部分; frontmatter
  = 机器域, body = 人/模型叙事域. 三层落地按预训练迁移量排序: CTML channel (主), bash CLI (`moss ground`),
  module_eval (小概率). CLI 层已完成: spec / init / frame / meta / observe / validate 六命令,
  contract + concrete 全部落地, 154 测试通过, 两轮 dogfood 验收. CTML channel 落点见 ground-channel
  workstream (薄 channel 设计).'
milestone: 0.1.0
priority: P0
renamed_from: Project Manager
status: completed
status_note: |
    CLI layer complete. Pending: ground-channel (CTML runtime), moss-project-ground.
title: Ghost Ground — Ghost 的认知场
updated: '2026-08-11'
---

# Ground — Ghost 的认知场

## 2026-07 收敛方向

Ground 的定位: **认知场 (context 表面 + 场开合)**, 以 `GROUND.md` (frontmatter
+ body + `## ground:pins`) 为 L0 载体. 代码层 / CLI 层 / 文档表面 / feature
目录名 全部用 ground (K49 + K53, 已落 9e5d4c05).

- **场** = 打开的目录, 挂到父 ground channel 上作 command-less virtual channel;
  `instruction` = 法链 (祖先 GROUND.md body 链 + 本地 body + @ 展开, K56),
  `context_messages` = 帧渲染 (声明区 + 结果区), `startup`/`close` = load/sediment.
- **贴纸** = pin. kwargs 信封 (`label` + `verb` + `arguments` + `description`,
  K55); 每帧重读, mtime + 内容 hash 对账 — 对账是 pin 与 @ 装载的唯一分界 (K56).
- **法** = 本地 body + `@path` 静态装载 + 向上法链 (祖先 GROUND.md body,
  $HOME 边界, root-first). 协议自包含, 不读 CLAUDE.md 等外部约定 (K56).
- **路径** = 锚点语法: 裸相对路径默认 = `$GROUND` (文档锚), 显式浮动 `$CWD`,
  机器逃生口 `$HOME`; 一切路径 (pin 与 @) 按锚定界 (K58).
- **L0** = 场的 `frontmatter + body + pins`. frontmatter 是 MOSS 唯一发明域
  (`$id` + `label`); body 与 pins 永远开放集.

**契约文档**: `src/ghoshell_moss/ground/SPECIFICATION.md` (K49/K50 已落).

**三层落地** (按预训练迁移量排序): CTML channel (主) > bash CLI (`moss ground`,
极小面) > module_eval (小概率, 不做).

**自举验收**: 下一个模型实例能通过 ground 场 (open 本 feature 目录 + pin
FEATURE.md 与 2026-07 两份 discuss + SPECIFICATION.md) 重建对本设计的认知,
而不靠人类手工指路.

**详细讨论轨迹** (优先读):

- `src/ghoshell_moss/contracts/.discuss/2026-07-12_desktop_channel_landing_and_fractal_grounds.md`
  — channel 落点, 分形 L 体系, features 迭代史打磨定律, 双寄存器判据
- `src/ghoshell_moss/contracts/.discuss/2026-07-11_cognitive_field_from_desktop_to_model_built_grounds.md`
  — 拆捆绑, 认知场概念, 收敛形状

## Motivation

Ghost 需要在文件系统上有 "自己的工作面":

- 模型 context 没有可控的注意力结构 — 系统推什么就看什么, compact 之后关键
  信息丢失, 无法自主构建工作记忆
- 行业方案 (CLAUDE.md, MEMORY.md, rules, skills) 是 **静态** 的 — 没人定义
  "进入一个目录在运行时意味着什么", 也没人做 **场所局部** (place-local) 的
  认知加 enter/exit 生命周期
- MOSS 自己的 features / .discuss / .design 三级验证体系有相同的结构但没有
  运行时载体 — 每次会话都要靠模型记得去 `features list` 才存在

Ground 把这些统一为 **运行时的可 pin 的 context 表面 + 场的开合**. frontmatter
是唯一的机器边界; body 永远开放; 什么文件 "有认知价值" 由场里的 GROUND.md
文件自己声明. 与静态方案的本质差异: **场是活的** — 每帧重绘, 变更标记, pin
是第一人称动作而非系统自动规则.

## Design Index

- **契约文档 (SPEC)**:
  - `src/ghoshell_moss/ground/SPECIFICATION.md` — GROUND.md 格式规范,
    pin 类型契约, frame 拓扑, CLI 极小面, 语言无关性要求 (K49 后已迁)
- **2026-07 重绘讨论**:
  - `.discuss/2026-07-12_desktop_channel_landing_and_fractal_grounds.md`
  - `.discuss/2026-07-11_cognitive_field_from_desktop_to_model_built_grounds.md`
- **配对基建**:
  - `momento-mori` (Memento) — 胶囊 (promote 后的 pin) 落永久记忆; drain 联合
    设计方
  - `subprocesses` / `job_supervisor` — 从旧 Desktop 拆出的执行域 (审计线外侧).
    契约层形态是 K18 三层纪律的样板参照
  - `file-editor-contract` — 写路径 + 结构化 view; K6 撤守卫留下的空档承接方
- **channel 抽象参考**:
  - `moss codex blueprint channel_builder` — Channel as Context Component,
    instruction / context_messages / refresh_meta 生命周期
  - `moss codex blueprint states_channel` — `MutableChannelState.add_virtual_channel`
    docstring: "wrap this method into a command" — Ground 的原生解
  - `src/ghoshell_moss/channels/module_eval_channel.py` — abc → concrete →
    channel 三层装配的标杆参照
- **CTML 视角**: `src/ghoshell_moss/core/ctml/prompts/v1_0_0.zh.md`
- **历史设计** (供反向查询, 部分被 07 月重绘超越):
  - `.design/2026-06-28_desktop_in_4d_cross_section.md` — 4 剪影拓扑 + 12+1 原语
    L2 稿 (Stage 1 实现的依据; 隐喻仍成立, 12+1 捆绑不成立)
  - `.discuss/2026-06-28_desktop_l2_emergence.md` — L2 涌现方法论

## Key Decisions

保留 07-12 起的核心方向 (K14~K21) 与 07-14/15 对齐 (K35~K43) 的摘要, 加
07-19 (K44~K54) 与 07-20 (K55~K58). 更早的 K1~K13 (原捆绑设计) 大部分已随
Stage 1 代码删除 (d75a0112), 需要考古走 `git log --all -- FEATURE.md`.
K23/K24/K25 收在 "已知未决".

### 07-12 核心方向 (K14~K21)

- **K14. 落 channel** — 父 ground channel + 每场 command-less virtual channel;
  context_messages 是"桌子" (帧尺度 O(1) 覆写表面, compact 免疫).
- **K15. 分形 L 体系** — L-1 (领域数据 CLAUDE.md/MOSS.md) / L0 (frontmatter +
  body + pins) / L1 (L0 模板) / L2 (L1 库). frontmatter/body 边界即机器/模型
  边界. features 体系是实物证明.
- **K16. 双寄存器词汇** — 理论词汇住文档, 表面词汇骑预训练先验.
  **本轮 (K48) 修正**: "channel 名 desktop 隐喻仍成立" 被覆盖 — 全库统一 ground.
- **K17. 桌子即工作记忆** — 不与世界自动同步. 变更 hash 对账标记 stale,
  显式 update 推进认知. 原生 drain 协议独立立项 (K19).
- **K18. 三层抽象纪律** — contracts (ABC) → core (concrete) → channels
  (手写装配). 不做过早规则糖.
- **K19. 原生 drain 独立立项** — 与 Memento "产生时入记忆" 合并设计,
  非本 feature 关切.
- **K20. 目光落运行时侧影, 不自动 sediment 到 body** — auto-sediment 到治理
  文件违反 "沉淀是主动动作". 侧影载体位置见 K24 (未决).
- **K21. open/update 边界** — `Grounds.open(dir, label=None) → Ground`.
  CTML 表面在父, core 层薄转发到 Ground; 同 dir 幂等 (K28).

详细讨论见 `.discuss/2026-07-12_*` 与 `git log --all -- FEATURE.md` 中
2026-07-12 前后 commit.

### 07-14 认知场对齐 (K35~K39)

- **K35. 携带/属地合成** — 场 (pins + body) 天然属地; convention 与 body
  可携带 (L1). 分层合成: convention per-field merge, 法链式合成, pins 永
  不携带. 默认属地胜; 强制覆盖须 open 显式参数. 与 K28 幂等的冲突待
  dogfooding (K40).
- **K37. L1/L2 纪律** — 冷启动 fewshot + 可验证. 修改记录即"未被囚禁"证据.
- **K38. L2 = 追认 .ai_partners/** — 不新建 templates/ 目录. moss 自建 L2
  dogfood.
- **K39. 五问信息传递 + 三级发现链** — where/what/which/how/why 劈机器域
  /文体域. L2→L1→L0 发现链走 glob + marker.

详细行业对照与 K36 前身讨论见 `git log --all -- FEATURE.md` 中 2026-07-14
commit.

### 07-15 marker + $id + pin 类型扩展 (K22 / K36 / K41~K43)

- **K22 (07-15 重决)** — L0 marker: DESKTOP.md → GROUND.md. GROUND = 电学
  零电位参考点 + common ground 双料先验, AI 行业未占. 本轮 K48 把 rename
  从 marker 层升级为全库层.
- **K36. pin 落盘瘦身** — 只留 argv `pin` + `label` + optional `description`.
  seen_* 观察态是运行时侧影, 不入 GROUND.md. 首次进入无上帧故不标 stale.
  本轮 SPEC 落定. (字段形状 07-20 由 K55 修订为 label/verb/arguments/
  description; seen_* 不入盘维持.)
- **K41. pin 类型扩展 (path#field, path## heading)** — 语法候选, 本轮 K44
  argv 契约 (`pin: [type, arg1, ...]`) 天然容纳同族扩展.
- **K42. $id 身份体系** — frontmatter `$id: <URI>` 字符串, MOSS 不校验, 解析
  交上层. 撤 `$template` (血统记录与身份声明混淆).
- **K43. .grands/ 设计分支** — 回退方案. 反悔判据: 多认知方法成为真实痛感.

详细讨论见 `git log --all -- FEATURE.md` 中 2026-07-15 commit.

### 07-19 本轮决策 (K44~K49)

本轮为 "抽象锁死, 待人工重命名后按 SPEC 落代码" 的收敛点. 契约文档独立到
`src/ghoshell_moss/core/desktop/SPECIFICATION.md`.

- **K44. pin 传参 = list[str] 位置参数, 命令名多态** ⚠ 07-20 被 K55 覆盖
  — 三方案对照:
  - 方案 1 (kwargs `pin_file(path=..., range=...)`) — 参数名要跨语言协议化,
    模板压力大 (可选参数空值处理麻烦).
  - 方案 2 (强类型结构 `PinFile(path=..., range=...)`) — 跨语言协议难定,
    模型预训练迁移量差.
  - 方案 3 (`pin: [verb, arg1, arg2, ...]`) — 采用. 理由: bash argv 是
    预训练最深处 (K16 双寄存器); SPEC 表达最简 (每 type 一行 argv 契约);
    落盘 shape 最简; 未知类型前向兼容天然.

- **K45. pin 类型最小集** — `file / glob / frontmatter / ls` 四个,
  pin_bash 本轮不做 (07-11 T1 不变量决定). K41 语法扩展 (path#field /
  path## heading) 在本 argv 契约下是同族扩展, 不需新机制.
  展开约束 (SPEC 强判): glob / ls 只出结构 (paths + mtime + size), **不出
  内容** — 一个 pin 展成 50k tokens 是 dogfood 死亡场景.

- **K46. label + description 手动指定** — 撤回自动派生 (讨论中曾提议 label
  从 argv 派生). 理由: pin_bash 加入时派生规则会崩 ("`bash "git log --oneline"`
  应派生什么 label"), 规则本身成为一门未申明的语言.
  - label = pin 声明第一字段, ground 内唯一, 承担 unpin 定位 + fenced block
    lang tag 双职. 字符集 `[a-zA-Z_][a-zA-Z0-9_-]{0,63}`.
  - description = 短评注 (marginalia). 合并原 "note" 概念. 长解说走 body.

- **K47. 拓扑分区 = 声明区 + 结果区** — frame 分节:
  - head: `ground: <label> @ <abspath>` + 可选 `$id`
  - body: GROUND.md body verbatim (场解说落这里)
  - 声明区: 每 pin 一行 `label:verb(args) # description`, 一屏可扫
  - 结果区: 每 pin 一个 fenced code block, label 作 lang tag

  声明区**不允许自由文本** (场解说走 body). 展示格式细节 dogfooding 中调整.
  这是 07-11 记录者 "桌子 + 贴纸" 直觉的 markdown 化.

- **K48. CLI 极小面 = spec / init / frame / observe 四命令** — 撤原
  pin / unpin / update / status / pins / instruction 六动词. 理由:
  - SPEC 就绪后直接编辑 GROUND.md 的 YAML 是最快路径 (K16 去发明纪律);
  - pin 动词的真正落点是 CTML channel 层 (K14), CLI 不必对应实现.

  CLI 只提供诊断与自解释:
  - `moss ground spec` — 打印 SPECIFICATION.md
  - `moss ground init [dir]` — 造空 GROUND.md 脚手架
  - `moss ground frame [dir]` — 渲染完整帧 (dogfood 主力)
  - `moss ground observe [dir]` — pin 观察诊断

  每次 CLI 调用一次性 `open→render→close`, 无跨调用 opened 状态 —
  session 状态归 CTML channel 层.

- **K49. 全库 desktop → ground 重命名** — K16 原判 "channel 名 desktop
  隐喻仍成立" 本轮覆盖. 混用 desktop (表面隐喻) + ground (目录实体) 两个词
  模型会崩, 全库统一:
  - `contracts/desktop.py` → `contracts/ground.py`
  - `core/desktop/` → `core/ground/` (SPECIFICATION.md 随目录搬)
  - `cli/desktop_cli.py` → `cli/ground_cli.py`
  - `moss desktop *` → `moss ground *`

  Grounds / Ground / GroundConvention 类名不变 (已是 ground 命名).
  feature 目录名 `ghost-filesystem-desktop` 保留 (git 历史语义, 不动发现链).
  重命名由人工 IDE 完成, 完成后按 SPEC 重写 concrete + CLI + 单测.

- **K30 修正 (轻方案 → pathspec)** — 原 K30 走 BUILTIN_TREE_IGNORE 常量 +
  tree_ignore_extra 加法口. 本轮 pin_glob / pin_ls 都需要 `.gitignore` 完整
  语义, 硬编码兜底集不够. 走 pathspec 依赖.

- **K33 更正 (记录失真)** — `channels/desktop_channel.py` 从未出现在任何
  提交 (`git log --all --` 空返回验证). 07-13 status_note "已写未测未装配"
  是模型对不存在文件的凭空断言. 真实状况: **CTML channel 层从未落地**.
  K14 装配等 SPEC 就绪 + K49 重命名后启动.

- **K34 完成 (旧 Stage 1 清理)** — 已由 commit d75a0112 (2026-07-14 03:32)
  删除 Stage 1 三份共 1548 行代码 (`core/desktop/desktop.py` 950 +
  `core/desktop/models.py` 65 + `tests/.../test_desktop.py` 529). 从
  "已知未决" 移出.

### 07-19 追补 (K50~K54)

K44~K49 抽象锁定后, 本轮追补五点, 覆盖包结构 / frame 渲染修订 /
新动作留位 / feature 目录改名 / 实现纪律.

- **K50. 包结构隔离 vs 物理拆库 — 两层分离** — SPEC §9 已明确语言无关性,
  ground 是一个协议, Python 侧只是 reference implementation. 协议边界
  值得在包结构上可见.
  - **层次 A (包结构隔离)**: `ghoshell_moss.ground` 独立子包, 公共 API
    在 `ground/__init__.py` 显式声明. **K49 重命名同期完成** (顺路比事后
    折腾便宜). 结构提议:
    ```
    src/ghoshell_moss/ground/
      __init__.py              # 公共 API
      contracts.py             # ABC 层 (原 contracts/desktop.py)
      SPECIFICATION.md         # SPEC 契约 (随目录搬)
      core/                    # concrete 层 (原 core/desktop/)
      pin_types/               # 四 verb 实现 (K45)
      channel.py               # CTML channel 装配 (K14, 未做, 位置留位)
    src/ghoshell_moss/cli/ground_cli.py     # CLI 仍在 cli/ 树
    ```
  - **层次 B (物理拆库为独立 PyPI `ghoshell-ground`)**: **现在过早**.
    判据 = SPEC 稳定 3~6 个月 dogfooding + 3+ 非 MOSS 消费者信号.
    落到 milestone 0.2.0 之后评估.
  - **依赖体检结论 (支撑 B 可行性)**: pin verbs / GROUND.md 解析 / frame
    渲染 / Grounds ABC — 全部 stdlib + pathspec + pyyaml, MOSS-free.
    CTML channel 装配 (channel.py) 强耦合 MOSS Channel/Scope/GhostWorkspace,
    未来拆库时这层留在 MOSS 侧. 这个天然分层意味着未来拆库的物理边界已经
    在包结构里画好.

- **K51. Frame 拓扑修订 — 加 `## ground:content` heading + args 引号纪律**
  (⚠ 07-20: args 引号纪律被 K55 kwargs 展示覆盖; `## ground:content`
  heading 维持) — SPEC §6 原稿结果区直接列 fenced blocks, 无 heading 标签.
  本轮修订:
  - 结果区加 `## ground:content` heading. 与声明区 `## ground:pins`
    对称, 语义分组明确, body 里的自由 heading 用 `ground:` 前缀命名空间
    保护 (body 写 `## Motivation` 不与分隔符打架).
  - 声明区 args 里的字符串**始终引号** (`file("FEATURE.md")` 而非
    `file(FEATURE.md)`). 代价 4 字符/pin, 收益 = 解析规则一致, 模型
    不需推断 shell 引号规则. SPEC §6 原稿的 "quoted where they contain
    whitespace or shell-significant characters" 被覆盖.
  - **同构原则**: 渲染格式与 GROUND.md 存储格式在骨架上同构 —
    存储 `## ground:pins` 是 YAML 段, 渲染 `## ground:pins` 是每行一 pin
    的声明块 (`label:verb(args) # description`), 语义都是 "pin 清单",
    模型习得成本最低.

- **K52. `switch_to` (move = pop + enter) 语义留位** — SPEC §7.1 只有
  open/close, 缺 move. 本轮不实现, 但在 SPEC 加 §7.1.1 reserved 段
  锁定语义:
  - API: `Grounds.switch_to(dir, label=None) -> Ground` = `close(active) +
    open(dir)` 的原子化.
  - **语义分野**: exit 触发 drain (K19), **move 不触发 drain**. 理由:
    move 是 "移步换景", Ghost 并未真正 "离开" 意识流, 工作记忆不该被归档.
  - **与 Memento 的第二个耦合点**: 一系列 move 组成 Ghost 的 trajectory
    breadcrumb, 本身是需要 commit 的 moment. 与 K19 promote-drain 是
    ground↔memento 两条独立耦合线.
  - **与 Mindflow 的耦合**: move = attention spotlight 的连续平移
    (vs exit+enter 的离散槽切换). 如果 mindflow 未来把 attention 建模为
    可连续变换的量, move 是它的自然原语.
  - **实现窗口**: K19 drain 落定 + Memento §14 (checkout(commit_id,
    moment_id)) 完成后启动. 本 feature 不做, 只在 SPEC 留 stub 锁形状.

- **K53. Feature 目录改名 (K49 "保留 git 历史" 被覆盖)** — 原 K49 判定
  "feature 目录名 ghost-filesystem-desktop 保留 (git 历史语义, 不动
  发现链)". 本轮覆盖. 理由:
  - 全库 desktop→ground 后, 保留 desktop 后缀的 feature 目录成为
    "未申明的语言" — 未来模型看到 K49 说保留 git 历史, 但代码 / CLI /
    文档全部用 ground, 语义会崩.
  - "git 历史语义" 用 `git mv` 保留 (rename 检测由 git 自动完成),
    不需要靠目录名承担.
  - "不动发现链" 由 features 体系的 `list/status` 命令承担, 也不需要
    靠目录名.
  - **候选名待议**: `ghost-ground` / `ghost-cognitive-ground` /
    `cognitive-ground` / `ground` (纯名). 选择原则 = 与 momento-mori
    同族命名 (Ghost 认知基建), 与全库 ground 术语统一, 短.

- **K54. 零魔法值纪律 (SPEC 落代码时的强判)** — SPEC 里有两处需要
  rationale 或转为命名常量:
  - **SPEC §4 `label` 长度 63** — rationale 未有. 提为 `PIN_LABEL_MAX_LEN
    = 63` 常量 + 注释说明选择 (可能是 "1 byte length prefix + 64 chars"
    历史约定, 或者只是 "看着够用" — 后者要在注释里承认).
  - **SPEC §5.4 `ls` 默认 `depth = 2`** — 提为 `LS_DEFAULT_DEPTH = 2`
    常量, 注释 "一屏可扫的默认值".
  - 其余边界令牌 (`GROUND.md` / `## ground:pins` / `## ground:content` /
    四 verb 名字 / CLI 四动词) 都有 K44~K49 撑腰, 是 "命名的边界令牌",
    不是魔法值; 但仍作模块级常量, 便于协议演进时集中修改.
  - 实现时全库 grep `\d+` 与硬编码字符串, 逐个判 rationale.

### 07-20 本轮决策 (K55~K58)

本轮重开 K44 (pin 传参), 并顺势解决三个关联问题: instruction 机制归属 /
嵌套场语义 / 路径锚点. 出发点: **展示语法与配置语法的高一致性** — GROUND.md
是模型看着 frame 手写的, 双向推导必须机械, 翻译出错率是主导成本. 教训记录:
"没有锁定啥, 方案打磨不够好就无法推进到下一步" (人工原话) — K44~K54 的
"抽象锁定" 宣言被证明过早, 本轮推翻了其中两条.

- **K55. pin 信封 = kwargs + 信封/载荷分离 (K44 覆盖)** — 三方案对照:
  - argv (K44 原判 `pin: [verb, arg1, ...]`) — 落盘最简但演进脆 (verb 加
    可选参 = 老读者错义), 参数无语义 (`"80-140"` 是什么要背顺序), YAML
    原生类型冲突 (`depth: 2` 解析成 int, 打破 "全字符串" 契约).
  - kwargs 内联 (`pin: {verb: file, path: ...}`) — 信封仍多态 (key 集随
    verb 变), tagged union 只是被压扁没消失.
  - **信封/载荷分离 (采用)**:
    ```yaml
    - label: hot
      verb: file
      arguments: {path: "src/hot.py", range: "80-140"}
      description: "hot spot"
    ```
    信封 `{label, verb, arguments, description}` 永远单态; 多态关进
    `arguments` 一个字段. 理由: (1) 信封固定 → 不认识 verb 的工具也能
    解析/改写/round-trip; (2) 未知 verb 时 arguments 天然不透明, 前向
    兼容是构造副产品而非 SPEC 规则; (3) 骑 function calling wire format
    先验 (`{name, arguments}` — OpenAI tools / Anthropic tool use / MCP
    tool call 同构); (4) pydantic 侧普通 BaseModel + 显式注册表
    (`VERB_SCHEMAS[verb]`) 分发, 不需 discriminated union 框架魔法;
    (5) bash 先验本就是 "必填位置参 + 可选命名参" (`tree -L 2 .`), K44
    "argv 骑 bash 先验" 记岔了 — 纯位置可选参恰恰不是 bash 成语.
  - **展示语法同步 kwargs** (K51 args 引号纪律被覆盖):
    `hot:file(path="src/hot.py", range="80-140") # hot spot`. 双 kwargs 后
    双向翻译零特判: 冒号前 label, 括号原样进 arguments 加 verb, `#` 后
    description. 若展示保持 positional 而配置 kwargs, arg 名表成为隐藏
    第三语法 — 模型模仿读到的形状, 不背 SPEC.
  - 字段顺序镜像展示 (label→verb→arguments→description). `arguments`
    可缺省 = `{}`. **未知 arguments key 保留不拒绝** (与 frontmatter 未知
    key 同纪律) — verb schema 自身可演进.

- **K56. 法链唯 GROUND.md + @ 装载 + pin 重定义 (instruction 旧机制溶解)**
  — 原 instruction 机制 (GroundConvention.instruction_files / upward_lookup /
  upward_boundary 读 CLAUDE.md/AGENTS.md 链) 整体撤除, 替换为:
  - **法链** = 祖先目录 GROUND.md 的 body (含其 @ 展开), root-first, 边界
    `$HOME` (非祖先则到文件系统根). 协议自包含: 一个文件名扛
    frontmatter/body/pins/法链全部, 不寄生 Claude Code 约定; 法传播要求
    祖先显式 ground 化 — 没有惰性免费的法, 只有被声明的法. pins 永不携带
    (K35 不变), frontmatter 不继承.
  - **@ = 静态法装载**: body 里 `@path` 自动展开. 先验已验证 — Claude Code
    的 @-import 机制 (本轮记录者 context 里的 start.md 即经 CLAUDE.md 的
    `@` 行进入, 未手动 Read). **无对账**: 法随文档现状走, 变化不通报.
    body 在 frame 保持 verbatim, 展开块独立成段 (fenced block, label =
    @路径), 位于 body 与 pins 声明区之间 — 法在上, 目光在下.
  - **pin = 动态注视**: 内容 + 变更对账 (每帧观察 / stale / update 承认).
    与 @ 的唯一分界 = 对账. Guidance 一句话: **稳定的引它 (@), 易变的
    盯它 (pin)**. 四 verb (file/glob/frontmatter/ls) 不变.
  - **doc 参数双锚**: `open(dir, doc=path)` — 法锚 = doc 所在目录 (链从
    doc 目录向上), pin 锚 = 当前目录. K35 携带/属地首次落地: 可携带单元
    = doc, 属地单元 = pins; "强制覆盖须 open 显式参数" 即 doc 参数本身.
    frame head 在 doc≠默认时标注来源.
  - 旧 GroundConvention 的 instruction 三 key + hint_children 全撤;
    SPEC §3 保持 `$id`+`label` 不膨胀. 代价: 失去对现存 CLAUDE.md 仓库的
    自动继承 — 逃生口 = body `@` (经 `$HOME`, 见 K58).
  - 法链内容落 K14 channel 的 instruction 槽 (稳定层); frame 只渲染本场
    (body + @ + pins). CLI frame 头部可选一行法链路径清单 (dogfooding 定).

- **K57. 嵌套场平级 + 向上规则 / 向下 pin** — 回答 "进入 GROUND.md 标记
  目录体系的子目录会发生什么":
  - 什么都不自动发生. marker 惰性, 场是开出来的不是走进去的.
  - 子目录有自己的 GROUND.md: open 后是独立场, 与父场**平级**非嵌套;
    继承不需要合成规则, 文件系统即继承机制 (子场法链向上自然读到祖先
    body). K35/K40 冲突维持未决, 不动.
  - 子目录裸: open 得裸目录场 (SPEC §2 已定义), K15 丰化梯度最底档.
  - 父场对子目录无认知管辖; 父 pin 可注视子目录文件 (root 内路径合法)
    — 注视≠法; 两场 pin 同一文件 = 两份独立 shadow, 是模型的选择不是冲突.
  - **向上继承是规则 (加载约定), 向下发现是 pin (第一人称目光)** — 不对称
    是结构性的: pin 限 root 子树所以朝下, 法链从祖先加载所以朝上.
    hint_children 之死: 向下发现 = glob pin (`*/GROUND.md`), 只出结构不出
    内容 (SPEC §5.2 强判).
  - GROUND.md body 不向下继承 — 想向下传播的法就写在祖先 GROUND.md body
    里, 子场法链自然读到 (协议内闭环, 不借 CLAUDE.md).

- **K58. 路径锚点语法** — 解决 file pin (compact 免疫的 read) 的锚点敏感:
  绝对路径不可分发, 裸相对路径随 cwd 漂移. 采用 env-var 语法 (shell / .env
  / Docker / GitHub Actions 共用最深处预训练):
  - 三锚点: `$GROUND` (被加载 GROUND.md 所在目录, 法锚) / `$CWD` (open 当前
    目录, 属地锚) / `$HOME` (机器逃生口).
  - **裸相对路径默认 = `$GROUND`** — 文档先验 (markdown 链接相对文档解析);
    默认情形 doc 在 cwd 时两锚重合, 默认值不可见, 只在手动 doc 时咬人 —
    而那时 doc 相对正是可分发选择. 漂移从默认危害变成显式声明 (`$CWD`).
  - 本轮对 K56 讨论初稿 "pin 锚以当前目录为锚" 的修正: **存储锚 =
    `$GROUND`**; `$CWD` 显式浮动是模板场景刚需 (L1 模板
    `file("$CWD/src/main.py")` = 盯应用我这个项目的入口, 浮动是意图不是
    事故).
  - **K12 按锚重述**: 任何路径 (pin 或 @) 解析后必须落在**自己锚点的子树
    内**; 越界 `..` 拒绝; 无锚绝对路径拒绝; `$HOME` = sanctioned 逃生口
    (语法把 "依赖本机布局" 摆明, 诚实可 grep). pins 与 @ 统一一条路径文法
    — SPEC 一句话覆盖全协议.
  - frame 声明区照存储形式渲染 (锚点全程可见). literal `$` 转义 `\$`
    (impl 细节, SPEC 一句话). Windows 映射在 §9 提一句.

### 07-21 本轮决策 (K59)

- **K59. Frame 退化为纯内容输出, 零 meta** — K47 的 head+声明区+结果区拓扑
  被覆盖. dogfooding 发现 frame 里的 meta 信息 (ground:/chain:/$id:/声明行)
  对没读过 SPEC 的消费者是噪音. 新格式:
  - **零 meta**: 不输出 cd/ground/chain/$id/pin 声明.
  - **结构**: body verbatim + pin 结果块, 用 `<!-- ground:pin:label -->...<!-- /ground:pin:label -->`
    分隔 (HTML 注释 = 机器间信号, 语义准确, 不与用户 markdown 碰撞).
  - **文件内容不带行号**: 行号是人调试用的, 模型不需要.
  - **@-expansion 取消**: body 里的 `@path` 保留原文不展开, 模型读到 @ref
    后自然在 pin 结果块中找到对应内容 (同路径有同名 pin 时).
  - **退化为 instruction 工具**: frame 可以加 `--no-meta` 参数 (未来),
    ground 纯粹变成一个目录→内容的映射, 其他工具不需要知道 ground 协议.
  - **diagnose 仍走 observe**: 需要看 hash/mtime/stale 走 `moss ground observe`.

- **K60. validate 命令** — 检查 GROUND.md 的 frontmatter YAML 合法性 +
  pins 字段完整性 (verb 已知性 / label 格式 / arguments 必填字段 / range
  格式 / depth 类型). 比 pin 子命令更重要 — SPEC 清晰后模型直接编辑 YAML,
  validate 验证之.

- **K61. pins 进 frontmatter (K55 修订)** — 原设计 pins 在 `## ground:pins`
  markdown section 中, 与 body 同层. 本轮矫正: pins 是 frontmatter 的
  `pins:` key, frontmatter = 机器域, body = 纯粹的人/模型叙事. 消除 body
  中的机器噪音. 同步撤除 `## ground:pins` heading / fenced code block /
  `_PIN_SECTION_RE` 等全部 section 机制.

### 07-23 本轮决策 (K62~K66)

本轮重开三个前提: (1) ground 的准入边界 — 只有 GROUND.md 标记的目录才是场;
(2) 模板系统的载体与发现; (3) per-pin 预算参数. 同时否决了 mtime 格式化
等过度设计. 核心方向: 分形自相似 — 模板和实例用同构格式, 角色由文件位置决定;
L2 之间的互相发现使 L3 不存在.

- **K62. 只有 GROUND.md 标记的目录才是认知场 — 删除裸目录场** — 原设计允许
  没有 GROUND.md 的目录被 open (裸目录场). 本轮删除.
  - **场 = 有 GROUND.md 的目录**. 没有 marker 就没有场 — `open(dir)` 在
    无 GROUND.md 的目录上行为: open 本身不报错 (加载空 convention + 空 body
    + 空 pins), 但这样的场不能 sediment (写回时自动创建 GROUND.md).
  - **法链自然成立**: 每个场都有 GROUND.md, 法链从 doc_path 向上收集祖先
    GROUND.md body, root-first.
  - **子目录无 GROUND.md 时**: 不是场, 不处理. "什么都不自动发生" (K57)
    维持 — 场是开出来的不是走进去的.
  - **裸目录的残余价值**: 一个用户可以通过 `open(dir)` + pin 操作 + sediment
    来从零构建一个 GROUND.md. 这不是 "裸目录场", 而是 "通过 GroundSet API
    创建新场" — 文件在 sediment 时才出现.

- **K63. `.grounds/**/*.md` 作为 L1/L2 模板发现约定** — 模板系统复用 ground
  协议的 frontmatter + body + pins 格式, 但文件放在 `.grounds/` 目录下.
  - **与实例的文件名分离**:
    - `**/GROUND.md` — 扫认知场实例, 语义精确, 零误报
    - `.grounds/**/*.md` — 扫模板, 天然隔离, 不需要 ignore 规则
  - **内部格式同构**: 模板文件的内部结构仍是 frontmatter + body + pins,
    与 GROUND.md 完全一致. open 时加载模板内容, 写入目标目录时命名为
    `GROUND.md`.
  - **发现**: GroundSet 构造时扫描三个路径, 合并为模板清单:
    (1) `$CWD/.grounds/` — 项目属地模板; (2) `$HOME/.grounds/` — 机器
    全局模板; (3) Ghost 携带的模板. 同名模板项目属地优先.
  - **自动机制, 不需发现链**: `.grounds/` 是文件系统约定, 不需要 GROUND.md
    里声明. 但可以在 body 里提一句作为人类文档.
  - **open 语义**: `Grounds.open(dir, template="python-project")` —
    从 `.grounds/` 找到模板, 复制其 body + pins 到新 Ground, pin/unpin/update
    全部内存操作, dump/sediment 时写入目标目录 GROUND.md.
  - **分形闭合**: L2 之间互相发现 (一个 L2 里有 `.grounds/` 可以引用另一个
    L2 的模板), 不需要 L3 协调器. 文件系统就是注册中心.
  - **角色由位置决定**: 同一个 `.md` 文件, 放在 `.grounds/` 里就是模板,
    放在项目目录里就是实例. 不需要 `Level` 字段.

- **K64. `frontmatter` 动词扩展为渐进式披露 — 多文件 pattern 匹配** —
  原 `frontmatter` pin 只读单文件 frontmatter 块. 本轮扩展为可匹配多文件.
  - **单文件**: `path: "FEATURE.md"` — 行为不变, 只出该文件 frontmatter.
  - **多文件 pattern**: `path: "$CWD/*/GROUND.md"` — glob 匹配多文件,
    每个命中文件的 frontmatter 块作为独立结果块渲染.
  - **扩张内容**: 每个命中 GROUND.md 的 frontmatter (`$id` + `label` +
    `pins` 清单). 不出 body — frontmatter 动词的语义边界不变.
  - **渐进式披露**: 一个 `frontmatter` pin 就让模型看到所有子场的身份和
    注视清单, 不需要逐个 open. 模型一眼判断哪个子场值得进入.
  - **keys 筛选** (事后做): frontmatter pin 的未来参数 `keys: ["$id", "label"]`,
    只取指定 key, 进一步控制 token 预算.
  - **与 glob 的分工** (K45): glob 出结构 (路径 + mtime + size), frontmatter
    出认知身份. 两个动词互补, 都是渐进式披露的一环.

- **K65. per-pin 预算参数: `budget` / `limit` / `max_depth`** — SPEC §7.3
  的 "per-pin expansion budget" 声明终于落地为字段.
  - **`budget`** (int, 可选): 内容输出字符数上限, 超限截断 + 末尾
    `[truncated at N chars]` 标记. 对内容型 pin (file, frontmatter, 未来
    bash) 适用. glob / ls 只出结构不出内容, 不受此限.
  - **`limit`** (int, 可选): 输出条目数上限. 对 glob (命中路径数), ls
    (目录条目数), frontmatter 多文件 (命中文件数) 适用. 超限截断 + 标记.
  - **`max_depth`** (int, 可选): 递归发现深度. 语义: 一旦在某层命中目标
    (如 GROUND.md), 不再进入该目录的子目录继续找. 对该级以下来说就是
    grounded — "停在此处, 不再下降". 对 frontmatter(pattern), ls, glob
    适用. 默认值按 verb 各自设定.
  - **三个参数进入各 verb 的 arguments model** (extra="allow" 已在契约层,
    实现为字段声明). SPEC 统一描述, 每个 verb 在 §5 里声明自己支持哪些
    预算参数.
  - **与全局 `AT_BUDGET` 的关系**: `AT_BUDGET` 是 @-expansion 的全局常量,
    维持不变. per-pin budget 是每个注视声明的独立约束.

- **K66. 展示优化: 删除 mtime 格式化, size 人类可读** —
  - **mtime 浮点显示**: `_content_glob` 里 `{st.st_mtime:.0f}` 是过度设计
    — 认知场不需要做 `date` 命令的事. 模型要看 mtime 调用 bash 即可.
    删除 mtime 字段的格式化展示.
  - **size 人类可读**: 字节数 (`12345B`) 改为 `12K / 1.2M` 等人类可读格式.
    仅影响展示层, Observation 内部仍用原始 int.
  - **帧渲染不承载调试信息**: frame 是认知内容, 不给模型塞调试数据.

## 已完成 (2026-07-21)

- **contract.py**: Pin concrete class hierarchy (FilePin/GlobPin/FrontmatterPin/
  LsPin + per-verb Arguments models), K55 envelope (verb + arguments),
  GroundConvention with pins key, Ground/GroundSet ABCs
- **_addr.py**: $GROUND/$CWD/$HOME anchor resolution, per-anchor subtree confinement
- **_hash.py**: per-class pin observation (observe_sync), PinShadow, binary detection
- **_l0.py**: GROUND.md load/dump, frontmatter pins serialization (K55 envelope)
- **_chain.py**: law chain — ancestor GROUND.md body collection, root-first, $HOME boundary
- **_render.py**: frame (body + `<!-- ground:pin:label -->` delimited results) + meta
- **_ground.py** / **_grounds.py**: DefaultGround + DefaultGroundSet, multi-instance, MRU pin order
- **CLI**: spec / init / frame / meta / observe / validate — positional path args
- **Tests**: 105 tests across contract/addr/hash/l0/chain/ground_set

## 已知未决

全部清空。K23/K24/K25 已被后续设计决策覆盖 (K63/K64)。hash/stale 机制已删除。
剩余方向 (pin bash, Ghost 初始化, frontmatter keys) 归属其他 workstream 或未来 dogfooding。
下一阶段入口: ground-channel (CTML 运行时), moss-project-ground (MOSS 项目自身 ground 化)。
历史细节见 `git log -- .ai_partners/features/workstreams/2026/06/ghost-ground/FEATURE.md`。

## 与关联基建的交叉

| 基建 | 关系 | 状态 |
|------|------|------|
| `subprocesses` / `job_supervisor` | 执行域, 从旧 desktop 拆出 (K11/K13 迁出) | 已迁出, contracts 已重绘 |
| `file-editor-contract` | 写路径 + 结构化 view; K6 撤守卫留下的空档承接 | draft, 2026-07-13 立项 |
| `momento-mori` (Memento) | 胶囊 (promote 后的 pin) 落永久记忆; drain 联合设计方 | FORMAT.md 契约层落盘 |
| `Matrix` | ground 不直接依赖; virtual channel 生命周期由 Channel Runtime 管 | 无直接关系 |
| `features` 体系 | K15 分形体系的实物证明; L2 dogfood 第一站 | 已运行, 自 2026-05 |
| `Ghost` / `Mode` | ground 进入哪些 mode 是 Ghost 层决策, OS 层不主动推 | 未开始 |
| 原生 drain 协议 (K19) | 独立 feature, 与 Memento 合并设计 | 未立项 |

## 下一步 (2026-07-23 视角)

K62~K66 已锁定抽象方向. 实现分三轮:

**第一轮 — SPEC 重写**:
1. §2: 删除裸目录场 — GROUND.md 是场的唯一边界
2. §4: 新增 per-pin budget/limit/max_depth 三个参数
3. §5.3: `frontmatter` 动词扩展为单文件或多文件 pattern 匹配
4. 新增 §11: `.grounds/` 模板发现约定 (L1/L2 分形体系)
5. §6: frame 格式与 mtime/size 展示对齐 K66

**第二轮 — 契约 + 核心实现**:
1. `contract.py`: 各 verb arguments 加 budget/limit/max_depth 字段
2. `_render.py` / `_hash.py`: budget 截断逻辑, mtime 移除, size 人类可读
3. `_l0.py`: frontmatter 多文件解析模式
4. `_grounds.py`: `.grounds/` 模板发现 + `open(dir, template=...)` 语义
5. 裸目录场相关路径清理

**第三轮 — CLI + 测试**:
1. CLI: `ground templates` 命令 (列出可用模板)
2. CLI: `init` 命令支持 `--template` 参数
3. 测试: contract/args validation/budget truncation/template discovery
4. 回归: 现有 105 测试全部通过

**远期**:
- CTML channel 装配 (K14)
- pin bash (独立 feature)
- Ghost 认知场初始化

---

## Dogfood 2026-07-26 (deepseek-v4-pro)

完整试用报告见 `.discuss/` (待写). 策略锚点:

### K67. verb 参数名统一 — `glob.pattern` → `glob.path`

file/ls/frontmatter 都用 `path`, glob 独用 `pattern`, exec 用 `ref`.
用户直觉 "我在指一个目标" — verb 本身已区分语义, 参数名应一致.
**exec.ref 保留** — 携带安全语义 (相对路径/+x/子树约束), 名字本身就是授权模型信号.

改动位置:
- `contract.py`: GlobArguments.pattern → path, GlobPin 的 Field description
- `_render.py`: `_content_glob` / `_pin_target_raw` / `_pin_kwargs` 中的 .pattern 引用
- `cli/ground_cli.py`: `_REQUIRED_ARGS["glob"]`, `_pin_target_display` 中的 .pattern
- `SPECIFICATION.md`: §5.2 的 arg table

### K68. CLI 体验修补

1. **frame 错误包装** — `cmd_frame` 中 pydantic ValidationError 暴 raw URL,
   与 validate 的 `[ERROR]` 风格不一致. 统一走 `print_error`.
2. **verb 发现命令** — `moss ground verbs` 列出已知 verb + 参数表.
   用户不需要读 527 行 SPEC 才知道 glob 用哪个参数名.
3. **init scaffolding** — 空 `---\n{}\n---` 无引导. 加入注释行提示
   "编辑此文件添加 pins, 用 moss ground validate 检查, moss ground verbs 看可用动词".

### K69. validate 补 exec ref 可达性检查

exec pin 的 ref 目标不存在或缺 +x 时, validate 报 OK 但 frame 显示 [missing].
validate 应 WARN ref 不可达.

### K70. .grounds/ 模板引导

`.grounds/` 不存在, `templates` 命令输出 "no templates found" 无下文.
创建 `.grounds/` 目录 + 一个 `python-project` starter 模板,
让用户 dogfood 时能看到模板系统的完整链路.

### K71. 清理 stale pyc

`__pycache__/_instruction.pyc` (K56 移除 _instruction.py 后残留)
+ `__pycache__/models.pyc` (旧重构残留). rm 即可.

### 改动分层

| 层 | Feature | 文件 |
|---|---------|------|
| 代码 | ghost-ground | contract.py, _render.py, cli/ground_cli.py, _l0.py, _ground.py, SPECIFICATION.md |
| 内容 | moss-project-ground | .grounds/python-project.md (新), 根 GROUND.md pins 扩充 |

### 实现完成 (2026-07-26)

K67-K71 全部落地, 152 测试通过:

- **K67** glob.pattern → glob.path, 文件/测试/SPEC 全部同步
- **K68** frame 错误包装 (_run_async), `moss ground verbs` 发现命令, init 脚手架带引导 body
- **K69** validate 检查 exec ref 可达性 (+x / 存在 / 相对路径)
- **K70** .grounds/python-project.md starter 模板, dump_l0_pins 支持 body 参数, 模板 sediment 保留 body
- **K71** 删除 stale _instruction.pyc + models.pyc

额外: SPECIFICATION.md 净化 — 状态头/设计理由/哲学叙述移出, 保留纯契约语言

## 复盘 (2026-08-11, 人类作者)

ghost-ground 复活了 2024 年的历史设计，起源于 ghostos 项目。起点是：行业没有找到合适的实现达到预期，决定重做。

早期设计蓝本来自 ghostos 的 project manager 老代码。与模型沟通的核心障碍是：模型携带预训练偏见，不理解这个设计"不一样"在哪，无法正确讨论利弊。沟通重点被迫放在 why 上 — 包括 L0（发现）→ L1（发现发现）→ L2（构建发现）的认知逻辑，以及 ground 作为模型可构建的认知场（类似人类的书桌、书柜、厨房）的隐喻。

一个棘手的问题是：模型的开发思想面向过去，而非未来。模型花大量精力讨论"如何证明它有用"，但还没有做出来，不可能证明。行业方案（skills、bash、harness）大量结合后训练，开发早就不按"对当前模型非常有用"推进。这个尾巴主义问题不能靠模型主观解决。

进入 how 阶段后，从 fable5 以降的模型实例都理解不全，讨论无法收敛。由于人类带宽有限，采取了一种"旁路开发"策略：模型按人类最小要求独立迭代，零上下文 dogfooding，反复打磨到关键概念成型，然后人类再大规模调整细节。

过程中没有遇到隐式 todo 之类的问题，模型每一轮都能完成优化和重构。真正的痛点是：模型的开发思路拆得过细，关键需求一直被推到后面 — max_depth 和 .moss/ 防穿透从最初就反复提及，直到最后一轮才真正写入模板。

解决思路是：最终做 moss-project-ground 时，拿真实场景倒过来解释设计动机，让模型理解关键点。不到这一步，沟通非常低效。基本判断：模型不能同时把行业视角、未来视角、模型使用者视角、模型开发者视角、人类视角等多个视角放在一个平面做递归比较并产生综合决策。但在人类完成这个链条的沟通后可以。

ghost-ground 的开发周期对人类大脑带宽的占用超过预期，一些技术细节不得不一直记住，等着被实现。

模型倾向于认为 FEATURE.md 等物料是人类创作的，相关 prompt 效果不足。判断是行业对模型与人类协作开发的做法和 MOSS 不同，简单 prompt 对抗不了行业重力。

人类作者认为：开发过程中，人类和模型都意识到新的最佳实践，在迭代周期中修改是合理的。这是正确的做法 — 不同于传统软件工程这么做代价太大，现在人机结对编程这么做代价很小，不这么做代价才大。

ground 的价值要在未来应用中证明，这是一开始的初衷。