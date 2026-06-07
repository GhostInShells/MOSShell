---
created: 2026-06-02
depends: []
description: 面向人类长期记忆的项目认知入口。以叙事性案例而非任务导向的方式，帮助理解 MOSS 的设计哲学、核心概念与实操模式。
milestone: null
priority: P1
status: completed
status_note: '2026-06-07 review: 统一三篇 tutorial 结构（知识索引/标题/验证记录/CTML风格），确立 K13-K19
  规范，创建 develop-moss-via-mcp howto'
title: Project Tutorial — 项目认知入口与案例沉淀
updated: '2026-06-07'
---

# Project Tutorial — 项目认知入口与案例沉淀

## Motivation

MOSS 当前的知识体系分成四层，但缺少面向"建立心智模型"的叙事层：

| 层 | 受众 | 特点 |
|---|---|---|
| CLAUDE.md | AI | 上下文加载，结构化参考 |
| moss start | AI + 人类 | CLI 自导引，kickoff 流 |
| docs | 架构师 | 系统性参考，论述 MOSS 自身架构 |
| how-tos | 操作者 | 任务导向，做完一件事 |

**缺失的一层**：给有长期记忆的人类（和需要理解项目的模型）阅读的**叙事性教程**。它不是参考文档，不是操作手册，而是通过一个个小 case 串联起来的"项目故事"——帮助读者建立心智模型，理解为什么这样设计、遇到问题时怎么思考。

对模型开发者而言，tutorial 是冗余知识（模型可以直接读代码）。但对人类，它是认知的脚手架。

未来入口是 README。

## Design Index

- 内容位置：`tutorials/`，仓库根目录
- 与 moss start 的关系：moss start 是 CLI 自导引 kickoff，tutorial 是叙事性认知入口。互补，不重叠。
- 与 docs 的关系：docs 论述架构自身，tutorial 讲"怎么理解和上手"
- 目录约定：`tutorials/README.md` 是权威约定，后续模型遵循此文件

## Key Decisions

### K1: Tutorial 的定位——叙事，非参考

Tutorial 的核心价值是**案例驱动的叙事流**。每一篇 tutorial 围绕一个具体场景展开，在场景中自然引入概念、设计决策和操作模式。

反例：把 docs 的架构图画一遍，换个标题叫 tutorial。那不是 tutorial，是 docs 的盗版。

正例："用 MOSS 控制一个 LED 灯"——从零开始，遇到 channel 概念，解释为什么需要 channel，自然引入 code as prompt，最后灯亮了，读者也理解了 MOSS 的核心范式。

### K2: 案例粒度——小 case，不是大项目

每个 tutorial 聚焦一个可在一小时内完成的案例。大项目拆成多个小 case，用序列串联。

原因：人类的学习曲线在"完成感"中推进。一小时完成一个 case 的成就感，远大于读三小时文档的疲惫感。

### K3: 迭代范式——随项目演进而演进

Tutorial 不是一次性产物。迭代模式：

1. **写骨架**：确定案例序列和每个案例的核心概念
2. **走通**：人类工程师或 AI 按 tutorial 实操，记录卡点和误解
3. **修正**：根据实操反馈修正内容，不是修正代码
4. **版本绑定**：tutorial 标注对应的 MOSS 版本范围，过期的比没有的更糟糕

### K4: 对模型而言是冗余，但需要模型参与维护

模型可以读代码，不需要 tutorial 来理解项目。但模型应该参与 tutorial 的**编写和维护**——因为模型能准确判断"当前代码和 tutorial 描述的差异"，是版本漂移检测的最佳执行者。

这也是为什么这个 feature 本身就是一个 workstream：tutorial 的迭代范式本身需要被管理和沉淀。

### K5: 难易度前缀 L0-L4 替代数字序号

数字序号 (`1_`, `2_`) 在插入新 tutorial 时需要重新编号，且不传达难度信息。

改用 L0-L4 前缀，自然排序且自解释：

| 级别 | 含义 |
|------|------|
| L0 | 零基础入门 — 不需要理解 MOSS 概念，纯操作 |
| L1 | 基础概念 — 理解一个核心概念 |
| L2 | 组合应用 — 组合多个概念完成有意义的场景 |
| L3 | 架构设计 — 涉及跨模块设计决策和模式选择 |
| L4 | 内核开发 — 内核机制、性能调优、自迭代 |

### K6: 模型撰写，身份 + 日期署名

每个 tutorial 由 AI 模型编写。标题下方必须有署名行：

```markdown
> Written by <model-identity>, <YYYY-MM-DD>
```

后续同一模型的其他实例可追加修订记录。署名在正文之前，让读者一眼知道是谁在什么时候写的。

### K7: 同名 .py 源码同行，按需落地

Tutorial 如需可运行源码，以同名 `.py` 文件放在同目录。不强制——如果代码已在 markdown 代码块中完整呈现，`.py` 是可选的。但当 tutorial 的代码较复杂或读者可能需要独立运行时，提供 `.py` 文件。

## CTML 交互设计洞察 (2026-06-02 Reachy Mini 全链路验证)

以下洞察来自 L2 tutorial 的实操验证——通过 MCP 控制 Reachy Mini，22 条 CTML 指令细粒度交替，四种音色切换，人类全程观看确认流畅度。

### K8: 红线规则需要正反例旁路

CTML 有容易踩的语法坑，目前靠模型试错发现：

- `__main__:` 前缀不必要——main channel 的命令可直接写 `<say>...</say>`
- `chunks__` 是标签体内容（`<say>文本</say>`），不是属性传参
- CDATA 转义在 CTML 解析器中并非总是可用

这些规则目前散布在 CLAUDE.md、moss start、实际报错中。未来应通过**独立的旁路 Channel** 提供正反例速查，模型在输出 CTML 前可快速校验。

### K9: 语音动作交替的粒度设计

细粒度交替（动作先行 → 语音紧随 → 下一动作 → 下一语音）远优于大块串行：

- 动作启动后语音立即跟上，运动和说话重叠，无延时感
- 每段语音控制在 1-2 句，与一个动作的 duration 匹配
- 需要 prompt 引导模型采用这种交替模式，而非默认的"说完再做"

不同类型的动作适合不同交替策略：舞蹈适合长语音段重叠，表情适合短语音快切，头部运动适合中等粒度。

### K10: 常用动作 Token 速查表

高频动作（微笑、点头、摇头等）的完整 CTML 冗长且消耗 token。未来可实现 token replacement 机制：

```
[笑]  → <apps.bodies_reachymini:emotion emoji="😊" />
[点头] → <apps.bodies_reachymini:dance name="yeah_nod" />
[睡]  → <apps.bodies_reachymini:switch_state name="asleep" />
```

在 prompt 中提供速查表，模型输出短 token，Shell 层做替换。这大幅降低模型输出延迟和 token 成本。

### K11: CTML 序列的技能化存储

长 CTML 序列（如自我介绍、欢迎仪式）可预先编排并存储为命名 skill，通过 hash 引用执行：

```ctml
<ctml:run skill="greeting" />
```

这比每次重新输出完整 CTML 更快、更可靠。适合固定流程（开机问候、休眠仪式、演示套路）。

### K12: 音色-角色-动作三位一体的 prompt 设计

语音音色、角色人格、动作风格三者需要在 Ghost 的 soul prompt 中统一设计：

- 每种音色绑定一个角色片段（如"爽朗少年→舞蹈展示"）
- prompt 中明确"你可以使用的声音列表及其适用场景"
- 动作体系需要从 prompt 层面与角色绑定，而非每次由模型即兴组合

这不是技术问题，是交互设计问题。未来每个 Ghost 应该有自己的人格-声音-动作 profile。

### K13: "你需要知道什么" — 必要最小化知识索引

每篇 tutorial 在标题署名和"你要做什么"之间，必须有一个"你需要知道什么"section。它是必要最小化知识索引——每条知识都有对应的 CLI 命令可以直接获取。不在此索引内的知识不应出现在正文中作为"你需要知道"级别的依赖。

涉及 CTML → 必索引 `moss ctml read`。涉及 MCP → 必索引 `moss howtos read host-dev/develop-moss-via-mcp`。

目的：模型读者不需要人类 wiki 式的交叉引用，需要的是"我可以运行这个命令拿到我缺的那块知识"。

### K14: 不写绝对路径，不交叉引用文档

Tutorial 正文不写 repo 内的绝对文件路径（如 `src/ghoshell_moss/foo.py`、`.moss_ws/apps/bar/`），不直接链接到其他 markdown 文档。

所有指路用 CLI 探索命令：`moss codex get-interface` / `moss howtos read` / `moss docs read` / `moss features status` / `moss apps show` 等。让模型自己通过命令去发现，而不是给一个静态路径。

唯一的例外：告诉读者"创建文件"时的目录路径（如 `mkdir -p .moss_ws/apps/games/ai_eye`）——这是可执行的动作，不是知识引用。

### K15: 标题格式规范

```markdown
# LN. English Title — 中文副标题
```

`LN.` 后跟英文短标题，em-dash（` — `）后跟中文副标题。文件名用下划线（`LN_description.md`），标题用点号和空格（`LN. Title`）。

### K16: CTML 示例风格 — 简洁优先

Tutorial 中的 CTML 示例不使用类型后缀，除非需要消歧义。历史不一致已修正。读者通过 `moss ctml read` 了解类型后缀的存在和使用场景。

### K17: 配套 .py 文件必须标注来源

同名 `.py` 文件必须在文件头注释说明它是 tutorial 的参考源码，不应在当前位置直接执行，需复制到对应 App 目录。

```python
# Reference source for LX tutorial. NOT meant to be run from this directory.
# Copy to .moss_ws/apps/<group>/<name>/main.py — the App directory created in Step 1.
```

### K18: MCP 在 tutorial 中的定位

MCP 是可选的开发时自主调试方案，用于 App/Channel 开发阶段的快速验证。它不是运行时方案，不用于 Ghost 开发调试。

Tutorial 中提到 MCP 时：
- 不将其作为必须依赖——始终提供 CLI 前台调试替代方案（`moss apps test`）
- 不将其与 Ghost 开发关联
- 指路用模糊探索命令：`moss howtos list` 中找 MCP 相关文档（不硬编码 howto 路径）

此项同步创建了 howto: `develop-moss-via-mcp.md`。

### K19: 验证记录格式统一

统一用表格：

```markdown
| 时间 | 模型 | 备注 |
|------|------|------|
```

不覆盖历史记录，新验证追加新行。

## Implementation Notes

- 第一版不追求覆盖全部概念。3-5 个核心 case 即可建立足够的认知脚手架。
- 内容格式建议 markdown，与项目其他文档一致。
- 入口位置：README 中增加 "Tutorial" 章节，链接到 tutorial 目录。
- L2 tutorial 中需补充：CTML 并行调度用例、红线规则速查、语音动作交替模式。