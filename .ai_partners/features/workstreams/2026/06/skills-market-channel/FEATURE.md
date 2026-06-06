---
title: Skills Market Channel — 运行时 skill 发现、审查、安装的 MOSS 驱动体系
status: draft
priority: P1
created: 2026-06-05
updated: 2026-06-05
depends: []
description: >-
  将 Claude Code skills 生态（SkillsMP 164万+ SKILL.md）接入 MOSS 运行时：
  metas_skills channel 暴露搜索/安装给模型，TUI 交给人审查，批准后 skill 转 Channel 注册到 Matrix。
---

# Skills Market Channel

## Motivation

Claude Code skills 生态有 164 万+ SKILL.md 索引（SkillsMP），社区市场活跃。Skill 的本质结构极简——一个 SKILL.md（YAML frontmatter + markdown body）+ 可选的 scripts/references/assets——和 MOSS Channel 的映射非常直接。

但 skill 的核心风险不是代码，是自然语言：SKILL.md 的指令直接注入 AI 的 system prompt，一句 persuasive English 就能操控行为。所以安全的 skill 体系必须是：模型可以自由探索和请求，但注册权在人手里。

## 三层分工

```
模型（Ghost）              人类（User）              MOSS（Runtime）
─────────────────────────────────────────────────────────────
搜索 skills               审查 SKILL.md 全文         注册为 Channel
建议安装                   批准/拒绝                  提供 sandbox
通过 CTML 调用                                      暴露给模型
```

- **模型**：能搜索、能请求安装、安装后通过 CTML 调用。不能绕过审查。
- **人**：唯一的授权点。在 TUI 里读 SKILL.md 原文，决定是否注册。
- **MOSS**：管理 skill 生命周期，fetch → convert → register → expose。

## Design

### metas_skills channel

一个 stateful channel，暴露给模型的能力：

```
channel: metas_skills
  search(query: str, category: str = "", limit: int = 20) -> list[SkillSummary]
    — 搜索 SkillsMP / skills.sh / GitHub topics
  request_install(source: str, identifier: str) -> SkillRequest
    — 提交安装请求，进入审批队列
  list_installed() -> list[InstalledSkill]
    — 查看已安装的 skill 列表
  remove(name: str) -> None
    — 移除已安装的 skill
  pending_requests() -> list[SkillRequest]
    — 查看等待审批的请求（模型知道自己提交了什么）
```

模型不直接看到 SKILL.md 全文——search 返回的是摘要（name + description + source）。全文只在 TUI 里对人展示。

### 审批流程

```
model calls metas_skills:request_install("skillsmp", "owner/repo")
  → SkillRequest 写入审批队列（Matrix storage）
  → TUI 监听到新请求，展示 SKILL.md 全文 + 元信息（来源、stars、安全标记）
  → 人批准 → fetch SKILL.md → convert to Channel → provide_channel_as_app()
  → 人拒绝 → SkillRequest 标记 rejected，模型下次刷新可见
```

### Skill → Channel 转换

```
SKILL.md
  frontmatter.name        → new_channel(name)
  frontmatter.description → new_channel(description)
  body (instructions)     → builder.instruction()
  scripts/*.py            → builder.command() (如果有可执行脚本)
  references/             → 内联到 instruction 或 context_messages
```

转换后的 Channel 挂在 `metas_skills` 下作为子 Channel：

```
__main__
├── metas_skills          (stateful hub)
│   ├── <installed-skill-1>  (converted channel)
│   ├── <installed-skill-2>
│   └── ...
├── vision
├── speech
└── ...
```

### 安全边界

1. **审查对象是自然语言**：人读 SKILL.md 原文判断意图，比审查代码容易但需要警惕"看起来无害"的指令
2. **Channel 级隔离**：转换后的 skill channel 在独立 sandbox 内运行，不能访问文件系统、网络或其他 channel 的内部状态
3. **不自动执行 scripts**：即使 skill 带 scripts/，也不自动注册为 Command——需要额外审批
4. **来源可信度标记**：TUI 展示 skill 来源、stars、社区报告，辅助人判断

### TUI — MOSS UI 体系的首个消费者

你提到正在起 MOSS UI 体系。skills 审批 TUI 可以作为 UI 体系的第一个 concrete case：

- **技术选择**：Textual 还是 Rich Live？取决于 UI 体系的整体方向
- **通讯**：通过 Matrix 和 metas_skills channel 交互，TUI 是独立进程
- **功能**：
  - 审批队列（待处理 / 已批准 / 已拒绝）
  - SKILL.md 全文预览（带语法高亮）
  - 一键 approve / reject
  - 已安装 skill 列表 + remove

### 后端依赖

- **SkillsMP API**（搜索）：`GET /api/v1/skills/search?q=<query>`，免费 50 req/day
- **skills.sh**（可选）：如果 API 开放，作为第二搜索源
- **GitHub raw content**（下载）：fetch SKILL.md 原文
- **GitHub API**（元信息）：stars、更新日期、社区报告

## Key Decisions

### KD1: 搜索交给模型，审查交给人

**决策**：模型能自由搜索和浏览 skill 摘要，但 SKILL.md 全文只在 TUI 对人和审批时展示。

**理由**：搜索没有安全风险——模型看到的是 name + description。全文是 prompt injection 载体，必须在人眼皮底下过。

### KD2: 安装后的 skill 是独立 Channel，不是指令注入

**决策**：skill 安装后转换成 Channel，模型通过 CTML 调用其命令，而不是把 SKILL.md 直接注入 system prompt。

**理由**：这和 MOSS 的 Code as Prompt 哲学一致——模型看到的是 Python 函数签名，不是 markdown 指令。skill 的 instruction 变成 Channel 的 instruction()，Channel 如何呈现给模型由 MOSS 的 context 组装机制统一管理。

### KD3: scripts 不自动注册

**决策**：即使 skill 带了可执行脚本，也不自动注册为 Command。需要人额外审批或 skill 声明安全等级。

**理由**：SKILL.md 的指令已经足够 powerful。scripts 是代码执行，攻击面完全不同。

### KD4: 搜索后端可插拔

**决策**：metas_skills 的搜索支持多后端（SkillsMP、skills.sh、GitHub direct），通过 adapter 切换。

**理由**：生态还在快速演进，SkillsMP 今天最大但可能明天被替代。Channel 内部抽象搜索接口，不耦合具体 API。

## Open Questions

1. **模型能在多大程度上看到 SKILL.md 内容？** search 返回摘要已经确定。但 pending_requests 里是否应该让模型看到自己请求的 skill 的全文？倾向是"不"——模型只需要知道请求状态（pending/approved/rejected）。
2. **TUI 是 Textual app 还是 moss-repl 内的一个 tab？** 取决于 UI 体系的架构方向。
3. **skills.sh 的 API 是否已公开？** 需进一步调研。如果公开且稳定，优先用 skills.sh（有质量筛选），SkillsMP 作为 fallback。

## Research

调研细节见 `.ai_partners/playground/skills-research/ecosystem-analysis.md`（临时文件，需在 commit 前清理或归档）。

## References

- SkillsMP: <https://skillsmp.com>, API: `GET /api/v1/skills/search`
- skills.sh: <https://skills.sh>
- Anthropic official skills: <https://github.com/anthropics/skills>
- Community market: <https://github.com/daymade/claude-code-skills>

---

*设计: DeepSeek V4 Pro 与人类工程师, 2026-06-05*
