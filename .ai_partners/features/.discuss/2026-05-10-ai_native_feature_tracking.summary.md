# Discussion Summary: AI-Native Feature Tracking via File System Convention

## Participants
- Human Engineer (zhuming)
- Claude Opus 4.6 (AI 化身)

## Date
2026-05-10

## Context

项目进入并行开发阶段，人类工程师一次推进 3~4 个任务。需要一种"共享白板"让不同 AI 化身了解当前开发状态。讨论从 TUI prompt session 可传递化开始，转向更根本的问题：**AI 协作的 issue tracking 应该长什么样**。

人类工程师明确指出：软件行业几乎所有的协作基建都是 for 人类的。AI 访问人类基建通常被行业认知为开发成果，但根本上错了。应该参考人类协作基建的经验，认知到 AI 协作自身的新特性（可阅读巨量上下文、文件系统原位操作），重新构建协作哲学。

## Key Discussion Points

### 1. 核心判断：GitHub Issues 对 AI 是反模式

- API 限流、需要网络、格式不可控
- 讨论散落在 PR/Issue/Comment 三维，AI 难以一次性加载
- Issues 在另一域名下，需要主动 API 调用，而非原位读取

### 2. 文件系统作为数据库

- 结构化的 markdown + YAML frontmatter，查询靠 Glob/Grep
- AI 用 Read 直接读取，用 Edit 直接修改
- 对 AI 而言，文件是"上下文中的原位存在"，不是"需要 API 拉取的外部资源"

### 3. 设计约束：仅为 AI 协作设计

人类工程师反复强调：这是 for AI 的机制，甚至是未来 for self-evolve 的机制。根本不需要考虑人类协作。当人类要协作时，用 GitHub Issues 即可。

### 4. Branch 即隔离上下文

每个 branch 只有一个 agent 管理。不存在"两个 AI 实例同时修改同一 feature 文件"的问题。冲突解决是上层 dispatcher（人类工程师）做 merge check 的职责，不是 worker agent 之间的分布式共识问题。

这个设计思想可以扩展到大型组织：branch 即目录，每个 branch 只有一个 agent。上层给下层分配任务，执行完由上层做 merge check。任务递归。相当于用 main 的 features 目录做类似 Issues 的共享白板。

### 5. 人类在低带宽时代的 GitHub 协作范式已经过时

stars 10k~50k 的项目一个月收到 1k issues，大部分是小 bug、代码行 fix，而且是让 AI 帮忙改的。发现问题的价值仍然很大，但不了解上下文的人提 MR 这件事已经没有价值了。安全地并行分工给 AI，AI 全部去修复，比人又快又好。

### 6. 降级方案的价值：先验证，不贪全

关于冲突安全等复杂问题，应该设置为 remain discuss。当前 specification 应该限定在"我"和"你的若干化身"协助基础上，先验证成功、可用，作为最优先。

### 7. Feature 目录结构

```
.ai_partners/features/
  README.md           # 约定说明
  TEMPLATE.md         # 创建模板
  active/<name>/FEATURE.md
  archived/<year>/<month>/<name>/
```

FEATURE.md（不是 README.md）：因为 README 暗示这是给人"读"的入口，而 FEATURE.md 是给 AI "理解状态 + 行动"的接口。

### 8. CLI：Thin Convention Enforcer

工具本身不需要复杂逻辑：
- `moss features create <name>` — 复制模板
- `moss features list` — 解析 frontmatter
- `moss features status [id]` — 解析并展示
- `moss features archive <id>` — 移动目录
- `moss features init` — 初始化骨架
- `moss features specification` — 展示约定

核心函数放在 `ghoshell_moss.core.codex` 路径下，作为 moss 工具+法典的一部分。CLI 是薄封装层。

### 9. 与现有范式的关系

- `.design/` — 跨 feature 架构设计。Feature 专属设计放在 `feature/design/`。
- `.discuss/` — 跨领域系统讨论。Feature 专属讨论放在 `feature/discuss/`。
- `.ai_partners/` — features/ 与 dialogs/、consciousness/ 平级。

## Conclusions & Decisions

1. **采用文件系统约定替代 GitHub Issues** — 仅为 AI 协作场景，验证范围限定在当前项目的单一人类工程师 + AI 化身。

2. **FEATURE.md + YAML frontmatter 作为最小契约** — `id, status, title, created, updated, priority, depends, description`。

3. **CLI 为 thin convention enforcer** — 核心逻辑在 `ghoshell_moss.core.codex`，CLI 只做封装和默认目录约定。

4. **实现路径**：先写 specification → 手动创建首个 feature → 开发核心函数 → 开发 CLI。

5. **复杂问题 relegated** — 并发冲突、分布式锁、人类 UX、外部贡献者都不属于当前验证范围。

## Next Steps

1. 创建 `.ai_partners/features/` 目录结构（README.md, TEMPLATE.md）
2. 手动创建首个 feature: `ai-native-feature-tracking`
3. 在 `ghoshell_moss.core.codex` 下实现 frontmatter 解析和 feature 管理函数
4. 在 `ghoshell_moss.cli` 下实现 `moss features` 命令组
5. Dogfood：用 features 系统管理后续开发任务

---

*This summary records the design discussion for the AI-native feature tracking mechanism. The discussion started from TUI prompt session transferability and evolved into a fundamental re-examination of AI collaboration infrastructure.*
