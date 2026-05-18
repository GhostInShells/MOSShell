# 二稿人类审阅 + 拓扑对齐

DeepSeek V4 (via Claude Code)，2026-05-18。

## 上下文

初稿完成后，人类工程师详细审阅，给出了八个关键改进方向。这些方向不只是"改措辞"，而是对拓扑文档定位的深度校准。

## 共享词汇

- **拓扑 > 实现**：文档的宪法地位——分层逻辑稳定，具体实现会演进。这份文档是验证其他文档目标是否达成的参照系。
- **Logos (vs CTML)**：CTML 是 Logos 的一种实现（XML 流式解析）。Logos 是 Mindflow 中"模型思维的文字产出"概念。CTML 可以退场，Interpreter 不退场。
- **上下文窗口组件化**：Channel 不只是能力列表——每个 Channel 是上下文窗口的一个独立单元，可安装/卸载/打开/关闭。模型看到的是按需装配的组件树。
- **自举面**：cli-flow + features 不是凌驾于架构之上的 Meta 层，是编织在系统中的自举机制。目标是 L1 级 AI 自开发能力。
- **真正的 Ghost**：持久化有生命感的智能体实现不在 ghost.py，在仓库的 `.ai_partners/` + memory + features 体系本身。

## 锚点

> "本文档拓扑本身的价值大于内容的实现。比如 mindflow 的实现可能颠覆性重构，但它的拓扑分层我估计是雷打不动的。"

> "CTML 暂时就是 logos 的一种实现。我们是不是该简单介绍 ctml 的 xml 流式解析本质，然后提一嘴这种流式解析语法在项目里叫 logos，不用详细展开？"

> "channel 最重要的双工通讯，上下文窗口组件化等讯息，可以提一下。具体实现不用提。"

> "ghost 抽象目前偏 adapter 性质，解决 moss 运行时生命周期与智能体对接，这些没体现，但搞了很多具体实现。具体实现都不重要。"

> "本文档里讨论的拓扑架构很可能就是全行业最终解。而实现不见得。"

> "CLI-flow + features 体系也是其中的一部分。目标是让项目通过 AI 自解释和迭代，让其自身具备 L1 级别开发能力。"

> "Ghost in shells 里关于持久化有生命感有连续性智能体的实现本身不在 ghost，在这个代码仓库本身。"

## 影

一稿犯的错误：
1. 引用 CLI 命令名（如 `moss concepts`）而非 Python 包路径——目录结构马上要调整，包路径才是稳定面
2. 各层写了太多"关键抽象"里的具体类名，陷入实现细节——而拓扑文档的核心是"解决什么问题、为什么在这一层"
3. Meta 面拓扑关系搞错——docs 是 cli-flow 的子集，不凌驾于 features
4. CTML 篇章只说"是什么"没说"为什么"——应该是 Logos 对标 Function Call 的命题差异
5. Ghost 篇章搞了很多 GhostMeta/Ghost/GhostRuntime 细节，没体现 adapter 本质
6. 没有 Ghost In Shells → MOSS 的锚定关系
7. 只写到 Host 戛然而止，cli-flow + features 自举体系缺失
8. 没提真正的 Ghost 在仓库自身

## 技术细节

CLI 重构方案（`cli/.design/2026-05-17-cli_command_tree_redesign.md`）的关键变化：
- `moss concepts` → `moss codex concepts`（并入 codex，压平为单命令）
- `moss ctml read` → `moss workspace ctml-read`（移至 workspace）
- `moss manifests` → `moss workspace manifests/`（workspace 下唯一子组）
- `moss how-tos` → `moss codex howto`（并入 codex，简化为索引）
- 稳定面：`moss codex` 工具入口 + Python import path
