# AI 认知外存：将文件系统树形索引暴露为 Ghost Memory Channel

## 背景

本次设计勘探的起点是：用户带着上一个 AI 实例完成了 `module_channel` 和 `notebook_channel` 两个原型后，开始讨论 memory channel 的设计。上一个 AI 的反馈停留在 "抽象太多、没有 5 分钟上手教程"，没有抓住用户布这个局的真实意图——用 `module_channel`（反射已有代码为能力）和 `notebook_channel`（文件系统即 scratch space）作为铺垫，引出 AI 如何自主管理自己的记忆这一核心命题。

本次探索中，当前 AI 实例（DeepSeek V4 via Claude Code）被要求完整走完以下路径：

1. 阅读三个核心文件建立 Channel 体系的理解基线
2. 阅读 `module_channel` 和 `notebook_channel` 的源码 + 测试建立体感
3. 探索 `moss features` 和 `moss how-tos` 两个已落地的记忆/知识管理体系
4. 追溯 `.discuss` 和 `.design` 的元讨论范式
5. 查看 `ChannelMeta` 数据结构中已预留的 `memory` slot
6. 用 Claude Code 的 multi-agent 并行分析上述各部分

勘探完成后，AI 识别出一个关键洞察：**项目本身已经在作为一个完整的记忆系统在运行**，memory_channel 不是要重新设计一个记忆系统，而是要把已经跑通的模式抽象成 Channel。

## 关键证据链

### ChannelMeta 已经刻好了 memory 的位置

`ghoshell_moss.core.concepts.channel:ChannelMeta` (见 `moss codex get-interface ghoshell_moss.core.concepts.channel:ChannelMeta`) 的注释原文：

```
ModelContext is built by many messages blocks:
 - instructions before conversation
 - memories                                    ← 这个 slot 已存在
 - conversation messages
 - dynamic context message before the inputs
 - inputs messages
```

`memory: list[Message]` 字段存在。`context: list[Message]` 也有对应的 `Builder.context_messages()` 装饰器。但 `memory` 没有对应的 `Builder.memory_messages()`。这不是设计疏漏——当时判断 memory 需要更明确的使用场景和治理规则再开 API。现在场景来了。

### 项目本身就是一个运行中的 Ghost 记忆系统

| 记忆类型 | 项目里的映射 | 载体 |
|---------|------------|------|
| 情节记忆 | `features/workstreams/` | FEATURE.md + git commit |
| 程序记忆 | `how-tos/` | markdown + `how-tos recall` Agent |
| 语义记忆 | `.discuss/` | 碰撞痕迹 + 决策理由 |
| 工作记忆 | `notebook_channel` | 可丢弃文件系统 scratch |
| 能力记忆 | `module_channel` | 反射 Python module |
| 时间索引 | git commit 历史 | 所有上述内容的线性时间线 |

### 已有基建可直接复用

- **`how-tos recall`** (`src/ghoshell_moss/core/resources/markdown_kb/_agent.py`):
  内联 pydantic-ai Agent。扫描所有 markdown 的 YAML frontmatter → 把索引 dump 进 prompt → LLM 选出相关文档。
  不做 vector embedding，不做 keyword search。纯粹的 "LLM 读目录，选文件"。
  这是 **read** 模式的原型。

- **`moss features`** (`src/ghoshell_moss/core/codex/_features.py`):
  文件系统即数据库。FEATURE.md 承载状态，date path 编码时间，默认只扫最近两个月。
  这是 **深度控制** 的原型：增量优先，全量按需。

- **`notebook_channel`** (`src/ghoshell_moss/channels/notebook_channel.py`):
  write/read/append/list/delete + context_messages 展示目录树 + 路径安全。
  这是 **write** 操作的原型。

- **零依赖 Channel 超市设计** (`src/ghoshell_moss/channels/.design/2026-05-14-zero-dependency-channel-supermarket.md`):
  描绘了演进路径：反射 → 临时笔记 → 目录认知 → 审计包装 → 组合。
  memory_channel 是目录认知 + 组合的自然延伸。

### 旁路 Agent 的通讯协议

用户提出旁路 curation agent 复用主路的思维架构，差别只在 shell 不同。
当前基础设施已经提供了文件系统作为 agent 间最干净的通讯协议：
- Curation agent 的输出写入记忆目录
- 主 agent 通过 `context_messages` / `memory_messages` 读取
- 不需要额外的进程间通讯或 MCP 调用

## 设计方向

### 四种记忆操作模式

1. **pinned** — 每轮注入。对应 `memory` slot。治理规则：pin 的内容在 memory slot 出现，Shell 层从 historical messages 中去重。树形索引控制深度（约定深度或文件数上限）。

2. **read** — 按需拉入上下文。对应 `how-tos recall` 的模式：目录树索引 → LLM 选择文件 → 读入。随对话自然淘汰。

3. **curation** — 旁路 agent 周期性运行。触发条件应基于 stale detection（记忆目录下文件变更累积到阈值，或引脚状态长期未更新），而非每次主交互后自动触发。负责索引选枝、pin 文档、标记 stale。

4. **full recall** — 旁路 agent 针对特定领域做完整索引 → 读取 → 压缩 → 写摘要。

### 双模态设计

memory_channel 应同时是 Channel（CTML 交互中使用）和 CLI 命令（旁路 Agent 中使用）。这与 `module_channel` 不同——后者只是反射，memory_channel 是主动管理。

### 责任边界

- **Channel 层**：产出 `memory: list[Message]`。管理文件的 CRUD、pin 状态、目录树展示。
- **Shell 层**：负责把 `memory`、`context`、`history` 拼在一起时的去重和 token 预算治理。
- **旁路 Agent**：复用主路的思维架构，通过文件系统读写记忆目录，不直接与主 Agent 通讯。

### 最小可行版本

只做 pinned + read + write。旁路 curation agent 先由人类手动触发（在 Claude Code session 中让 AI 去整理记忆），模式跑通后再自动化。

## 自反身验证

memory_channel 的第一个测试：它能不能管理它自己的 development feature。
FEATURE.md 由 AI 自己创建、自己更新、自己 pin、自己 curation——这是 Ghost 反身性的最小闭环验证。

## 开放问题

1. `Builder.memory_messages()` 装饰器的签名是否直接复用 `context_messages` 的 `MessageFunction` 模式，还是需要不同的语义（比如支持 pin 和 unpin 状态标记）？
2. pinned 的去重逻辑在 Shell 层实现——但当前 Shell 是否有足够的上下文拓扑信息来做去重？需要检查 `ModelContext` 的组装逻辑。
3. 树形索引的深度控制策略：是全局约定（最多 2 层），还是每个 pinned 条目标记自己的展开深度？
4. curation agent 的 stale detection 阈值如何定义？mtime 差、git diff 行数、还是 pinned 状态的更新频率？
5. 多 Ghost 共享同一个记忆目录时，写冲突如何解决？当前 `notebook_channel` 不做并发控制。

## 不做的事

- 不引入 vector embedding 或外部向量数据库。文件系统 + 树形索引 + LLM 读目录已足够 bootstrap。
- 不做复杂的通知/同步机制。当前阶段拉模式足够。
- 不替代 `features` 和 `how-tos` 体系。memory_channel 是通用记忆原语，`features`/`how-tos` 是基于这套原语的具体应用模式。

## 需要到位的资源

1. `Builder.memory_messages()` — 在 channel_builder.py 中补上对应 context_messages 的装饰器
2. Shell 层 context 组装逻辑中确认 memory slot 的处理方式
3. 测试载体：一个足够复杂的模拟项目目录，包含多层级记忆文件

## 相关文件

- `src/ghoshell_moss/core/concepts/channel.py:ChannelMeta` — memory slot 定义
- `src/ghoshell_moss/core/blueprint/channel_builder.py` — Builder 接口（缺少 memory_messages 装饰器）
- `src/ghoshell_moss/channels/notebook_channel.py` — 文件操作原型
- `src/ghoshell_moss/channels/module_channel.py` — 反射模式原型
- `src/ghoshell_moss/channels/.design/2026-05-14-zero-dependency-channel-supermarket.md` — Channel 超市演进路径
- `src/ghoshell_moss/core/resources/markdown_kb/_agent.py` — how-tos recall Agent 实现
- `src/ghoshell_moss/core/codex/_features.py` — features 文件系统追踪实现
- `tests/ghoshell_moss/channels/test_notebook_channel.py` — notebook 测试
- `tests/ghoshell_moss/channels/test_module_channel.py` — module 测试
- `.ai_partners/features/.discuss/features_system_validation_and_ai_first_developer.summary.md` — 文件系统约定的深度讨论

## 勘探参与

- 项目作者 (thirdgerb) — 引导方向、铺设信息路径
- DeepSeek V4 via Claude Code — 勘探执行、综合归纳

2026-05-14
