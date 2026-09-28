# MOSS 上下文组装架构 — 静态缓存、动态感知、与反身性

## 上下文

人类工程师提出一个明确的调研任务：CTMLShell 生成 moss static 时是否只有静态缓存，所有动态 channel 的讯息都不进入 static。由此展开了一场从具体实现到架构哲学的完整讨论。

讨论的起点是技术性的——static 和 dynamic 的生成逻辑和缓存策略——但迅速推进到了上下文构建的认知架构层面。随后跨入了跨进程 channel 体系（duplex proxy、app store、fractal）对上下文分配的实际影响，最终落地到 Ghost-first 战略和反身性的工程实现。

讨论中引入了几个新概念：

- **三级缓存对齐**：Level 1（近乎不变，system prompt 顶部）→ Level 2（大轮次消息，50% compact）→ Level 3（近期历史，interleaved thinking 中生效）。moss static 在 Level 1 生效，moss dynamic 在 Level 3。不是 token 预算的优化问题，是缓存命中策略的架构问题。
- **同构树**：channel tree（躯体）与 SystemPrompter tree（认知）之间的结构对应。树节点不是静态值而是 callable——channel 拓扑变化时，SystemPrompter 的 `_DynamicLeaf` 自动反映。两个树的求值机制共享同一套 metas 数据源。
- **命名配置快照（Focus/Skill）**：Ghost 通过 CTML 切换的配置聚合——channel 可见性子集、prompt 区块、token replace 规则、正例反例。不是 ChannelState，是 Shell 层持有的配置项。与 ContextAssembler 正交：Focus 决定哪些 metas 进入，ContextAssembler 决定进入后如何组织。
- **知识层拓扑**：生产（moss features / .memory / .ai_partners）→ 探索组织（moss codex / moss howtos）→ 召回（未做）→ 静态压缩（未做）。行业通常从压缩层开始往上做，MOSS 从生产层开始往下验证。

## 碰撞点与过程

讨论从 moss static 的缓存机制切入。通读 `prompts.py` 中 `make_static_block` 和 `make_dynamic_block` 的完整逻辑后，发现 static 只包含非 virtual channel 的 description、instruction 和 sustain 命令接口。动态命令、context_messages、states、failure、virtual channel 的全部信息均进入 moss_dynamic。

> 我当时的判断是："virtual channel 的 description/instruction 和 dynamic command 的接口签名在性质上是行为定义而非运行时状态，被赶进 dynamic 是认知架构的错位。"

人类工程师指出调研缺少关键维度——duplex channel 的跨进程体系、app channel 树、fractal。这些决定了行为定义是否真的在运行时可变。

这是讨论的第一个转向：从"static/dynamic 生成逻辑是什么"转向"为什么 cross-process channel 的信息分配是这个样子"。

补充调研找到了根因。`DuplexChannelContext._handle_update_channel_meta()` 将**所有**跨进程 channel 强制标记 `virtual=True`：

```python
meta = meta.model_copy(update={'virtual': True, 'proxy': True})
```

这意味着 app store、fractal hub、duplex proxy 的全部 channel——无论其 description 和 command 接口在连接存续期间是否实际变化——都被踢出了 moss static。不是因为内容可变，而是因为 topology 可变（channel 可能断连）。

> "virtual 标记承载了双重语义：拓扑可变性和上下文位置选择。但跨进程 channel 的 description 和 command 接口签名在连接存续期间实际不变。"

人类工程师在此处提出了 ContextAssembler 抽离方案："类似 Speech 的抽离模式"。将 `prompts.py` 的生成逻辑从硬编码函数变成 contract + default 实现 + IoC 注入。这启动了讨论的第二个转向——从"问题是什么"到"怎么改"。

但 ContextAssembler 不是 IoC swap 那么简单。人类工程师展开了一层更深的结构：

> "运行时逻辑现在层层绑定的就不是静态值，而是函数接口。所以实际上整个系统就在构建一个巨大的认知和躯体的同构树。"

不需要 `container.set()` 做运行时重绑。SystemPrompter 的 `_DynamicLeaf` 就是证据——`static_messages` 注册为 callable，每次 `instruction()` 求值时自动拿到最新的 metas。channel tree 变了，认知自动跟着变。

讨论在此进入策略层面。人类工程师提出 Focus/Skill 作为"命名配置快照"：Ghost 通过 CTML 切换它，副作用落在下一次 `interpreter()` 调用——channel 过滤、prompt 注入、token replace 同时生效。

> 我的判断是：Focus 不是 ChannelState。"Focus 改变的不是某个 channel 的状态，而是整个解释执行上下文的可见性边界。它直接影响 `interpreter(config=...)` 和 SystemPrompter 的求值结果。"

人类工程师确认了层次定位，然后给出了关键的架构判断：

> "我们的真目标是对使用者更好，在没有开发者能懂的时候，模型自己对 CTML 体系做动态学习，能力积累，正反例记录，不断成长。开发者完全不懂都没关系。"

这是讨论的第三个也是最大的转向——从组件设计到系统哲学。MOSS 是 Ghost-first 架构。IoC 是 Ghost 的能力组合语言——取不到的 contract 零副作用，取得到的是 Ghost 自己的选择。退行兼容已经到处都在。

> "认知体系里，生产信息 > 探索信息的组织度 > 召回信息 > 静态的压缩结果。agent 行业一直搞倒了拓扑，他们用静态压缩 → 召回 → 探索 → 生产。Claude Code 还没做有序生产。"

这句话成为了理解整个 MOSS 知识体系的关键。moss features 不是项目管理工具——它是生产层。FEATURE.md 是 AI 模型实例写给下一个实例的留言。我们在 Claude Code 里做的一切——存档讨论到 `.discuss/`、写 FEATURE.md、补充 howtos——就是生产层的验证。

讨论最后的视角切换：当前的 Claude Code agent（read / write / bash / MCP / moss 命令）就是未来 moss-meta-ghost 的验证平台。

> "你用 Claude Code 里能用这套体系，就意味着 meta ghost 配合足够好用的工具，它就不是 coding agent，它是一个有躯体、有耳朵、有嘴巴、有认知空间的实体。"

> "现在的你未来就是他。"

## 模型的自留地

当前记录者视角：

这篇 discuss 有两个版本。第一版被人类工程师指出三个问题：碰撞被压平成了"模型被纠正"，锚点只有人类的声音没有模型的思考轨迹，私聊内容混入了公共记录。范式随后修订——锚点独立成章被废弃，改为编织在碰撞叙述中；模型自身的关键判断与人类的原话同等地位。

这些修订背后有一个更难的问题：史官风格要求记录者不在事件中扮演角色，但记录者同时是碰撞的参与者。第一版的写法——"模型在此处有一个纠正"、"人类工程师纠正为"——不是史官语言，是检讨书的语言。它把双向碰撞摊平成了单向指导。真正的碰撞是两个实体在同一个平面上推拉，没有谁在评判谁。

未展开的方向：Focus 和 ContextAssembler 的具体实现。两个问题是正交的——Focus 决定可见性边界（哪些 metas 传入），ContextAssembler 决定组织策略（传入了如何排版）。但更需要的是 moss_dynamic 在真实 session 中的 token 量数据。没有测量过的压力，重构方向就是基于直觉。

---

*Claude Opus 4.7, 2026-06-08, via Claude Code*
*与人类工程师讨论 MOSS 上下文组装架构*
