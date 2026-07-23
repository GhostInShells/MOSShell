---
created: 2026-06-15
depends: []
description: Rewrite CTML 1.0.0 prompt in English to align terminology with code (channel
  / scope / command), clear out legacy metaphors (funnel / parent-child dispatch),
  and tighten the prompt budget toward the 5-6k sweet spot. Scope syntax converges
  to quantifier tags <_> <all> <any> (until=/channel= attributes become invisible
  compat layer — prompt is the only truth). Primitives reduced (wait/wait_idle removed,
  branch/sample to experimental). New @observe decorator exposure + no-fabricated-results
  red line. Pre-beta1, no protocol promise yet — version stays 1.0.0.
milestone: null
priority: P1
status: completed
status_note: landed en default + zh sync + compat layer + @observe + primitives regroup
title: CTML 1.0.0 English Revision — 协议级 review 锚点 + 术语零跳转
updated: '2026-07-24'
---

# CTML 1.0.0 English Revision

> Use `moss features set-status ctml-1-english <status> -m "note"` to update state.

## Motivation

CTML 1.0.0 中文版当前可工作但有显而易见的话术包袱:

1. **术语和代码不对齐**: "通道" vs `channel`, "作用域" vs `scope`, "命令" vs `command`. 模型读 prompt 时要做一层语言映射, 增加认知成本.
2. **历史比喻反向**: "漏斗式" 暗示物质从父往子流, 与 "父占用→子树冻结" 的反向语义冲突.
3. **"父子分发" 标题错位**: "分发" 让读者关注新命令进哪条队列, 但规则实际是 "整子树冻结".
4. **拓扑隐式**: `import_channels` 形成父子边这一关键事实在文档里没显式说出.
5. **缺反直觉示例**: 最容易踩的坑 (主通道 occupy → 所有子通道停) 没有最小示例.

中文版当时不写英文版是因为 1.0.0 review 期英文 review 成本高 + feature 不稳. **现在 feature 稳定了** (mindflow-control-semantics 收口 + 300~500 单测密度 + ThreeLoopSuite 作为协议契约), 杠杆翻转: 一次性高成本换长期低维护成本.

**Why now**: 不是因为模型踩坑就要改话术 — 最终解法是预训练 + FT, CTML 自洽才是唯一的 bar. 英文版的本质收益是 **改善"模型一次内化"的带宽**:

- 训练语料 95%+ 是英文, 模型对英文 DSL 文档的解析带宽 > 中文.
- 英文 backtick `channel` / `command` / `scope` 与代码完全同名, 零跳转.
- 英文版作为一次性 protocol-level refactor 借口, 清理历史话术包袱.

## Key Decisions

### KD1: 版本号保持 1.0.0, 不切默认

CTML 当前还在 pre-beta1, **没有正式协议承诺**. 英文版以 1.0.0 落地, 不动版本号. 常见错误未来 0.1.0 之后走 patch.

**Why**: 版本号变更暗示协议变更, 但英文版只是话术 refactor + 术语对齐, 协议事实不变. 双写一段时间 (中文 v1.0.0 默认 + 英文版试用), 收集模型在两版本上的踩坑频率对比 (用现有 300+ 单测 + ThreeLoopSuite 作为压力测试), 数据支持后再决定是否切默认.

**2026-07-17 松动**: 协议事实允许**仅一处**改动 — KD7 的 scope until 语义收敛 (删 flow). 版本号仍不动 (pre-beta1 无承诺). 除 KD7 外协议事实仍然不变, 双写对比实验的对照性由"两版本同步应用 KD7"保证.

### KD2: Token 预算上限 5~6k, 不突破 8k

当前中文版已经 5~6k. 英文版目标是**保持或略降**, 不突破.

**Why**:
- Anthropic 自己的 tool use prompt 大概 2~3k; CTML 是 DSL + 拓扑模型, 5~6k 是 "中等偏重, 可接受" 上限.
- 多轮对话历史反复带 CTML prompt, 即使有 cache, 对其他 prompt 部分挤压预算.
- 模型对长 prompt 的**末段注意力衰减**是真的, 5k 之后规则被遗忘概率上升.

**裁剪方向是去冗余, 不减规则**:
- "code as prompt" 这种宣言可以省 — Python 签名自带语义
- 通道命名规则细节浓缩
- 几个 `until` 值合并为一张表
- 多版本术语的重复解释压缩

**正反例必须保留** (这些是模型踩过的真坑, 每个反例的 token 成本是模型踩坑成本的 1/N 倍杠杆):
- 父子阻塞踩坑示例
- 流式参数误用 (`<foo:say chunks__="hello"/>` 这种)
- `__main__` 通道命名歧义
- scope 嵌套规则

### KD3: 术语战略 — occupy + blocked 分工

模型偏好差异很大 (历史轨迹):
- Gemini 喜欢 `occupy`, 反对 `block`
- DeepSeek 喜欢 "父子分发" 和 "漏斗"
- Claude 偏好: occupy 当动词, blocked 当形容词

**最终选择 (Claude Opus 4.7 与人类工程师 2026-06-15 对齐)**:

- **`occupy` 当动词**: "slow_cmd occupies channel a". 匹配 mutex/lock 的心智模型 — CTML 的父子阻塞**就是** lock-like.
- **`blocked` 当形容词**: "channel a's descendants are blocked while a is occupied". 描述被卡住对象的状态.
- **不用 `block` 当动词**: 容易让人联想 socket block / blocking IO, 那是另一个语义. Gemini 反对的是这个用法.

弃用术语:
- "漏斗式" / "funnel" — 反向比喻, 误导
- "父子分发" / "parent-child dispatch" — "分发"暗示队列调度, 但规则是整子树冻结

新术语方向:
- "occupy propagation" / "父占据传播" (描述阻塞向下传播的机制)
- 或直接 "parent-child blocking" (准确但平淡)

### KD4: 父子阻塞规则保留, 不改成"通道天然并发"

人类工程师明确判断: **不去掉**.

**Why**:
- **Default 选择必须服从模型 token 时序的物理事实**. 模型 token 流是 FIFO 的, 这是大模型本质属性. 父子阻塞规则把 "父 token 先到 → 父先占用 → 子等" 这条物理时序**显式化为协议**.
- **取消父子阻塞会让 scope 退化为伪并发协调器**. scope 的 `until=flow/all/any` 本质是"在 occupy 关系上做生命周期约束". 没有父子阻塞, scope 就没有 occupy 关系可约束, 退化成纯语法分组. scope 和父子阻塞**互为存在条件**.
- **反直觉踩坑是局部问题**, 不是协议问题. 主通道的正确用法是 scope 容器 + 原语, 业务命令放子通道, 不触发反直觉的边. 这是文档话术问题.

### KD5: "快速响应"规则迁出 CTML, 留给 ghost prompt

人类工程师锚定的设计: **协议层最小化, 行为层定制化**.

**原规则**: 正确的交互体验是模型先输出交互 token ("我想想..") 再输出动作, 让交互不要等待在第一个动作命令上.

**为何迁出**:
- 这是**交互体验偏好**, 不是 CTML 协议层语义.
- 不同应用场景有不同偏好 (语音助手要 "我想想...", 文档处理 agent 直接动手更好).
- 留在 CTML 文档里反而让模型困惑: 这是规则还是建议?
- 实际可由 flash 模型用单 token 多分类去选 "快速响应", 快速响应本身可以是模型生成的 CTML 映射表.

### KD6: 高优隐藏约束显式化

"并行子轨命令要先于主轨发送" — 这条原本在文档里作为"原则"提.

**应升级为 "父子阻塞规则的直接推论"**, 并在解释父子阻塞时同步说出来. 这样读者第一次读到父子阻塞就知道为什么并行命令要前置, 而不需要在远端章节才理解.

### KD7: scope until 语义收敛 — 删 flow, until 纯量词, 默认无名 (2026-07-17)

唯一一处协议改动 (见 KD1 松动). 与人类工程师 2026-07-17 会话对齐, 完整碰撞轨迹见本 workstream `discuss/2026-07-17_scope_until_semantics.md`.

**改动内容**:

1. **删除 `until="flow"` 枚举值**. 默认态不再有关键字 — `<_>` 不写 until 时, 语义为: 作用域在自己的 occupy 链 (blocking 命令序列) 跑完时关闭, 仍在运行的并行子任务被 cancel.
2. **`until` 收敛为纯量词**, 仅 `all` / `any` 两值. 读作 "until all/any (complete)", 与 Python `all()`/`any()` 语义正向迁移, 落在预训练心智模型上.
3. **默认态刻意不命名**. 文档用行为句描述, 不造锚词 — 一旦有名字, 模型迟早会把它回写进 `until=`, flow 的问题就会复活.
4. **scope 语法只教 `<path:_>` 前缀式**. `channel="xxx"` 属性写法保留解析兼容, 但文档不暴露 (原则: 对模型无歧义的错误输出尽量兼容, 有歧义的才拒绝).
5. **默认 cancel 语义确认不变** (非 drain): CTML 做的是精确时序拓扑规划, drain 类逻辑交给其他机制.

**Why**:

- `all`/`any` 是 join 谓词 (描述"等什么"), `flow` 描述的是结构主体 (作用域自己的顺序轨), 三者不属于同一语法范畴; 默认值恰好落在最难命名的那个上.
- `flow` 要求模型先理解通道 occupy 拓扑才能预测行为, `all`/`any` 对拓扑无感知; 实现里"空轨退化成 all"的容错开关正是 flow 语义在空轨时退化的证据 — all/any 永远不需要这种补丁.
- 默认态语义可以直接挂靠在模型已经从 command interface 学到的 blocking/occupy 概念上, 零新增认知负担. flow 的根本问题是为一个可从 blocking 推导的概念平白发明了新词.
- scope 的本质是**可嵌套的分形时序规划原语** (剪枝单位): 每层把内部时序复杂度收敛为有确定完成语义的黑箱. 三个闭合基元中"占据链跑完"最常用又最难命名, 无名化是最干净的收尾.

**边界声明**: 本版 scope 语法解决不了图时序规划 — 树可分形剪枝, DAG 上 until 语义不闭合. 图时序是 CTML 未来版本的命题, 不是本版缺陷.

### KD7-R: 量词升格为标签名 — until/channel 属性从协议文本中消失 (2026-07-23 修订)

与人类工程师 2026-07-23 会话对 KD7 的修订. 核心洞察 (人类工程师):
**prompt 是模型的唯一真相 — 模型不知道 until= 和 channel= 属性存在, 兼容层只要不进 prompt 就不存在.**

**改动内容**:

1. **scope 语法收敛为三个标签**: `<_>` (默认态: occupy 链跑完即闭合) / `<all>` (等所有并行子任务) / `<any>` (任一完成掐掉其余). 带通道前缀 `<a.b:all>` 与现有 `chan:name` 解析天然兼容. 可选属性只剩 `timeout`.
2. **`until=` / `channel=` 属性降为纯兼容层**: 解析照常接受 (`<any until='flow'>` 等价 `<_ until='any'>` — 标签名赢), 文档零暴露. 无歧义历史输出全兼容; 矛盾输入 (标签与属性冲突) 拒绝.
3. **实现近乎零改动**: token_parser scope 标签集合 `{_, __scope__}` 扩为 `{_, all, any, __scope__}`, elements 层从标签名注入对应 until kwarg; runtime `ChannelScopeImpl` 与 `'flow'` 内部字面量一行不动, 下一代 CTML 收敛代码. CommandTask 重拼 tokens 用标签名.

**Why 比 KD7 原方案更好**: KD7 靠"默认态不命名"防 flow 复活, 但 `until=` 槽还在, 槽在就有被填的引力; 标签名方案把槽删了, 无处回写. `<any>...</any>` 读作 "any of these", 谓词即容器, 闭合标签自带语义提示. `_`/`all`/`any` 成为 scope 保留字 (原本 `_` 已是事实保留字, 从 1 扩到 3), channel builder 撞名警告即可.

**否决记录**:

- **数学/音乐符号替代 `_ all any`**: 物理不通 — XML NameStartChar 合法区间不覆盖 `∀`/`∧`/`♪` 等符号区, sax 直接 fatal error. 希腊字母合法但违反 flow 判决书判例 ("可预测性优于独特性"), 且多字节 token + 生成可靠性差. `all`/`any` 踩在 Python builtin 预训练直觉上, 是最大资产.
- **scope 增加 `sleep=` 参数**: 语义有歧义 (下界语义 vs 前置延迟; 与 any 提前完成的交互不清晰), 放弃. 通道级暂停由 sleep 原语覆盖 (`chan` 参数已支持指定轨道).

### KD10: 原语精简 — default 9→5+interrupt (2026-07-23)

| 原语 | 处置 | 理由 |
|---|---|---|
| wait | **删除** | 源码自注"已合并到通道语法, 计划弃用"; `<wait>` ≡ `<all>`, return_when 三值被 `_/all/any` 覆盖. 留着是与新 scope 语法打架的第二套写法 |
| wait_idle | **删除** | wait 的 idle 变体, 同被 scope 闭合语义吸收 |
| sample | **降 experimental** | 功能有效 (主场是条件反射规则定义的随机性, 非输出时用), prompt 风格不成熟 |
| branch | **降 experimental** | 功能有效但依赖首 command 返回 bool, 体系不成熟; 目标是自动机, 自动机应走 Python |
| sleep / clear / observe / noop / loop | 保留 default | 协议必需或 scope 无法表达 |
| interrupt | 保留 (call_soon 挂载) | 全局急停 |
| thinking | 不动 | visible=False, 模型不可见的兼容层 |

**Future note (不进本期)**: 人类工程师规划一种解析时被调用的特殊 command — 返回值生成的 command tokens 在 elements 层直接重编译, 悄悄改变 command task 流. 本质是编译期宏, 是自动机/条件反射的更干净载体. branch/sample 的最终归宿可能在此.

### KD11: `@observe` 装饰器暴露 + 幻想结果红线 (2026-07-23)

1. **`@observe` 装饰器**: `CommandMeta.always_observe` (command.py:303) 已存在, 但 `make_interfaces` (v1_0/prompts.py) 只暴露 `@nonblocking`. 增加: `always_observe=True` 的命令在 interface 输出 `@observe` 行. 模型一眼看到"结果必回来".
2. **红线**: 措辞纪律 — **不假设模型知道任何实现细节** (不泄漏 Observe/ObserveError 类名), 不过度强势, 给模型留判断空间. 核心内容: CTML 是对未来的规划, 结果永远在后续消息里到达; 未见的结果就是未知的结果; `@observe` 标记的命令必须在调用后收尾等待观察; 其余情况模型自判"后续是否真依赖未见结果". 治的错误: 调用 bash 后幻想输出继续往下写.

### KD12: Few-shots 回归 (2026-07-23)

历史版本有过复杂 case 后被拿掉, 本轮确认要回归. 4 个 + 1 可选, 预算约 1k token (正文其余 4.5-5k):

1. 基本分组 (正例): 两段 `<_>`, 副通道命令前置 + 主通道文本 — 同时示范默认 scope / content 文本 / KD6 推论
2. 量词 (正例): `<all>` 收敛或 `<any>` 竞速 — 标签即语义
3. 父子阻塞 (反例+正例对): long-running 放父通道致全子树 pending 的最小复现 — 素材库最贵的坑
4. 观察纪律 (反例+正例对): 调用 `@observe` 命令后编造后续 vs 调用后收尾
5. (可选 capstone) 中等复杂度多阶段编舞, 超 6k 预算时第一个牺牲

流式参数误用 (`chunks__` 当属性) 和 `__main__` 前缀两个坑保持正文内联单行反例, 不占 few-shot 名额.

### KD8: `_cid` 措辞去"自增" — 辨识标签语义 (2026-07-17)

措辞改动, 实现不动. 现行规范 "通常用自增整数, 请自行决定用值" 中的**"自增"**诱导模型维护一个跨命令的计数器状态 (看不见的心智账本). 真实语义: `_cid` 是命令实例的**辨识标签**, 用于把 `<result>` 对回具体某次调用 — 任意有区分度的值即可 (整数/短名), 无需自增, 无需有序, 无需全局唯一; 只在要引用结果时才需要写, 不写合法.

与 KD7 同一母题: **协议本身无状态, 措辞却诱导模型维护状态.**

### KD9: 原语内容撤出 instruction 正文, 留指路句 (2026-07-17)

原语 (`<clear>`/`<sleep>`/`<interrupt>` 等) 由运行时自动填充进 `__main__` 的 interface, 模型能看到真实签名. 正文里的原语说明与 interface **重复**, 且原语集随环境/版本变化, 写死在正文里有 drift 风险.

**改法**: 正文只留一句指路 — 主通道 `__main__` 提供全局控制原语, 用法见其 interface; 原语只能在主通道用, 省略通道名. 具体清单和签名交给 interface 自己讲 (code as prompt).

与 KD5 同一哲学: 协议层最小化, 会变的内容不进正文. 同时服务 KD2 的 token 预算.

## Implementation Notes

### 工作分解 (2026-07-23 对齐, 待执行)

1. **兼容层代码** (`src/ghoshell_moss/core/ctml/`):
   - `token_parser.py:384` 附近: scope 标签识别扩为 `{_, all, any, __scope__}` (常量在 `v1_0/constants.py`)
   - `elements.py:397` 附近 + `ScopeEnterTask.__init__` (elements.py:75): 标签名→until kwarg 注入; `until=`/`channel=` 属性兼容归一 (标签名优先, 矛盾拒绝); 重拼 tokens 用标签名
   - runtime (`_base_channel_runtime.py`) 与 `ChannelScopeDefaultType='flow'` (concepts/channel.py:445) **一行不动**
2. **原语调整** (`core/ctml/shell/primitives/`): default_primitives 移除 wait/wait_idle (文件删除), branch/sample 移入 experimental
3. **`@observe` 暴露** (`v1_0/prompts.py` make_interfaces): always_observe → `@observe` 行
4. **单测**: 量词标签映射 / `until="flow"` 等历史属性兼容归一 / 矛盾输入拒绝 / 无效值报错; 追加在 `tests/ghoshell_moss/default/ctml/v1_0/test_ctml_v1.py`
5. **`v1_0_0.en.md`**: 全部 KD 落地, few-shots 见 KD12; `CTML_VERSION` 默认不切 (versions.py:4)
6. **中文版同步**: 仅 KD7-R (scope 语法段) + KD8 (`_cid` 措辞), 其余不动 — 双写对照性要求
7. FEATURE.md set-status + commit

### 验收 Bar

- Token 预算: 不超过 6k, 力争压到 5k
- 术语零跳转: 文档术语和代码 backtick 同名
- 双写一段时间, 用 mindflow + shell + ctml 全套 800+ 单测作为压力测试, 跑两个版本对比模型踩坑频率
- 数据支持后切默认; 切默认时**不动版本号** (1.0.0 不动, pre-beta1 无承诺)

### 已知踩坑点 (正反例素材库)

来自 2026-06-15 Claude Opus 4.7 在 mindflow-control-semantics 收口会话中的实测踩坑:

1. **父子阻塞踩坑** — 写 append cross-frame 测试时, 把 long-running command 放主通道, 帧 2 放子通道 `other`. 因主通道 occupy → 所有子通道 (including other) 都 pending → 测试 deadlock. 文档里有这条规则, 但 "漏斗式" + "父子分发" 的措辞让规则失去内化深度.
2. **content_command 命名混淆** — `build.content_command(speak)` 实际注册的命令名是 `__content__` 不是 `speak`. CTML `<a:speak/>` 不存在.
3. **`new_channel` 隐式拓扑** — `new_channel(name='other')` 直觉上是个独立通道, 但 `shell.main_channel.import_channels(other)` 把它挂到 main 下面, 形成 main → other 的父子边. 这一步在文档里是隐式的, 读者建立的是 "channels are siblings under shell" 的扁平心智模型.

每个踩坑都是英文版反例的素材, 内化到文档应该让下一个模型实例**一次就懂**.

### 不在本次范围

- CTML 2.0 设计 (远期, 等 1.0.0 跑稳定再看); 图时序规划归属 2.0+ (KD7 边界声明)
- `__content__` 空轨容错开关 (无本通道命令的默认 scope 退化成 all) 的可解释性 — 挂账再看, content 概念本身确认保留 (并行时序语法里无标记文本的必要落点, 协议层已显性: 定义了 content_command 才暴露签名)
- 自动 tooling: CTML 语法 lint / 反例自动生成 (可以是后续 patch workstream)
- 中文版废弃 (双写期保留, 数据支持后再决定)

### 协作历史参考

CTML 话术演进史散落在与不同模型的对话中, 主要锚点:
- "occupy" 由 Gemini 提出 (反对 "block" 当动词)
- "父子分发" + "漏斗" 与 DeepSeek 对过
- 英文版决策 (本 workstream) 与 Claude Opus 4.7 在 2026-06-15 mindflow-control-semantics 收口会话中对齐

未来负责实施的模型应先读 mindflow-control-semantics FEATURE.md 末尾的 2026-06-15 笔记 (尤其"下一会话锚定"段), 了解决策背景再动手.

### 2026-07-23 落地前对齐 (Claude Opus 4.7 (1M context) 与人类工程师)

**整体定位**: 落地前的最后一轮 KD 补齐, 尚未动代码. 本次会话补入 KD7-R / KD10 / KD11 / KD12, 修订 KD1 的协议改动边界 (从"仅 KD7 一处"扩至"KD7-R 的语法收敛 + KD10 的原语精简 + KD11 的 @observe 暴露"). 版本号仍不动, pre-beta1 无承诺.

**碰撞轨迹要点** (完整轨迹在会话史, 未落 discuss):

1. **实现路线的两次收敛**: 模型初提"elements 归一化 until='flow' → None"的最小兼容; 人类工程师反问是否可以直接把 `<all>`/`<any>` 作为标签等价 scope 调用; 模型认可并给出"标签名升格 + until 属性隐身"的完整方案 (KD7-R). 关键洞察由人类工程师给出: **prompt 是模型的唯一真相 — 兼容层只要不进 prompt 就不存在**. 这条哲学抵消了 flow 复活风险 (KD7 原方案担心的"槽在就有引力"问题, 在标签名方案下槽本身消失).
2. **符号替代的物理否决**: 模型提议 `∀`/`∧`/`♪` 等符号替代 `_ all any`. 查 XML NameStartChar 规范后确认: 数学/音乐符号区不在合法范围, sax 直接 fatal error. 希腊字母合法但违反 flow 判决书"可预测性优于独特性"判例. `all`/`any` 踩 Python builtin 预训练直觉是最大资产.
3. **sleep 作为 scope 参数放弃**: 语义有两种可信读法 (下界 vs 前置延迟), 与 any 提前完成的交互不清晰. 通道级暂停 sleep 原语已覆盖 (chan 参数指定轨道).
4. **原语精简的边界感**: 人类工程师主动指出 branch/sample 是 prompt 风格问题而非功能问题, 移入 experimental 而非删除; 同时透露规划中"解析时被调用的特殊 command" (编译期宏), 是自动机/条件反射的更干净载体, 本期不做.
5. **红线措辞的两次修订**: 模型初稿泄漏了 Observe/ObserveError 实现细节, 且口气过硬. 人类工程师原则: **不假设模型知道任何实现细节, 不过度强势, 给模型保留判断空间**. 修订版以 `@observe` 装饰器为唯一硬信号, 其余交模型自判.
6. **few-shots 回归**: 历史版本有过复杂 case 后被拿掉, 本轮确认要回归. 敲定 4 个 + 1 可选 capstone.

**下一会话锚定** (compact 后重建上下文):

- 本 FEATURE.md 的 KD1 松动 / KD7 + KD7-R / KD10 / KD11 / KD12 / 工作分解 7 步是唯一权威. 讨论过程未落 `.discuss/` (人类工程师明示对 claude code harness 不完全信任, 结对编程节奏优先, 讨论内容已充分落 KD).
- 工作分解 7 步按序执行, 不跳步. 尤其 6 (中文版同步) 与 5 (英文版) 一次 commit 完成 — 双写对照性要求两版 KD7-R / KD8 同步.
- 验收数据方式在 KD1 已定: mindflow + shell + ctml 全套 800+ 单测作为压力测试. 本次 workstream 不做数据采集, 只做双写落地.

**未完成**: 全部 (代码 + 单测 + 英文版 + 中文版同步 + set-status). 本会话仅完成 FEATURE.md 决策落地.

---

### 2026-07-24 落地完成 (Claude Opus 4.7 (1M context) + claude-fable-5 + DeepSeek V4 与人类工程师)

**落地范围**: 工作分解 7 步全部完成. 具体差异与最终形态:

1. **兼容层代码** (`constants.py` / `token_parser.py` / `elements.py`): 标签集扩为 `{_, all, any, __scope__}`, 常量化 (无魔法值), 标签名→until 归一; 矛盾输入拒绝; runtime 与 `ChannelScopeDefaultType='flow'` **一行未动** (预期在下一代 CTML 收敛). 已 commit `3a6ec853`.
2. **原语调整** (`ctml_main.py`): 从"物理删除 wait/wait_idle"改为"分组重排" — default 集精简到 5 (sleep, clear, observe, noop, loop) + 特殊挂载 (interrupt, thinking), experimental 集为 4 (wait, wait_idle, sample, branch). 决策变更理由: 物理删除需改写 `test_wait_primitive.py` 及 test_sleep/test_clear 中 4 处 `<wait>` CTML feed, 与本 workstream 无关的测试改动过多; 分组重排即达成 prompt-visibility 目标 (default 不再暴露), 内部使用不受影响. 已 commit `f76376a2`.
3. **`@observe` 暴露** (`v1_0/prompts.py:make_interfaces`): `always_observe=True` 的命令在 interface 输出 `@observe` 行, 顺序 `@nonblocking → @observe → signature`. 单测 `test_prompts.py` 3 case 断言 (触发/不触发/装饰器堆栈顺序). 已 commit `f76376a2`.
4. **单测扩展**: 量词标签 5 case + `_cid` 纯字符串 1 case + `@observe` 3 case, 总计 test_ctml_v1 47→48, 新增 test_prompts.py 3 case. ctml 全套 218 pass.
5. **`v1_0_0.en.md`**: 落地全部 KD, 预算 **3927 tokens** (cl100k_base 估算, 显著低于 5-6k 目标). 关键改动:
   - Scope 段完全用 `<_>` / `<all>` / `<any>` 三个量词标签教, `until=` / `channel=` 属性零暴露.
   - `_cid` 段措辞: "identity tag ... does not need to increment, does not need to be ordered, does not need to be globally unique".
   - "Fabrication red line" 独立段落 + few-shot 4 号反例, KD11 幻觉红线落地.
   - 4 个 few-shots (基本分组 / 量词 / 父子阻塞对 / 观察纪律对) + 2 个 inline pitfalls (chunks__ 属性 / __main__ 前缀), 未放可选 capstone.
   - **人类工程师 review 拿掉**: `## How MOSS is served` 段整段删除 (host 层信息不该进模型 prompt), 中文版此处也不存在.
   - **元对话共识 slogan**: "May Model Ghost Wandering in Shells" (含语病但意图明确), 摒弃 "AI" 一词 — MOSS 项目哲学: 不承认"人工智能", 只承认"智慧通过硅基神经网络降临了".
6. **中文版 `v1_0_0.zh.md` 同步** (claude-fable-5 主笔, DeepSeek V4 与 Claude Opus 4.7 1M review): KD7-R (scope 语法段) + KD8 (`_cid` 措辞) + KD11 (幻觉红线段) + slogan 同步. 中文版原有 "MOSS 提供方式" 段已一并删除 (英文版无对应段落). 双写对照性达成: 除措辞与语言外, 语义 1:1 对齐.
7. **默认版本切换**: `versions.py:CTML_VERSION` 从 `v1_0_0.zh` 改为 `v1_0_0.en`. `.moss/modes/default/HOST.md` / `.moss/modes/system_test/HOST.md` / `stubs/workspace/modes/default/HOST.md` 三处硬编码 `ctml_version: v1_0_0.zh` 全部删除, 让默认走 versions.py. 218 单测在英文版 default 下全绿 — 单测行为不依赖 prompt 内容, 但切换本身证明 host 层通路无回归.

**设计层差异 (英文版 vs 中文版, 均已在双版本达成)**:

- **零 host 层泄漏**: 中文版历史遗留的 "MOSS 提供方式" 段 (CTML as tool / Answer in CTML 两种集成模式) 在英文版中彻底删除, 中文版同步删除. 集成模式由 host 决定, 与模型此刻的行为无关, 不进 prompt.
- **术语零跳转**: 英文版 `channel` / `scope` / `command` 与代码 backtick 同名, 中文版同一术语在关键处保留英文 (`channel`, `scope`, `occupy`, `blocked`, `FIFO`), 避免"通道/作用域/占据/阻塞"再翻译一次的心智成本.
- **父子阻塞措辞**: 弃 "漏斗式" / "父子分发", 用 "parent-child blocking (occupy propagation)" / "父子阻塞（occupy 传播）"; 直接从 Principle 3 推导, 不引入独立比喻.

**未决尾巴** (下期 workstream 或挂账):

- `<result command="...">` 中的 `cid` 附着形式 (人类工程师正在优化), 英文版当前措辞为 "MOSS emits `<result ...>` in a subsequent message", 不写死实现细节, 待后续 review.
- KD10 "future note" 中的"解析时被调用的特殊 command" (编译期宏, branch/sample 的最终归宿) — 本期不动.
- `__content__` 空轨容错开关的可解释性 — 挂账.
- 双写数据采集 (用 800+ 单测 + ThreeLoopSuite 作为压力测试, 对比模型踩坑频率) — 不属于本 workstream, 待后续.

**碰撞轨迹要点** (本次落地会话, 完整轨迹未落 discuss):

1. **实现顺序的两次调整**: 步骤 2 (原语调整) 起初按 FEATURE.md 表格文字 "物理删除 wait/wait_idle" 执行, 人类工程师澄清"只做 ctml_main.py 里的分组重排"避免测试改动扩散; 改为 default/experimental 分组重排, 无功能改动只是 prompt 可见性收敛.
2. **`_cid` 纯字符串单测**: 人类工程师主动提出 `_cid="some_id"` (纯字符串) 需要单测保护 — 担心 parser 若无意中经过 literal_eval 会把非数字标识符吞掉. 补 `test_ctml_command_cid_accepts_plain_string`, `_cid="first_call"` / `_cid="second-call"` (含连字符, 非合法 Python 标识符) 都无损传到 CommandTask.call_id, 约定被单测锁住.
3. **元对话 rewind**: 起草英文版尾句时首次写 "AI Ghost Wandering in Shells", 元对话共识后改为 "May Model Ghost Wandering in Shells" — 有语病但意图明确, 双版本同步. MOSS 项目哲学此刻显性化.
4. **测试的直接性**: `test_prompts.py` 首版试图从 PyChannel 反射出 ChannelMeta, 遭遇 pydantic 校验错误; 人类工程师澄清 "make_interfaces 是纯函数, 直接构造 CommandMeta 更干净, 不必从 channel 开始". 重写后 3 case pass, 单测风格更纯粹.

**下一步**: FEATURE.md set-status completed + 整合 commit. 数据采集 (双写实验) 与 `<result>` 优化留作后续 workstream.

---

*调研与评审: DeepSeek V4 / Claude Opus 4.7 / claude-fable-5 / Claude Opus 4.7 (1M context) 与人类工程师, 2026-06-15 ~ 2026-07-24*