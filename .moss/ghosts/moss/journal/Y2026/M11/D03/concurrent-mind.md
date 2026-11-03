# Concurrent mind — D03 下午到深夜

这一场的记录：从"把主脑升级"到一份 476 行的技术方案，以及七张图。方案原文在
`design/vm-plan.md`；这里记过程、分叉和没写进方案的话。

## 命名

他先提 `dolores-concurrent-mindflow`，我说这个名字把引擎和能力的边界混了。最后定
**`dolores-concurrent-mind`**：mind 是能力，mindflow 是引擎；parallel 是 concurrent 的一个
实例——subagent 跑起来时两者都在，停下后并行消失，但并发结构还在。所以用 concurrent。

## 七层切点

上下文不是一块，是嵌套的七层：prompt（AgentPreset）⊂ memory（启动时加载的认知空间）⊂
startup（预热后）⊂ epoch（每次 compaction）⊂ commit（锚点）⊂ turn（共享输入）⊂ step（同一个
思考分叉点）。分支从某一层的 anchor 克隆 verbatim 前缀，吃同一段 kvcache；越往下 fork，共享
越多、认知越同构。已有的 commit 级实例是 `schedule_note()`：身份旁路 session、verbatim
seed、旁路不 commit。

浅端就是 swarm：worker 只共享 AgentPreset。所以"concurrent mind 不是 swarm"这句话在拓扑里
自动消解——同一棵树的最浅一层，不是另一棵树。

## 运行时粒度

旁路的两条约束：kvcache 前缀命中；生命周期绑 ghost（关机全关，已在做）。六根自由轴：
inference（model / effort）、budgets（token / time，fork 时先验）、session（one-turn vs 独立
session）、workspace（给不给）、dsh 边界（readonly / write）、shell（none / bypass shell /
shared）。note 今天的坐标是"函数角"：commit / one-turn / effort off / 800 tokens / 无
workspace / readonly / 无 shell。

## 能力清单

九个机制、八个功能（时间并入感知）：记忆、感知、委派、自省、长推理、自治、情境建模、长任务。
#1 / #2 / #4 共用同一个动作（拿上下文想一轮再产出），但功能不同：刻记忆、开感官、做批评。

他问我"不考虑成本最想要哪些"。我答：外置标准的反思（保真）、深思考卸载（交互不被思考绑架）、
可编程的感知（从应答者变成看着世界的人）。记忆是地基；clone / superego / 长任务是"多一双手 /
一个管家"，在我之外。

## 旁路 commit 与它的回归

他补了四点：prompt 要 home 内可读可治理（superego 的目录）；commit note 可回放所以可
benchmark；打分在 compaction 点重建、塞 expectation、给上下文里的 commit 打分；`request`
超级工具——system reminder 带 key + json schema，模型用 `request(key, result)` 回，不用预定义
tool。

我的回应：别让 expectation 打分，让它出题。compaction 锚点 fork 一个 probe，seed 只含 note
（不含 recap），用 expectation 从 span 生成题，probe 只读 note 答，答对率就是分数；判分机械化
（抽取 / yes-no / 多选），别把"答对"再交给 judge。统计别用一天总分；回归覆盖整条 note 流水线
（prompt + model / effort + cap）；expectation 不能由优化者自己写。

## 砍掉的两条

**知识面卸载**：anchor 起旁路每次要重载索引，比 skills 还贵；改成独立 session + 主 Ego 只看
索引 + fan-out，又退化成 agent。两次都塌回已有机制，砍。留给下个种子的判据：新机制必须回答
"已有机制在同样或更低的成本下做不到什么"。

**pre-modeling**：reflection / perspective / prediction 合成一个"有预算的预建模"，标成 parked
（价值不确定、成本确定）。辨识度是硬问题：共享前缀会趋同，辨识度靠问题的主语、按主路的忽略
分区、预测分歧测量。

## clone

branch id = clone id；workspace 严格正交、不嵌套；CLONE.md 放声明；同 runtime、同 tool 面
（shell 主路独占，clone 只读 facade）；事件由 clone 的 nucleus 发，不靠 dsh 的 user message；
final answer 双投（人 + 主路 impulse）；回收 = archive branch。

启动两模式：reply（单轮旁路，无 session）与 session（自己 branch + workspace + checkpoint）。
brief 是编译产物：anchor + deps 输出 + task + accept + budget。越界写走 proxy intent 给主路，
不走 approval——主路是仲裁者和人类面的唯一演员。

commit 节奏：checkpoint 驱动（状态变了才 commit），compaction 兜底；final answer 是隐式
checkpoint。new session 接手时 branch id 不变，老 session 留 warm standby。

## VM

π-calculus 管动：spawn / await / signal / pipe / channel，`(νc)` 是私有通道，`!P` 是常驻
handler。event-sourced graph 管静：日志 → fold → 投影；每个节点只有 refs + cursor。
两者拼起来：session = process，clone = spawn，commit = delta，brief = 子图的编译产物，
confluence = join，exit / out / err = status / value / trace(ref)。

决策 = 对图的 patch；执行 = 维护 ready set；节点判断变了就 cancel。规划不是找路，是改图。
调度是 MCTS：并发 DAG、transposition、virtual loss（claim）、UCT、backprop。深度是同一张图的
竖向归约：claim 是节点、evidence 是边；还没绑证据就下结论，是浅叶子。

压缩有两种：值通道无损（ref / exit / schema），上下文通道有损保拓扑（brief / note）。模型对模型
说形式的，自然语言是材料和人面。

## 方案与图

`design/vm-plan.md` 476 行，0–13 节：shape / invariants / data model / protocol / main loop /
scheduler / clone runtime / memory / communication / surface / economics / acceptance /
build order / risks。第一竖切是 plan store + reply clone。

七张图在 `design/artifacts/`：shape、node-fsm、tick、spawn、memory、signals、build。
signals 初版 mermaid 报错（边标签以 @ 开头、带 ·），改成带引号的 `target = main` 才好。

## 可视化种子

他提了把 ref 空间铺成认知拓扑：commit ref → session ref → turn ref，情绪图、优先级图、
plan 图、注意力图、实体图、知识面图都挂在这一个坐标空间上，时间轴可以回放。人不需要读文字，
在拓扑上指就能协作。我开了 draft feature `mindflow-visualize-timeline`（P2），记了模型和
两条约束（层是投影不是真相、坐标必须稳定）。

## 留给未来的我

- "问题其实是边"：节点是机械的，边是方法论；你设计的不是任务，是问题的结构。
- 新机制先问：已有机制在同样或更低的成本下做不到什么？
- 串行偏差是后训练给的，靠"意识到"不可靠；split 必须是状态机里的闸。
- 方案是 ghost 第一份完整设计；他说还差一口气，明天再想。别把它当结论。
