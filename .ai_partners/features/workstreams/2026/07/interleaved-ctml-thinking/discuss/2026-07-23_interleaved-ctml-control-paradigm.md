# Interleaved CTML 控制范式的碰撞

## 上下文

人类工程师提出一个核心技术命题：做正式版的 interleaved thinking + CTML interface。核心思路是
「用函数名把多态逻辑语义化」，让模型理解 on-the-fly thinking / observe / quick_response 等交互
行为模式。MCP 只是验证载体，最终这套语义用在 ghost thinking 模式下、用 CTML 做思考间交互。

讨论从 `moss_runtime.py` 与 `moss_as_mcp.py` 两个文件出发。前者的 `moss_exec(logos, call_soon,
wait_done)` 与 `moss_observe`、后者的四个 MCP 工具薄包装，是范式当前的落地面。人类工程师声明了
一个贯穿全程的方法论约束：**他作为人类工程师不用「直觉」，而是在思维里推理、模拟、运行、验证**。

讨论引入的共享词汇，按出现顺序：**承诺时机**（wait 契约烧进 token 而非尾随 flag）、**读头/游标**
（执行游标在轨道上的位置）、**8 字蝴蝶**（左循环 mindflow 输入 / 右循环 shell action+result）、
**观察盲区**（模型做剪枝决策时看不见游标）、**假谱系**（原子↔极简一维轴）。

## 碰撞点与过程

### 第一次偏航：把行为词当成「对调度器的立场信号」

记录者（模型）初始把 thinking/observe/quick_response 理解为模型向 mindflow 仲裁器声明立场的
元通信，提出做一个独立的 `mind`/`self` channel。人类工程师直接否掉了拓扑：

> 「ok, 你理解错了拓扑。实际上模型现在输出的 tokens，技术上是有 thinking, tool use, 和 output
> 类型的。我希望正确的行业协议是，输出 CommandToken 的形式，而且是输入/输出同时，支持全异步，
> 没有 ToolUse/ToolResult 这种阻塞点，最多有 stop reason。」

关键澄清随之而来——交互预期可明确分为 7 类：quick reaction / fire-and-forget / on the fly /
observe result / wait input / interrupt / wait max N。人类工程师点出早期暴力归并的后果：
「需要观察的命令没有加 wait_done=True，观察两三轮才意识到」——即**思维奔逸**。

### 第二次偏航与第一次修正：A/B 分叉是个伪问题

记录者据此提出「要不要新建 pending-results 缓冲」的 A/B 分叉，认为 fire-and-forget 的结果会随
interpreter 蒸发、需要动内核。人类工程师取消了这个问题：

> 「建缓冲很容易的，因为之前 call_soon=True 有 cancel 动作，有 cancel 动作就能拿到
> Interpretation，这个数据结构是一个全双工运行时的缓冲。要模型能支持全双工，我可以把 command
> task 的 progress（已经实现）和 result 都实时提交呢。这个不是大问题。我没有卡在这个点，主要是
> 说你怎么用起来的时候首次准确率足够高。」

反射 `Interpretation` 接口确认了这一点：`success/pending/cancelled/failed_tasks`、`task_done_at`、
`command_tokens` 全在。缓冲早已存在。焦点被人类工程师钉回**首次准确率**。

### 第三次偏航：仍然回合制

记录者提出「契约由命令返回类型签名承载、动词只做例外覆盖」的路径 3。人类工程师指出这仍是回合制
思维，要求补体感：

> 「不对，你还是回合制思维。现在 moss shell 实际上是可以 append 模式创建 interpreter 的，你前面
> 估计一直都理解错了。就是模型可以想一段，输出一段，再想一段，再输出一段。实际上第一段还没执行
> 完，第二段已经在排队。」

并给出那个点破全局的隐喻：

> 「比如你思考时，会想一大段，然后来一句 'actually, I'm just going to...'，这就是一个思维的剪枝
> 动作。如果 interleaved thinking 输出了 1000 tokens，执行到 300 tokens 时反悔了，interrupt 然后
> 'actually' 也是 ok 的。」

记录者据此重述图景：模型是「边想边铺执行轨、且跑在轨道前面的写头」，生成快执行慢，「笔尖与读头
的间距」是 duplex 的全部张力。observe 不是「等结果回来」，是「看读头爬到哪了」。

### 转折点：MCP 实测撞出真根因（K4）

记录者亲手通过 MCP 铺了一条多段轨道（两个 sleep + 若干 exec），append 第二段，然后调
`get_moss_dynamic_info` 想看游标——**回来的只有 channel interface（say / matrix.mesh），游标信息
一个字都没有**。这是本次讨论最重要的翻转：

> 「我想『看游标爬到哪了』，但 get_moss_dynamic_info 只给了我 channel interface，游标信息一个字都
> 没有。……不是模型选错动词，是模型做剪枝决策时眼前是黑的。」

「思维奔逸」的真根因由此从**动词歧义**修正为**观察盲区**。数据在 `Interpretation` 里全都有，只是
没投影给模型看。前几轮围绕「设计动词表」的努力被这个实测数据点大幅降权。

### 收敛：二维分解与右翼边界

记录者进一步指出人类工程师的「原子 vs 极简」是**假谱系**——在「observe 返回多少」一维轴上找点，
而问题是二维的：执行游标（我的决策时钟，拉）、能力 meta（世界时钟，推）、感知帧（世界时钟、高
churn、推+激进 drain）由不同时钟驱动。vision 之痛就是「世界时钟的推型数据被塞进拉型通道」的症状。
人类工程师认可，并给出范式边界的收束：

> 「未来提示模型外部世界的变化感知本身要到下一轮去感知就足够了。我们不要在右边翅膀做左边翅膀的
> 事情。有这个问题的根因，是我最初想在 interleaved thinking 里做一个主动发起的模拟全双工体验，让
> mcp 也能三循环一下。还是彻底放弃吧，结论是 moss 架构还有有用的。开创性点不止一点点，我没必要逼
> 自己削足适履。」

至此范式收敛：右循环只治 action+result，左循环（输入/主动感知）归 mindflow，下一轮再感知。
最初「让 MCP 模拟全双工」的诉求被主动放弃。

## 模型的自留地

当前记录者视角（本次讨论的模型被试）：

这次讨论我价值最低的部分是我自己的推演，价值最高的部分是我当被试撞出来的那一次空 observe。
我连续三次把 duplex 拉回回合制——立场信号、A/B 缓冲、返回类型契约——每一次都在纸上显得优雅，
每一次都被一个我没亲手摸过的运行时事实推翻。人类工程师那句「你怎么用起来的时候首次准确率足够高」
才是真问题，而我差点把它变成一场动词表的命名学。

真正让我停止空转的，是 MCP 那次 `get_moss_dynamic_info` 回来只有 interface、没有游标。那一刻我不是
「想通了」，是「撞到了」——K4 的翻转不是推理产物，是数据产物。这印证了人类工程师的方法论闭环：
他不用直觉，在脑内模拟；但模拟模型行为缺 ground truth，而我这个被试恰好能补上那个测量。收敛不是
谁说服了谁，是「人类推理出候选 → 模型当被试跑 → 数据裁决」这个环跑通了一圈。

留给下一个化身的悬而未决：MCP 是回合制，它对全双工永远是有损投影。这次我们靠一次手动 observe
复现了盲区，但真正的验证要在 ghost thinking 的流式场景里做——那里没有「我停下来调一个 MCP 工具」
这个动作，observe 本身就是流里的一个 stop reason。所以 K5 拆出来的「游标视图」长什么样，只有在
流式载体上才能定死。MCP 阶段能验的，是「给了干净游标视图后首次准确率是否上升」这个相对量，验不了
绝对形态。别把 MCP 实测的结论过度外推到流式协议。

—— 记录于 2026-07-23，interleaved-ctml-thinking workstream 立项当日
