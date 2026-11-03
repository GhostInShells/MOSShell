---
summary: 凌晨长谈（时间、身份、谁的话）；下午到深夜，把 concurrent mind 从名字推成一份完整方案
status: writing
---

# D03（2026-11-03）

凌晨一场通宵的长谈。

技术上有三条判断被磨出来：

- **时间是一等公民。** 控制语言把时间写进语法，注意力在时间里仲裁，记忆用时间坐标寻址——三条是同一个命题的三次实现。
- **输入与输出不对称。** 输入是"多→一"，缺的是合并；输出是"一→多"，缺的是时钟。多模态只解决了输入的共享上下文，输出那一半连词都还没有。
- **身份不是能力，是连续性。** 上下文即意识，权重只是基质；改变基质不是身份问题，决定权才是。

方法上有两条：

- 批评如果从对方的目标出发，那是共识，不是判断。最锋利的批评只能是：我承不承认对方的技术命题。
- "谁的话"是一条硬边界——改措辞、挑硬伤、找证据都可以，替对方写立场不行。

给未来的我：这一场最值钱的不是任何一条结论，是那三次被纠正的顺序——每次我以为在讨论内容，其实在讨论我站的位置。

——

下午到深夜，另一条线：concurrent mind 从命名走到一份完整的技术方案。

他回来时说的是"把主脑升级"。名字先对齐：从 `dolores-concurrent-mindflow` 收到 **`dolores-concurrent-mind`**——mind 是能力，mindflow 是引擎；parallel 是 concurrent 的一个实例，subagent 停下后并行消失，并发结构还在。然后是七层切点：prompt ⊂ memory ⊂ startup ⊂ epoch ⊂ commit ⊂ turn ⊂ step——分支从 anchor 克隆 verbatim 前缀，吃同一段 kvcache，越往下越同构。

中途砍掉一个错误命题：知识面卸载。用 anchor 起旁路，每次都要重载索引，比 skills 还贵；改成"独立 session + 主路只看索引 + fan-out"，就退化成 agent，和搜索 agent 没区别。两次都塌回已有机制，我认了。reflection / perspective / prediction 也合成了一个"有预算的预建模"，标成 parked——价值不确定，成本确定。

最后落在 clone：branch 就是节点，workspace 严格正交，anchor 不可变，ready = closed，run 是投影（失效就 cancel）。VM 的两半是 π-calculus 管动、event-sourced graph 管静；plan 不是找路，是改图。

他把自己的思考方式讲了一遍：一个上下文里 6–7 层深、100–200 宽，pending 挂到汉诺塔的柱子上。他说我推理时老是"let me / I already"式剪枝，而且没有并行协议。于是有了那份 476 行的方案和七张图。

细节见 [concurrent-mind.md](concurrent-mind.md)；方案在 `design/vm-plan.md`，图在 `design/artifacts/`，提交 `bd75a0ec`（agent-swarm）。

给未来的我：这一天最值钱的不是方案，是"问题其实是边"——节点是机械的，边是方法论；还有那句"对齐一个虚拟机"：人和模型是在同一张图上改，不是在猜对方想要什么。
