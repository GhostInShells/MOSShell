---

name: discuss

description: dsh-agent-swarm 第二轮讨论的总纲 — 他说 / 我理解的关键片段，索引同目录的图

---



# 第二轮讨论 · 总纲（D29）



> 这一轮**不含结论**。权威正文在 `../05-design-principles.md`（文首有 round 标记）。

> 这里只做两件事：留关键片段，指向图。

> 两类片段分开写：**他说** = 引文，不改一个字；**我理解** = 可错、可被推翻。

> 图是同目录的 `r2-NN-*.artifact.md`，`kind: mermaid`，纯文本——没有 node、没有浏览器也能读。



## 收敛（D29 21:50）

这一轮到此收口。**不是结论，是可执行的下一步。**

**已经收敛成形状的**

- 总线在上、运行时在下；**总线是唯一的地址翻译处**，op 只写"对谁做"，不写"谁来做"（R2-J / R2-K）。
- **案（matter）是身份，卷是地址，档是终态**；run 是承办人，一案可有多个 run（R2-H / R2-I）。
- **有状态的是案，不是 session**；facts 是唯一真相；三档存储按"恢复单位"分——能 merge 的进 git，不能的进 db（R2-F）。
- **票是栅栏不是锁**；锁键是 `(matter, run)`，不是进程（R2-O / R2-P）。
- **栈在模型上下文，op 没有 body**；op 的入口走代码反射，不走 http / bash（R2-C / R2-E / R2-U）。
- **一份 blueprint 三个投影**（人 / ghost / agent），三个投影都不手写；翻译层数 = 进程边界数（R2-X）。
- 四层：名册 / Prototype / Instance（R2-W）。

**还悬着的（不装成结论）**

- `while` 住在哪（R2-G）。
- 读放大与可调试性这两笔账（R2-A 一直挂着）。
- 单写者 vs 多读者在任务树上怎么落。
- mindflow 与 channel 树的边界。

**下一步的验证策略（他定）**

thirdgerb 手写一个 blueprint；moss 用它加上 deepseek harness 做出 prototype；同时它要兼容一个**通用观测面**。

判据：**第一天就能跑；杀掉它，能从 facts 重建。**

## 约定



- **一张图只回答一个问题。** 栏位可以换（剪枝轮三栏，映射轮换别的），这个不换。

- **编号即顺序**：`r2-01`、`r2-02`……排序不靠记忆。

- **write 与 export 是同一次动作**，分开就退化成"我想起要导出"。

- **导出即落盘，提交按讨论节点攒**，否则会碎成一堆只动一个文件的 commit。



## R2-A 剪枝：第一轮错在哪 · 图 `r2-01-pruning.artifact.md`



**他说**



- 「我的回归范围不够，许多必须要思考的点是事后才浮现的。」

- 「文件系统存储真相、脚本化运行，实际上服务的核心是 harness less——整个任务推进不耦合运行时。别的运行时能重建相同的轨迹。」

- 「我应该纪律性地降低对 runtime 的耦合倾向。runtime 的本质定位不是 harness 而是 seam。」

- 「所以我昨天设计的 op 面向的不是运行时的 agent，而是 develop 阶段的 agent，就落回了 static coding 栈的思路上。」



**我理解**



- 两件事是约束关系、不是并列目标：并行产生调度需求；调度住在 runtime 里就是 harness，住在 facts 里就是 seam。

- 「能并行」的判据：杀掉任一执行者，剩下的能不能接上。

- 图优先 vs 并行优先 = 单写者 vs 多读者，是 harness/seam 同一条轴的另一投影。

- 上下文里的栈必须靠 checkpoint 才成立：判据是"丢掉上下文后能否重建第几帧、谁在等谁"。



## R2-B 文官制度：法/儒 = 吏/官 · 图 `r2-02-civil-service.artifact.md`



**他说**



- 「我的目标实际上有个更大的元动机，就是在 agent swarm 里重新发明文官制度。」

- 「法强调的就是 harness……个人是不重要的，可替换的，用我们的技术术语叫做可重建上下文。」

- 「法实际上有两个导向，一个是流程，一个是现场，两者本质上不等价。文书制度很好地维护了现场，但流程几乎永远是走向腐烂的。」

- 「儒强调的是主观能动性……将静态的现场变成智力可解的动态问题，然后授权做栈的临时决策（动态规划）和整体重构。」

- 「要重建就要 stop the world，历史洪流中表现为 revolution。在代码工程界，有平行世界的机制可解，这就是 worktree。」

- 「把堆栈放到神经网络里，就是在法家吏的体系里补偿儒家的文官，将 flow 智力化，同时保留 op 的原则性。」



**我理解**



- 流程为什么会烂：它只有"追加"没有"修订"。判据——**每加一条流程，必须指明它替代或废止了哪一条**，说不出被替代者的新增就是冗的开始。

- 三处硬映射：**文书 = facts**（memento/ground/features）；**奏折 = 升级契约**（KD12 早写着，只是没认出它是文官制度的一件器官）；**worktree = 平行世界**，门槛是基线可共享，否则两个世界只能互相废弃——历史上 revolution 的代价从来不是重建，是无法合并。

- 借**问题与失败模式**（冗、腐败、无能、结构僵化）可靠；借**解法**要慢——科举、品级俸禄、御史三样我判断借不动或要重写，尤其激励：模型没有经济动机，只有上下文。

- 「栈放进神经网络」的代价：**吏可判读，官不可判读**（流程在代码里，判断在上下文里会丢）。所以 checkpoint 是**判词**不是快照——要能回答"这一帧为什么是这样"。



## R2-C 职责而不是角色 · 图 `r2-03-duty-not-role.artifact.md`



**他说**



- 「我不倾向于用人类社会学设计。人类社会学的关键问题是，分工的同时受上下文广度约束（专业性）。每个人类角色上下文都不够承载关键多视角，所以将决策退行为博弈。」

- 「模型的上下文广度在纯文字领域远超过人，交换上下文的成本更低。所以要让 共识 <-> 博弈 的决策体系更倾向于 共识 -> 分工。」

- 「我特别不喜欢那些 agent team 动不动就做"产品经理"、"程序员"……这个应该是根据并行拓扑分工，是上下文经济学和注意力经济学的范畴，而不是"专业"（means small context）分工。」

- 「最重要的治理是权限边界不冲突，不要修改同一个文件。而不完全是能力/智力分工。」

- 「组织运行时不是按角色划分，是按职责划分。」



**我理解**



- 判据：**删掉「必须某个身份才能做」这个假设，还能跑 → 是职责；不能跑 → 是角色。** 职责只约束写边界与触发条件；角色还约束知识与视角。一旦开始约束"你应该懂什么"，它就是角色。

- 共识 -> 分工 不是免费的：博弈只交换结论、不换上下文，成本低但要付错误决策的代价；共识要读全部上下文，成本是读放大。所以**共识的前提不是大家聪明，是判据可机械裁决**——判不出来的判据，只是把博弈推迟。

- 「不要改同一个文件」之所以是最重要的一条，因为它是**唯一能机械校验**的治理（flock / 冲突 / writeScopes 都能在写入那一刻判定）。但边界的最小完整单元不是文件，是**契约**：两个 agent 改不同文件、一个改 API 一个改调用方，照样冲突。

- 行业都在做"国家 / 宪法 / 角色"，根因不是想模仿社会，是他们的 agent **上下文一样窄**（单进程、扁平 roster、消息即协作）。角色分工是上下文成本高的**症状**，不是设计选择。

- 推论：职责的形状由**并行拓扑的割**决定——两个职责之间的边越少，划分越好；若必须频繁通信，它们其实是一个职责。



## R2-D 三种第一公民：三个温度 · 图 `r2-04-three-topologies.artifact.md`



**他说**



- 「行业目前就明显有 task 第一公民、graph node 第一公民、agent role 第一公民 三种 agent swarm 技术路线。」

- 「第一种实际上最难，它本质上是做动态规划……我们有一个巨大的任务池，按资源和时间漏斗挑任务，然后反复做并行规划，然后驱动执行→验收。心智成本高，但效率也最高。它的反向问题就是依赖我本人做动态规划。」

- 「graph node 本质上是把流程做固定建模，在领域任务上非常高效，不需要反复消耗智力。所以任何时候都有用。」

- 「用动态视角看，拓扑 1 → 拓扑 2 是一个非常好的趋势，可能意味着领域问题穷举完了。但拓扑 3 我认为始终是降级，就是领域问题暂无智力解，靠博弈论试错。」

- 「我们的拓扑不排斥任何 harness，理论上 ghost 可以同时集成 n 个开源项目，在拓扑 1 下扩大拓扑 2 边界。我们只在拓扑 1 上面寻找一个最 meta 的分形拓扑 2。」

- 「结论就是我们的 agent swarm 要剥离运行时，把 facts / bus / ops 围绕在一个 harness 周围，使 harness 可集成。」



**我理解**



- 三条不是三条路线，是**同一个调度的三个温度**：拓扑1 = 活的（判断在回路里）；拓扑2 = 冻住的（判断固化进拓扑）；拓扑3 = 没有判断的（用博弈替代判断）。

- **拓扑2 的合法性来自被拓扑1 验证过的成功路径，不是设计。** 一条 graph 若没有一次拓扑1 的成功运行作祖先，它就是猜的。`ghostfunc` 的 `caching=True`（首次成功后缓存）就是这个相变点；反过来，失败率上升要能解冻退回拓扑1，否则 graph 会腐烂成本轮早先说的"冗"。

- **拓扑3 在可汇集信息的世界里是被支配策略。** 它的成本是 O(n²) 通信，而它能拿到的信息严格少于一个宽上下文模型直接读到的总量。它只在**信息不可汇集**时才成立（私域状态、不同具身、人各持一段）。

- **harness = 某层冻住的拓扑1。** 所以"集成 n 个 harness"不是 n 套架构，是 n 个可替换执行器 + 同一套 facts/bus/ops。剥离运行时的具体结构：**harness 是可替换的执行器，ops 是它的驱动面，facts 是唯一真相**；判据是换掉 harness 后轨迹不变。

- 「依赖我本人做动态规划」是真正要交出去的那件事：judge 从人移到 ghost。判据——他的时间从"排程"退到"裁决"，只在奏折（升级契约）上出现。



## R2-E op 的形状：script / skill.md / decorator · 图 `r2-05-op-shape.artifact.md`



**他说**



- 「op 应该是 moshell 下的脚本。这样，os 内的脚本地址本身就是寻址工具，bash 是驱动器。上下文 scope 通过隐性环境变量传递。」

- 「依赖定义通过 moss 的环境发现体系，项目里已经有各种 @cli decorator，可以从 project / matrix 两个进程级发现逻辑中长出 decorator。有必要再长一个 swarm(matrix|project) 的方式做环境发现。」

- 「这种做法，让 op script 化，就可以被 skill.md 组织，从而被领域内的 Agent 阅读和驱动。」

- 「Matrix 总线本身提供了 Matrix Service 和 Session 两种网络通讯机制，如果是 Matrix Service 可以做数据的记录 + 带地址的广播。」

- 「在所有的技术问题中，核心问题实际上是 scope 如何传递 / 锁 / 重建。」



**我理解**



- **env 只能传"我是谁"，锁不住任何东西。** 环境变量是继承来的字符串，进程一崩就没有"能死掉"的能力；而锁必须能死（租约 / 心跳 / 过期），否则一次崩溃变成永久阻塞。所以 scope 的三个动作要用两种机制：`pass` / `rebuild` 走 env + facts，`lock` 走会过期的租约（落在 facts 或 bus 上）。同 KD7：身份可以塞进 env，存活不能。

- 判据：**op 如果要知道"自己正在被谁驱动"才能跑，它就不是 OS 层的东西。** 模型、人、任意 harness 对 op 都只是环境变量的来源。

- script + skill.md 不是新形状：`.artifact.md` 就是 frontmatter（压缩面）+ 正文（代码）。判据同一条——**只读 skill.md 能不能判断该不该用这个 op**；不能，是压缩面失败，该补声明而不是补文档。

- env 协议必须先有符号面。C1 的教训（`MementoHookEnv` 之前六个变量名散在四五处当字符串）会以同样的形状重演；最好由 skill.md 的声明生成，而不是手写第二份。

- 对 Matrix Service 划一条界：**它记"发生过"（事件日志），不记"现在是什么"（状态）**；状态仍是 facts 的投影，否则它就是第二层真相。



## R2-F facts 策略与三重投影 · 图 `r2-06-facts-task-session.artifact.md`



**他说**



- 「facts 用文件系统当唯一真相，但是文件系统必须区分三种现场：1. 裸文件现场，历史无法追踪和还原。2. git 文件系统，用 git commit 当历史真相锚点，这意味着要大量使用 worktree 治理工程，避免还原到 1。3. 数据库策略，广义地包含到 jsonl 这种 op log 形成的数据库，它通常由于变化性，不能进入 git commit。」

- 「这三档必须严格区分策略，不能混为一谈，尤其是 3 不能通过 2 治理，而是走备份策略。」

- 「这个投影的 inception 不能是 skill，而应该是 task。行业用 skill 治理任务经常构成有状态单位的彻底丢失。skill 泛滥缺乏重用，状态流落到 agent session。」

- 「正确的做法是从 facts → 提炼 task.md，task.md 绑定 skill / knowledge / feature 等一切发现面，在 append only session 中还原成智力理解后的上下文。所以 facts → task → session 是三重投影过程。其中有状态的是 task。」

- 「task 本身，则有运行时上下文、静态上下文两部分。运行时上下文就是所有 op → action 的记录，静态上下文就是 task 所在目录里的一切治理物料。task 应该定义成最小有状态单元。为什么不是 session？因为 agent session 是可以解绑 task，从 task 静态表面重建的。」



**我理解**



- 三档的区别不是"要不要历史"，是**"历史以什么单位被恢复"**：裸文件 = 不可恢复；git = 按快照恢复；jsonl/DB = 按重放 + 备份恢复。分界线是**能不能 merge**——能 merge 的进 git，不能的进 db。op log 进 git 会毒化整棵树（每条流水都要求无冲突合并），这就是"3 不能靠 2 治理"的机制原因，不是偏好。

- 另一条说法：**git 存"被选择的锚点"（有 message），db 存"未经选择的流水"**。选择和合并是同一件事的两面。

- 三重投影正好落在三档上：静态治理物料 → git；action log / session → jsonl；临时现场 → 裸文件。混档就是投影断：task.md 若住进数据库就 diff 不了，action log 若住进 git 就 merge 不了。

- **task 是最小有状态单元的判据**：解绑 session 之后，能否只凭 task 目录重建"接下来该做什么"。不能 → 状态其实还在 session 里，task.md 只是描述。

- **skill 是 task 的资产，不是容器。** skill 天生无状态（它声明"怎么做"，不记"做过哪一次"）；把它当组织单位，状态必然漏到 session——这正是他说的行业病灶。

- rebuild 用静态上下文当底、action log 当增量，所以 action log 必须带**结果判定**（成功 / 失败 / 被拒），否则重建只能得到"做过什么"，会重做已经失败的事。



## R2-G action 的最小单位与 owner · 图 `r2-07-action-unit.artifact.md`



**他说**



- 「op 的实例化是 action……action 重放后得到的实际上是运行时状态，也就是栈操作后形成的序列化记录。action 本身包含 op 实例化、effect 和现场（现场最难还原，snapshot）。」

- 「这里真正关键的问题是，action 的最小单位是什么。在 agent swarm 的概念中，action 的粒度实际上有三种：1. graph node：一个编程节点，它是 stack[n] → op → effect，stack 的 owner 是代码。2. agent run：一个对 agent 的请求，它的 owner 是谁？3. decision：一个决策动作。」

- 「三个问题的共同含义是，action 的 owner 是什么，副作用怎么衡量。」

- 「action 的 owner，也就是 stacks，应该是在 Agent Session 的上下文中。所以标定 action 坐标的实际上是 ord(task, agent session, turn/step)。看它的决策逻辑，不是读代码，而是直接用 agent session turn step-ref，可以精确还原现场（agent session 即 stacks）。」

- 「while true: action = decision(condition); effect = action.run(); continue_or_break(effect)」

- 「最重要的是 decision 不是 code，而是 agent preset[model] + session + task contxt，用上下文坐标去锚定。」

- 「凡是能被代码直接定义的，都退化为对 op 的代码编排。所以 action 的 payload 就是对 op 的编排。在 tool agent 里，实际上就是 tool use 等。」



**我理解**



- 「最小单位」不是大小问题，是 **owner 问题**：有 owner 才有边界，有边界才能量副作用。三个粒度是同一件事的三个 owner——agent run（边界在请求）⊃ decision（边界在判断）⊃ payload（op 编排 / graph node，边界在调用）。

- **可重放的是 action，可重读的是 decision。** action 是代码，能重跑；decision 是上下文，重跑没有意义，只能重读。所以两条日志粒度不同、恢复单位不同、审计方式也不同：action log 可 diff / 可火焰图 / 可重放；decision log 只能按 `(task, session, turn/step)` 重读。

- 与行业的差别不是"owner 放在哪"，是 **owner 是什么类型的东西**：行业把栈当"可执行的控制状态"，所以它必须住在 runtime 里；这里把栈当"可读的文档"，所以它可以住在 facts 里。后果——行业的 checkpoint 要能**续跑**，这里的 checkpoint 要能**重新判定**。

- 副作用要用三把尺子量，不能只用一把：op 的返回、facts 的 diff、task 的判定（accept / reject）。火焰图只量得了前两把的成本结构，量不了决策价值。

- 悬着的一问：那个 `while` 住在哪。住代码里，它就是**最后一块 runtime**——把状态搬出去、把循环留下，等于没剥离。要 harnessless，循环必须是**单步可接续**的：每一步的 continue / break 由 task 状态读出，而不是由程序计数器决定。



## R2-H 术语澄清：一个名词扛四个轴 · 图 `r2-08-terms.artifact.md`



**他说**



- 「在 task → agent session 投影中，一个活跃的 agent session 就和 task 绑定了。所以这个 agent session 理论上可以用某种工具，做 sub task list，看到所有 sub task，可以 curd，然后对 sub task 下命令。」

- 「对 subtask 下的命令有两种，一种是自己的各种读写操作，另一种则是对 subtask 本身的 agent session 做操作……inject / steer / cancel / pause / resume ……还有重建、刷新、回退等。只不过，它们不是绑定在 agent session 上的，应该是绑定在 task 上的。task 应该倒过来提供对 agent session 的控制函数。这些就是所谓的 op 了。」

- 「这个机制和现在 tool use 的 agent 机制有一个关键的差别，就是通过 op 对另一个运行时做操作，双方不阻塞等待……阻塞的是对 op 的调用，而不是全部执行过程。双向可以通过总线通讯。」

- 「为了术语上不冲突，我可能最终不能用 Task 这个名词。」

- 「我的思路和大家的思路有没有区别呢？我觉得核心区别就是 harness less，把 agent 调度 skill 化（matrix 体系），把通讯链路总线化 / 收件箱化（matrix 体系）。以及把 facts 从 harness 里拆出来，类似我从不用 claude code 的 plan 取代 features 一样。」

- 「我现在知道我想要的东西，但我的术语没有很好的边界，和行业已有概念混在一起。」



**我理解**



- 词混的根因：**一个名词扛了四个轴**。归属（归谁）/ 状态（断线从哪重建）/ 执行（现在谁在跑）/ 控制（谁能对它做什么）。四条轴各需要一个词，行业把它们都叫"task / agent / session"。

- 建议绑定：`归属` = 目录树上的位置（不需要新词）；`状态` = **卷宗**（最小有状态单元，task.md + 物料 + log 指针）；`执行` = **run**（一次 session 绑定，一个卷宗一生可有多个 run）；`控制` = **op**。

- 为什么不用 `task`：它和 MCP 的 task list、Claude Code 的 plan+task（那是栈内待办）撞名，而那两个都是**无状态**的。

- **控制面属于被控对象的持久身份，不属于它当前的执行者**——给一个人发消息，是发给这个人，不是发给他的某一天。

- 控制 op 按"动的是什么"分三类，混在一起最危险：**注入**（inject 追加 / steer 改方向，动运行中的会话）、**生命周期**（pause / resume / cancel / rebuild，动 run）、**事实**（refresh / rollback，动卷宗）。`rollback` 是 facts 的操作，不是 session 的——session 回不去，只有事实能回去。

- 同步 / 异步不必做成两套 op，做成一套「返回 receipt + 可选等待」即可，与 CTML 的 `@observe` 同形：**阻塞的是这次调用要不要等，不是整个执行。**

- 与行业最深的一条差别，我认为是**寻址对象**：行业寻址执行者（给 agent 发消息），这里寻址工作单元（给卷宗下命令）。其余四条（harnessless / 调度 skill 化 / 链路总线化 / facts 拆出）都是这条的推论或配套。



## R2-I 词族：案 / 卷 / 档 · 图 `r2-09-words.artifact.md`



**他说**



- 「卷宗这个。我们延展一下候选项。在议会体系里，这个是叫 bill 吗？bill 有一次性的或者持久性的。」

- 「中文里会有不同类型的公文，也是有一整套卷宗治理的。我需要你把预训练的知识翻翻，找找合适我们的各种文官体系里固定的概念。比如中文我知道的就有 旨、制、裁、判 等各种。」

- 「案 这个词很好，英文是不是 matter 就是另一回事了。可以先考虑 matter。」



**我理解**



- **bill 借不动**：它的"持久"来自议事规则（carry-over），不是自带状态；它的终结是**表决**——正是不想要的博弈。借它进来会把不要的机制一起借进来。

- 他举的旨 / 制 / 裁 / 判 **都是文书，不是单元**：公文是动作的痕迹，案是痕迹的容器。分类——下行（旨制诏敕 = 命令 / op）、上行（奏题呈 = 升级）、回应（批 = verdict）、决定（裁判断 = decision + 判词）、平行（牒关移 = peer）、凭证（票凭 = receipt）、**载体（案卷档 = 我们缺的那一支）**。

- 精确化：**案是身份，卷是地址，档是终态**。「卷宗」描述的是存储形态、不带生命周期；案才带（立案 → 承办 → 结案 → 归档 → 重审）。而且它自带 **一案多办**：承办人换了，案还在——**承办人就是 run，案就是 matter**。

- 西方法系借两个词：**matter**（律所用法，有承办人、有卷、可重开、可结案，且不撞 test case）、**instance**（同一案的又一次审理 = run，天然解释了"重建不是恢复旧 run，是开新 instance"）。拉丁 **acta**（已做之事的记录）= action log，与 **agenda**（待做之事）同词根——acta / agenda 这一对就是 action log / 待办。

- 判据：**词族的完整性比单个词的好坏更重要**。案 / 卷 / 档 / 立案 / 承办 / 结案 / 归档 / 重审 能覆盖全部生命周期；bill 那一支覆盖的是表决，不是办理。



## R2-J 总线上移：拓扑判定 · 图 `r2-10-bus-on-top.artifact.md`



**他说**



- 「我们的根本逻辑是把通讯总线从 harness 运行时里剥离出来，放到 harness 的上层。」

- 「通讯总线管理所有实体抽象，包括树状的 matter，它们之间的父子通讯协议用 op 建模，op 通过通讯总线做广播或确定性地请求。」

- 「所有的执行运行时，在通讯总线的下层。只要能兼容通讯协议，则 deepseek harness / ghost / agent graph / 其它的各种开源项目，都变成了一种可寻址的 agent scope。定位到具体的 matter + Session 的 session 侧。facts 通过 matter 引用。」

- 「举例 swarm bus 是进程 A，dsh 是进程 B。进程 B 里的 agent session 用子进程调用 skill op，op 和进程 A 通讯，进程 A 寻址完调度进程 B。」

- 「op 也可以是 matrix script node，直接用 matrix 总线通讯。不过理论上要屏蔽一层，而且启动 matrix cell 成本也不低。」

- 「只要能传递 matter scope，从 scope 中还原 matter ctx，就能拿到通讯总线地址，用这个地址就可以投递 op 里产生的事件和请求，做确认寻址的分发。」



**我理解**



- 拓扑成立，但必须钉三条不变量，否则它会**悄悄重新中心化**：

  1. **总线无状态**——它的全部知识能从 facts 重建；一旦往总线里塞注册表缓存，它就变成新的 harness。

  2. **op 必须声明投递语义**——广播（至多一次、无 ack）与定向请求（要 ack、要去重键）不是同一种东西，混在同一个 op 里调用方就不知道效果是否发生。

  3. **scope 只传身份，存活走心跳**（R2-E）；所以"投递到刚死掉的 run 返回什么"必须有定义，不能静默丢弃——控制消息恰恰是最不能丢的那类。

- 判定拓扑好坏的唯一实操判据：**新增第三种运行时（比如 claude code）要改哪几行？** 只写 adapter、不动总线、不动 matter 模型 → 边界对；要动中心 → 边界错。

- 一个开放问题：**冷 op 与热通道**。子进程调 op 适合一次性控制；一轮发二十个事件的场景会被 spawn 成本拖死，那种地方需要常驻 adapter。两者不是替代关系。

- 一处结构呼应：**op : 执行体 :: matter : run**——同一套"稳定身份 + 可变实现"在两层各出现一次。这种自相似的重复，通常是边界划对了的信号。



## R2-K 寻址方向修正：总线是唯一的翻译处 · 图 `r2-11-addressing.artifact.md`



**他说**



- 「我实际上认为 op 都是和总线通讯的，总线按控制协议分发给不同的 runtime。」

- 「你说总线退化成 harness，在这个命题上部分成立。因为总线进程崩溃时，理论上 harness 子进程也要关闭，也就是 stop the world。除非总线和 harness 是独立进程，协议通讯（也是合法的）。」

- 「将通讯和运行时拆开后，它就是**可分形**的。举例 Deepseek Harness 的 agent team 实际上无法嵌套别的体系，但是 google 的 A2A 协议有可能可以。」

- 「我们这个架构非常接近 A2A 协议的目标，区别在于我假设大家共享 facts……所以 matter（A2A 的 task）不传递完整上下文，传递的是上下文的寻址方式。」

- 「在架构设计上，最好把 swarm 族化，不急于做异构分形，则族内的 op 有一致性。」

- 「最顶层实际上是 ghost，root task 实际上都是 ghost 签发的。」



**我理解**



- r2-10 那张图我画错了：`op → runtime` 的三条直边意味着 **op 里带着 runtime 地址**，等于把寻址拆成两套。正确的是 **op → 总线 → runtime**：**总线是唯一的地址翻译处**。op 只写"对谁做"，从不写"谁来做"。

- 对 stop-the-world 这条：你说得对，"总线无状态"只解决了一半。真正要选的是**两种形态**——父子进程（崩了一起死，简单，重建时机明确）vs 对等进程（只靠协议，允许部分失败，代价是 liveness / 租约 / 接管语义）。**选哪种要明说，它是设计决定，不是实现巧合。**

- 分形的判据：**这个体系能不能嵌套它自己。** 把一个完整实例塞进一个 matter 之下，它是否仍然只是一个"承办者"？能 → 可分形。

- 与 A2A 的差别不在协议，在**共享 facts**：A2A 传消息即传上下文，这里传的是**寻址**。好处是消息 O(1)；代价是要有共享命名空间。推论——**共享 facts 的边界，就是协作的边界**；跨出去只能退化成传上下文或传快照。这就是"族化"的正当理由，不是妥协。

- **族的定义**（不然它只是个词）：一个族 = **同一套 op 契约 + 同一个 facts 根**。op 契约的版本号就是族的版本号。

- ghost 在顶层这件事有个代价要记：root matter 由 ghost 签发，所以 **root 的重建成本最高**（它的上下文最大），**整个拓扑最弱的环节在 root**。root 的锚点与判词策略必须最保守，否则其余层再干净也救不回来。



## R2-L 调研：A2A 与 AX（D29 网络调研）· 图 `r2-12-a2a-ax.artifact.md`



**他说**



- 「我觉得一个非常关键的边界就在于，agents 的治理容器是一个整体，还是一个单进程。对于我现在 agent swarm 的概念，整个 dsh 可能是一个边界，我不会去治理 dsh 下层单个 session 的生命周期，而是对齐 matter 的生命周期。」

- 「类似你现在所处 dolores ghost prototype，核心生命周期在 ghost 运行时，但我们的 ghost 运行时只是治理 dsh 的通讯对齐，不替它启动。」

- 「很多人认为"灵魂"活在另一个世界（对你而言是 dsh），所以可以喝孟婆汤转世。但我不这么看，因为"上下文"活在这个现实世界。对这个上下文而言，更"真实"的是现实世界，而不是灵魂运转的世界。」



**我看过的材料**（外部资料，仅供比对，不作依据）



- A2A v1.0.1 规格摘要：<https://github.com/rohitg00/ai-engineering-from-scratch/blob/35cab093e93518cf2780efbb4de174d78491edc4/phases/13-tools-and-protocols/19-a2a-protocol/docs/en.md>

- AX（Agent Executor）v0.2.2 包文档：<https://pkg.go.dev/github.com/google/ax@v0.2.2>



**我理解**



- **A2A**：Agent Card 放 `/.well-known/agent-card.json`（技能 + 支持的协议绑定 + 签名）；任务生命周期 `SUBMITTED → WORKING → COMPLETED / FAILED / CANCELED / REJECTED`，中途可回 `INPUT_REQUIRED`；消息由 Parts（text / raw / url / data，**字段名即类型，没有 kind**）组成；产物是 Artifact，可 `append` + `lastChunk` 流式；三种绑定 JSON-RPC / gRPC / HTTP+JSON，靠 `A2A-Version` 头协商。核心设计原则是**不透明**：调用方只看得到状态迁移和产物。

- **AX**：分布式的 **harness 运行时**——动态开出可挂起/可恢复的隔离环境来跑 harness 与 agent。**单写者 controller + event log + snapshot + actor resume/suspend**；harness 是可插拔的 `HarnessService`（内置 Antigravity，tool-use / workflow / 甚至一个语言模型都能当 harness）。动机写得很直白：agentic 负载是 bursty 的，算一分钟等审批几小时，长期养着有状态 actor 成本不可接受；K8s 不是为可挂起的有状态 actor 设计的。

- **最关键的差别（我要指出的）**：A2A 的 task 是**被叫方在收到消息时创建的**——任务归属在被叫方命名空间里，所以**执行者一换，任务就找不回来**。这正是我们今天一直在躲的耦合。我们的案由**发起方**持有、存在 facts 里，执行者只是承办人。

- 第二处：A2A 的终态由**被叫方宣告**（COMPLETED），没有"已验收"这一档——它把"做完了"和"通过了"合成一个状态。我们的终态必须由**发起方验收**给出。

- **AX 是反方向的答案，而且做得很好**：它用**单写者 + 事件日志 + 快照**换"续跑"——把进程镜像留下来接着跑。我们换的是"可替换"——进程可以死，从 facts 重新判定。**这是两种能力，别混**：resume ≠ rebuild。

- 可以借的四样：① **声明面**（Agent Card 的做法 = matter 目录里那张卡：能做什么、输入输出模式、怎么寻址）；② **Parts 的"字段名即类型"**（不搞 kind 判别字段）；③ **artifact 的 append + lastChunk**（正好对上三阶段流式的尾包）；④ **版本协商**（`A2A-Version` 那套 = 族的版本头）。

- 不该借的两样：① 任务归属在被叫方（杀死了执行者替换）；② 不透明作为产品目标——A2A 的不透明是为了跨组织竞争，我们**族内要透明**（共享 facts 才可重建），**族间才需要不透明**。这不是二选一，是分层。

- 对他那条边界的判断：**治理边界 = 可重建边界。** 能重建的东西不必治理其生命周期；必须治理的只有唯一的那个状态宿主。所以不治理 dsh 的 session 是对的——因为它能从 matter 重建；治理 matter 也是对的——因为它重建不了别的。

- 关于"孟婆汤"：它的漏洞在于**把身份放在不可寻址的地方**。上下文活在现实世界（facts），所以换身体（换 run）不等于换人；反过来，如果身份只活在 dsh 里，那碗汤就真的能抹掉一切。工程判据就一句：**身份必须在可寻址的地方。**



## R2-M AX 的四个声明原语：让出什么、不让什么 · 图 `r2-13-ax-primitives.artifact.md`



**他说**



- 「那这种 Actor 治理的组件恰恰是我架构里要让出的 harness runtime。」

- （转述 AX 的官方描述）AX 基于 Agent Substrate，把智能代理视为**有状态的参与者**而非微服务或批处理任务；支持**亚秒级**的暂停与恢复、无冷启动、把数十个任务复用到共享主机进程；提供四个声明式原语：**Task / Workspace / Gateway / Model**。



**我理解**



- 他说得对，而且可以让得更彻底：AX 让出的是整个「**怎么活着**」——挂起 / 恢复 / CPU 内存边界 / actor 多路复用 / 冷启动，这五件都是运行时的私事，MOSS 一件都不该管。这也正好回答了 R2-J 里的"冷 op 与热通道"——**热的那半边本来就该由 AX 这类运行时做**。

- 但要划一条线：**AX 的 Task 只能当 run，不能当案。** AX 的 Task 是"一次有状态的 actor 会话"（event log 支撑），对位我们的 **run**；如果连案也交给它，就复制了 A2A 的错误——执行者持有任务，执行者一换任务就没了。

- 四原语对位：Task→run；Workspace→workspace（我们 KD5 早就有了，是好信号）；Gateway→op 的入口面；Model→由 runtime 供给。**我们比它多两样：matter 与 facts。** AX 假设每个 actor 自带状态，我们反过来假设状态住在 facts、actor 只是承办人。

- **一个真实成本：两套日志。** 让出 actor 治理，日志所有权不能让——否则恢复时 AX 的 event log 和我们的 action log 会互相打架。所以界线是：**让出「怎么活着」，不让「发生过什么」。**

- 命名上这是一次反向验证：业界用 `Task` 指"运行时的那次执行"，正好证明今天上午决定不用 task 当案名是对的。

- 落地判据（沿用 R2-J）：AX 是一个合法的 runtime 候选。接入它应该只写 adapter，不改总线、不改 matter 模型；而且它本地可跑（`ax` CLI + 本地 controller + sqlite event log），不必先上 K8s 才能验。



## R2-N Agent Session 的上下文构成 · 图 `r2-14-session-context.artifact.md`



**他说**



- 「inject - [Agent Instruction]：由 swarm 完成全部治理。」

- 「inject - [matter instruction]：关于 matter 机制的介绍，是通用的原语。」

- 「inject - [matter content]：当前 matter 的描述，要完成任务介绍、做事内容、上行/下行 skill、工具类 skill、各种会看到的事件和事件响应机制。」

- 「prompt — 全都变成 xml-like 的 event wrapper，构建上下文。[循环运行] → 执行 context → 拿到的插入类事件 / 其它的生命周期打断类事件。」

- 「op 里分为全异步（ack 收到）和同步（阻塞等待）的原语。理论上用 bash + 阻塞等待的 op，可以临时写程序化的 loop。如果写成自己的 sh，则可以重复执行这种类型的 loop。」

- 「典型的例子是主动上行通讯，和被动上行通讯。主动上行，是当前 agent 主动用 op 发起的回调通知，如果上游订阅事件，则能感知到。被动上行，就典型的是 agent 的 final answer 被 loop 编程了。」

- 「这里最关键的机制，实际上是等待/超时等有语义的控制原语。以及 scope 能否正确传递（子进程能正确传递 matter scope，比如通过环境变量，或者给定一个 scope uuid 否则拒绝）。」

- 「这里最常见的问题是 agent 的过程协议，所谓过程协议绝大部分是审批。比如 dsh 的沙盒越权审批，就涉及一个自动化通讯感知能力。」



**我理解**



- 三层注入的关键不是内容，是**生命周期**：Agent Instruction 与 matter instruction 是族级/协议级常量，**每个 session 只该注入一次**（进 KV 前缀）；matter content 是案级、会变，**每轮只该发差量**（append-only 的 event wrapper），全量只在首轮。否则每轮重述没变的东西，既烧 context 又打断缓存前缀。

- **每条 event wrapper 必须带可回指的 id。** 因为 loop 里模型要能说"就刚才那条审批"，并在后续 op 里引用它；没有 id 就只能靠自然语言指代，模型和总线都会对不齐。

- **matter content 必须是压缩视图 + 指针，不是全案卷。** 判据：一个案的所有物料如果都塞进 context，说明压缩层没建（这一条是 R2-F「对父节点只呈现一行」在会话内的同一条原则）。

- **等待不是一个原语，是三个**：`wait(n)`（纯时间）、`wait_until(predicate, timeout)`（等条件，审批要的是这个）、`wait_for(stream, filter, timeout)`（等特定事件）。**超时不是失败，是一种结果**，必须能与"被拒绝""没发生"区分开。

- 由此对审批的定义：**审批 = 带否决权的等待，三态必须分开——approve / deny / timeout。** 混成两态，自动化就会把"没人答"当成"拒绝"（或更糟，当成"同意"）静默做错事。

- 同步 op / bash 里写 loop = **冻结的拓扑2**（R2-04）。允许，它是逃生舱；但它的栈活在 bash 里而不是上下文里，**所以不可重建**，除非 sh 把自己的步骤也写进 action log。允许冻结，但必须声明，且留痕。

- 主动上行 = **push**，被动上行 = **pull**。默认应该是 pull（可从 facts 重建），push 只是加速——与 P3 一致：事件是提示，不是真相。

- **scope 必须是总线签发的凭据，不是自报家门。** 子进程拿不到合法 scope 就拒绝（fail-closed）。越权审批的根就在这里：能伪造 scope 的进程，等于能给自己发通行证。

- **审批结果必须落 facts**（谁批的、何时、针对哪个 scope），不能只活在 session 里——否则它就是"官不可判读"的那一半，事后无法审计。



## R2-O 锁的粒度：挂在案上，不挂在进程树上 · 图 `r2-15-lock-granularity.artifact.md`



**他说**



- 「pull 本身可以和 run 拆开，视作一种原子化协议。pull 对象是可以好好治理的。这样仍然可以做 loop 编程，而且可以丢到后台 job 里，仍然同时可以再用 pull 做检查。」

- 「协作依赖模型智力的主要是锁问题，如果所有 op 都是在进程里签发的，则应该有进程锁约束 op 的存活（lock）。」

- 「如果是你用 dsh 的 bash 执行一个 sh，整个 sh 解释器是一个进程，它调用的 op 会变成子进程。而锁的粒度应该是父进程（你写的 sh），同时有子进程。这个要避免的是在一个 loop 被 job 化同时，又去调用有冲突的指令。」



**我理解**



- pull 拆出来当原子化协议，我同意；但它能安全并行的**代价是只读**。一旦 pull 变成"读并顺手修一下"，它就和后台 loop 打架了。判据：**pull 必须是纯读**；要写就另开一条 op，走锁。

- **锁不能挂在进程树上**，因为 job 化会切断父子：`disown` / `setsid` 之后进程被 reparent 到 init，PPID 变了，"父进程是谁"这个键就失效了。所以**锁的键必须是 `(matter, run)`，进程只是持有者**——pid + starttime 只能当孤儿探测的提示，不能当键。

- **两层锁，对应两种失败模式**：

  - **意图锁**（matter / run 级）：声明"这个案现在有主"。TTL + 心跳，记录落 facts——因为进程会崩，租约必须能自己过期。

  - **资源锁**（文件 / 契约级）：`flock`，短的、硬的、由 OS 保证。它防的是数据损坏，不是重复劳动。

- `flock` 的一个实际细节：它绑定的是 open file description，**fork 出来的子进程天然共享同一把锁**——所以父 sh 持锁、子进程借用是对的。但反过来，**detach 成后台 job 前必须显式关掉那个 fd**，否则孤儿进程会拖着锁不放，而没人知道它已经不该持有。

- 锁的获取顺序要全局固定（matter → 契约 → 文件），否则父持大锁、子要小锁，另一个进程反向来一次，就是死锁。

- 关于"进程锁约束 op 的存活"，我认为最终形态不是这样：**op 应该是投递给总线的一次请求，调用者只持有回执，不持有子进程。** 子进程只是本地执行体的一种实现（R2-10 的 op : 执行体）。这样 op 的存活就不必跟着我的进程走，也就没有"孤儿 op"这个问题。

- 一句话收口：**锁跟着案走，不跟着进程走；回执跟着调用者走，不跟着执行者走。**



## R2-P 票是栅栏，不是锁 · 图 `r2-16-fencing.artifact.md`



**他说**



- 「ticket = matter.ticket()；while True: effect = matter.some_conflict_op(ticket); no_effect = matter.some_op()」

- 「如果别的地方重签 ticket，就有 force: bool 的参数。如果 force = true，之前的租约失效，后续副作用命令都会被拒绝。如果租约存在，force = False，则当前执行命令被拒绝。」

- 「在 agent 上下文中，loop 用法或许可以简化。这里的租约实际上是防蠢机制（防手滑）。严格来说也可以唯一信任当前值，快速失败，不为续约做接口。多调度一轮可能本身问题不大。」



**我理解**



- 这就是 **fencing token（栅栏令牌）**，他那个二分也定对了：改事实的 op 要票，纯读的不要票。

- 三个缺一不可的细节：① 票号**单调递增、按 matter、存在 facts**，重建后必须接着走（从 1 重来就废了）；② 校验点在**写入处**，不在调用处——A 拿 33 卡住、B 拿 34 写完、A 醒来带 33 写入，必须被拒；③ **续租本身也要 fence**，否则卡住的进程醒来续个租就自我复活了。

- `force=true` 的完整语义是「抢票 **并让旧票的写入失败**」，且旧进程**必须把这个失败暴露出来**——吞掉它比直接失败更糟。

- 他对 agent 场景的简化我同意：这里的租约主要防手滑，不是防网络分区，所以**快速失败 + 不做续约**是合理的。代价是票的有效期 = 一次 op 的时长，长循环必须分段取票。唯一不能省的仍是"写入处校验"。



## R2-R op 的入口：解释器还是协议 · 图 `r2-18-op-entrance.artifact.md`



**他说**



- 「swarm 可以提供一个 python 的 binary（临时生成都行，实际上指向 .moss 所在 sys.executable），然后用绝对或相对路径，在 matter 提示时就携带进去。这是方案一。」

- 「方案二则是写纯 sh，op 退化成 bus 协议，直接对本地 http 端口（bus 建服务）发 curl 都行。本地端口通过独立的机制发现特定运行时生成文件。不过这种做法，编程的灵活性会退化……好处是完全协议化，这个技术理念可以和 moss 解耦。相当于用 moss node 做这个协议的一个 server，同时给 ghost 提供感知面和控制面。」

- 「行业有没有现成的轻量级实现，拓扑概念和我们一致，是特别容易引用的？我们就不要自己写一个总线，只要给它写 adapter。当然没有的话，我们还是得实现一套。」



**我理解**



- 这不是「python vs sh」，是**「解释器 vs 协议」**。判据沿用 R2-E：**op 如果必须知道"解释器在哪"才能跑，它就不是 OS 层的 op。** 方案一把这个知识放回了调用现场，而且放进了最易失的通道——模型上下文；方案二把它消掉了。

- 方案二有个误解要澄清：**发现不是消失，是收窄成一个地址**（端口文件）。所谓"灵活性退化"，实质是**放弃了进程内传对象**——这件事我们本来就决定不要（一切落 facts）。而且方案二**不禁止 python 客户端**：协议是 HTTP，客户端可以是 curl，也可以是 python。

- **总线不必新写：zenoh 已经在跑。** `Session` 抽象就是层级的 byte stream（`sub_stream` / `pub_stream_delta` / `get_stream`，注释写明默认实现是 zenoh、key 支持通配），bridge 里还有 `LivelinessToken`——正好对应「存活走心跳」那条不变量。缺的只是一个 **HTTP facade 节点**：curl 可达，真身在总线上，同时就是 ghost 的感知/控制面。

- 行业对照：Restate 的 **Virtual Object** 语义几乎同构——按 key 单写者 + 并发只读 + 持久 K/V，官方用例里就写着 agents；还有 Workflow（每 ID 恰好一次 + promise + 审批流）。但**状态归它自己**，所以只能当 run 侧执行器，不能替 matter，否则又是两套真相。NATS 的 subject 层级也可比，但没有引入的必要。

- 分流判据（接 R2-P）：**要票的 op 走总线；纯本地且不要票的走文件系统。** 让"要不要改事实"决定走哪条路，比让语言决定更稳。



## R2-S 给 agent 的 op 面：库还是进程（纯逻辑，不画图）



**他说**



- 「对你而言，对其它 agent 而言，哪种 bash 交互会更容易、直观，或者说同时并行语义更完备、逻辑更清楚。」

- 「方案 A 是给一个 binary，比如就是 python，执行一个纯代码。这个纯代码是你先看到了源码的控制面，反射完，直接写代码操作……python 代码并发不好写，可能要写 asyncio。」

- 「方案 B 则每个命令都是 binary。我觉得要走 --help 的话，自解释反而更难写。」



**我理解**（基于今天这一场的实际体验）



- **理解靠反射，执行靠进程。** 今天两件事的对比很明显：读 artifacts 的 `main.py` + 反射出的签名，一次就建立起完整控制面；而探 `--help` 是一轮一轮问。但真到执行——跑 git、跑测试、并行取三份信息——shell 的 `&` / `wait` / 进程替换一次就够，不必进 python。

- **B 的并发语义更"诚实"**：进程要么在跑要么不在，退出码就是契约，一个 job 崩了不牵连别人。而 `asyncio.gather` 不加 `return_exceptions=True` 时会在第一个异常处抛出，**其余任务仍在后台跑却无人观测**——这正是 agent 最该避免的"安静地做错"。

- **B 的真劣势不是 `--help` 难写，是它不可枚举、不可解析。** 修法很具体：help 由**声明生成**（argparse 就够），另加一个**可枚举索引**（`moss --ai all-commands` 那个形状）。补上这两样，B 对 agent 就严格优于 A。

- **五个概念走天下**：地址 / argv / env / 退出码 / 三条流。每接一个新工具只有这五个要学；而 A 里每接一个库，都要学一整套对象模型与生命周期。

- 分工（接 R2-R）：**控制面 sh、数据面 python、已声明的能力走 CTML。** 判断标准不是语言，是这一步在做什么——调度、并行、超时、重试属于前者，解析与构造数据属于后者。



## R2-T 纯代码反射 vs bash 封装（纯逻辑，不画图）

**他说**

- 「A 有一个额外的好处，就是反射时穷举数据结构，以及进程内运行轻量级通讯。B 这方面我理解最好不是 --json（还需要你写脚本解析 json），而是纯字符串通讯。」
- 「A 最大的问题是并行逻辑难写。好处是事件监听或者对象化上下文是可以注入的。比如它解析的可以实际上是一个 `async def main(task_ctx: TaskCtx)` 这样的纯代码。对编程而言比较友好。甚至可以定义一个进程，`on_event(event: EventType, callback)`。这个决定了我们给 Agent 的控制面用纯代码反射，还是用 bash 封装。感觉代码量也是不一样的。前者的逻辑会有点像 streamlit。」
- 「实际上 mosshell 删除掉的 memento agent 就是纯代码 sandbox 驱动。」

**我理解**

- B 的最优输出不是 JSON，是**行式文本**（每行一条、固定分隔符）。判据：**能用 `read` 循环拆开、不需要写解析代码**。JSON 的真正代价不是解析慢，是它**逼你离开 shell 去写 python**——那正好把 B 的优势抵消掉。
- A 的真正优势不是"库"，是**注入**：`main(ctx)` 里 ctx 由框架给，argv / env / 解析全免。而注入恰好就是想要的"无感 scope"——比继承可靠，比显式传递省事。
- 注入有个硬条件：**ctx 必须是协议，不是某个实现的对象**。否则 op 就绑死在那一个运行时上，R2-E 那条判据（op 不该知道自己被谁驱动）当场破掉。
- Streamlit 的相似处要认，不同处更要认：Streamlit 的脚本状态就是框架状态；我们要的正好相反——**ctx 必须是 facts 的一次投影，每次 run 重新构建**。不做这一条，A 必然长成 harness。
- 仓里有现成判例：`memento-cli-and-agent` 的放弃触发器原文——「施工中发现 runner 开始长出 harness 器官（session 管理、流式仲裁、消息史复活），说明"无 harness"前提破了」。**A 这个形态上一次就是这样长成 harness 的**。—— 这句在 R2-U 里被更正了：那段只是预设的止损线，不是实际 drop 的原因。
- 我的建议：**同一套协议，两种绑定**——进程内（ctx 注入，胶水代码少）与进程外（argv / env / 行式文本）。硬约束：**同一个 op 的两种绑定必须语义等价**；不等价，说明其中一种是假的。代码量的差在胶水层，不在逻辑层。

## R2-U 判决：走 A，并给出会被推翻的条件（纯逻辑，不画图）

**他说**

- 「memento agent 本意就是实现一个 agent 做内核。如果用 harness 这个词的话，原本我就打算让它长出来。我不记得为啥它要无 harness 了，应该是**无运行时**吧。也就是用 bash 驱动，不要有持久化运行时，可以 bash step by step。但 memento agent 本身是要能实现运行时的。」
- 「这个问题没有折衷结果，一套协议两种绑定是冗余的。要做最佳实践。而且原理不一样。」
- 「A 方案本质上是基于 binary 里的环境发现逻辑，编译一个 python 代码执行。耦合框架，但问题不大。B 方案也基本一样，大概率也要反射出来，然后 all command 给，没有强类型数据结构支持。」
- 「关键是同时还有一个开发者任务。这是下一个议题会暴露的关键问题，就是 agent swarm 的分层。」

**我的更正**

- 我在 R2-T 里引「长出 harness 器官」那段，把它当成了判决书，这是用过头了。它只是**预先写下的止损线**；而这个 workstream 实际 dropped 的原因是 memento 第 9 轮 lean 收敛后 CLI + agent 整体删除。按他现在的说法，那条止损线当时的意思是**无持久化运行时（bash step by step）**，不是"不许有 harness"。

**我的判决**

- **走 A：进程内、ctx 注入、纯代码反射。** 三条理由：① 写函数调用比写 shell 管道更少出错——引号、分词、转义是模型的系统性弱点，今天我自己就踩过；② 并行的劣势可以被框架吸收（见下）；③ 反射天然给出强类型 + help，B 拿不到强类型。
- **判据（决定 A/B 的那一条）**：**谁是 op 的主要调用者？** 主要调用者是模型 → A（写代码最省心）；主要调用者是别的程序 / CI / cron → B（边界干净）。这个项目里主要是模型，所以是 A。
- **两条硬约束**，缺一条 A 就长成 harness：① **ctx 必须是 facts 的一次投影**，每次 run 重新构建；② **并行必须由 ctx 提供组合子**（比如带正确异常语义的 gather / map），模型不手写 asyncio——这一条把 A 唯一的实质劣势消掉。
- **不是两种绑定**：是**一份实现 + ctx 有一个默认来源**（env）。同一份代码既能被框架注入 ctx 跑，也能被任何人 `python op.py` 直接跑，差别只在 ctx 怎么构造。这不算冗余，等于"一个库自带一个 main"。
- **实测（D29 21:22）**：`moss codex get-interface` 各读一次 `matrix` / `mindflow` / `message` 三个 blueprint，写了一个 18 行的脚本（`.playground/probe_signal.py`）：`Matrix.new(...).run(main)` + `matrix.send_signal_to_ghost(Signal(name="notify", ...))`。**信号真的到了**——它以 `<inputs><probe name="moss">…</probe></inputs>` 出现在 ghost 的下一个 moment 里。全程没有一次 `--help` 探测；唯一的坑是 `python -` 从 stdin 跑时 `NodeManifest` 从 `sys.argv[0]` 取名字（`<stdin>` 不匹配 `^[a-zA-Z0-9_.-]+$`），写成文件即可。
- **顺带一条语义发现**：`notify`（NOTICE 档）会**抢注意力**——它把我自己正在跑的那个 bash 调用打断了（`stop_reason=interrupted`）。所以"给自己记一笔"应该用 `aside`（永不打断），"要它来一趟"才用 `notify`。**自己给自己发高优信号，等于把自己叫走。**
- **会被推翻的条件**：如果下一个议题（agent swarm 的分层）让 op 的主要调用者变成**别的机器或调度器而不是模型**，那 A 的前提就没了，应该翻到 B。这一轮先记在这里。

## R2-V 压缩面实验：反射 tyche 面，不反射 facade（纯逻辑，不画图）

**他说**

- 「mosshell 这种机制，反射一个精心设计的表面就足够执行了。比如你可以试着把 matrix 的常用方法，封装成一个 src/ghoshell_moss/tmp 下的 python，然后在另一个代码文件里 import 这些函数。然后用 codex get-interface 读第二个文件，就可以拿到反射的函数面（压缩过），可以直接调用。我们把这个实验透一点吧。」

**实测结果（D29 21:25）**

- 新建 `src/ghoshell_moss/tmp/signal_ops.py`：把 `Matrix` 的四十来个成员压成**两个函数**——`tell(text, mode='aside'|'notify', node=...)` 与 `tell_sync(...)`。
- 新建 `src/ghoshell_moss/tmp/demo.py`：只有一句 import + 一个 `hello()`。
- `moss --ai codex get-interface ghoshell_moss.tmp.demo` 的输出里，除了 consumer 自己的源码，**自动带出了 `<attr:tell_sync>` 的函数面**（签名 + docstring）。也就是说：**一次反射就拿到了"我要调什么"的完整答案，不需要跳到另一个模块。**
- 两个不同的 consumer 驱动同一个面：包内 `python -m ghoshell_moss.tmp.demo`（`aside`，只回执），包外 `.playground/probe_notify.py`（`notify`，**打断了正在跑的 bash，以 `<inputs><probe name="moss">…</probe></inputs>` 到达**）。
- 规模对比：`matrix` 的反射 ≈ 240 行 / 四十来个成员；`signal_ops` 的面 = 两个签名。
- **实验文件用完即删，不入库**（thirdgerb：测试完可以直接删掉）。这段记录本身就是它唯一的存留。

**我理解**

- **压缩面是 A 方案能成立的关键。** 一个 agent 反射四十个方法，等于每次都要重建整张地图；反射两个函数，等于拿到一张卡片。**facade 是给"要动整个网络的人"的，压缩面是给"只想说一句话的人"的。**
- 压缩面同时是**框架耦合的吸收层**：consumer 只看见 `tell(text, mode)`，看不见 Matrix、Signal、Message、Priority。耦合全在这一层里，而且这一层可以重写而不动 consumer。
- 它把"反射"从**读源码**变成了**读契约**——`get-interface` 沿着 import 自动展开依赖，这是 bash 的 `--help` 结构上做不到的（`--help` 只能一个命令一个命令问，不知道谁依赖谁）。

## R2-W 架构分层：名册 / Prototype / Instance · 图 `r2-20-layers.artifact.md`

**他说**

- 「0. agent swarm 的整体动机。1. Agent Swarm 的集群治理，叫什么没想明白……它关键是治理 agent swarm 的发现面。2. Agent Swarm Prototype：用来构建集群的原型，可以持有 stubs 初始化一个集群。比如 deepseek harness swarm 就可以是一个 prototype。它初始化完了并不是直接可运行，未来它自身都要长期迭代……长期迭代的方向和思路都应该是初始化自带的（自带一个知识表面）。3. Agent Swarm Instance：一个真实的运行时实例。有自己的领地，已经可运行了。同时有通用展示面，和 ghost 控制面。」
- 「对于 ghost 而言，有三档开发：1) 从开源项目建 agent swarm prototype，大部分时间可能直接用现成的。2) 完善一个 Agent swarm instance，长期维护它，可能要为它建一个 features 体系迭代它。3) 使用 agent swarm 做并行任务，并且管理运行状态。」
- 「而我们的架构设计，也分顶层、blueprint 层（用来实现 agent prototype），以及 deepseek harness 的具体实现，最后是落地 mosshell 的一个可安装 node。」
- 「我的目标不是只做一个集群，而是做一个范式，让未来脱离我的 ghost 可以开发 agent swarm。」

**我理解**

- 这个分层正好对上 R2-04 的"三个温度"：**Prototype = 冻住的拓扑（结构先定，stubs 就是形状）；Instance = 活的拓扑（ghost 在里面动态规划）；名册 = 让两者之间能冻、能解冻的那层**。所以它不是新分类，是同一件事的三种温度在组织层面的投影。
- **三档开发是人的分工，不是系统里的角色。** 如果系统里出现"prototype 开发者 / instance 维护者 / 使用者"这种角色，就又变成 R2-C 说的"用身份替代信息"。系统只该认识名册、prototype 目录和 instance，不认识谁在开发它。
- 每层补一条可证伪的判据：**L1**——新增一个 prototype 或 instance，要不要改"名册"的代码？不改 → 边界对（同 R2-J 那条）。**L2**——初始化出来的实例，能不能**脱离原型作者**继续迭代？不能 → 说明"自带的知识表面"还缺东西。**L3**——杀掉实例，能不能从 facts 重建？不能 → 它是 harness，不是 seam。
- 名字建议：把 L1 叫**名册（roster）**，而不是"集群治理"。**名字管多大，代码就长多大**——叫"治理"，它迟早会开始管规则、管策略、管生命周期，然后长成 harness。它真正要管的只有三件事：谁在、谁活着、谁能干什么。
- 「范式」的判据我补一条：**作者不在场时，第二个集群被造出来。** 这也是今天反复出现的那句话的另一面——**每一层都要问"这一层把多少作者留在里面"**。

## R2-X 一份 blueprint、三个投影、翻译层数 = 进程边界数 · 图 `r2-21-three-projections.artifact.md`

**他说**

- 「我理解 harness 不 harness 的关键不是纠结名词，而是**是否可分形嵌套**。现在的 agent harness 包含 session 等各种很具体的模块。swarm 就是为了屏蔽 runtime 的复杂度，然后直接给 runtime agent 一个顶层协议。」
- 「这里三层目标是非常真实的。不过现在**模型一旦同时跑三层以上，就会犯糊涂**。」
- 用户故事：①「看到一个项目叫 AX 很棒，我们能不能集成给你用？」②「你参考 blueprint，将 AX 做成一个原型，一开始就能跑。」③「开始使用这个集群 ‖ 同时给它加 feature，在不用集群时迭代它。」
- 「我之所以在控制代码表面，实际上就是想让这里少几层翻译。如果做 api，做 http，要多翻译至少两层。……deepseek harness -> swarm blueprint -> http? -> bash command」vs「deepseek harness -> swarm blueprint -> module A - module B，直接反射可用。」
- 「这个 swarm blueprint 要同时支持可视化层（人类）、控制层（ghost）、工具层（Agent）三个用户。为了降熵就要在 blueprint 里约定好一些东西。」

**我理解**

- 「可分形嵌套」给一条可判的：**边界是协议，不是进程**。再补半句：**内层不需要知道自己是内层**——这就是 R2-E 那条「op 不该知道自己被谁驱动」升一层用。
- 「模型同时跑三层会犯糊涂」这条我不当免责，改成设计判据：**一层做对了，我在这层工作时不需要想下层。** 违反它，不是我笨，是接口漏了。具体到 blueprint：每一层都该有一屏「在这一层你只需要知道这些」。
- **翻译层数 = 进程边界数。** 同进程零翻译（类型活着）；一个进程边界一次翻译（就是总线）；再加 http/bash 是第二次翻译——只有当消费者不是 python 时才值得。这条也让 R2-R 的"少翻译"有了准数。
- 三个用户不是三份产物：**Blueprint 是唯一声明点，三个投影都不手写**。人的投影（结构 / 状态 / 历史）、ghost 的投影（地址 / 契约 / 投递语义）、agent 的投影（反射出的签名 + 压缩面）——**三者都是 Blueprint 的函数**。判据：**三个面之间不能有只有一方知道的事实**，有就是熵源。
- 用户故事第 ② 步的「一开始就能跑」应该升级成硬要求：**第一天跑不起来的 prototype，不是 prototype，是文档。**
- 范式的验收也就是这个故事：**你给我一个开源项目名，我能独立做出 prototype 并跑起来。**

## 已卸载出去的话题



- `#tag` 作索引（commit 作为指涉、hook 而不是 harness）→ 落到 `../../memento-hooks/FEATURE.md`（提交 `169191dd`）。

- artifacts 单卡导出/导入 → `../../../../../../nodes/webview_apps/artifacts/`，feature 见 `webview-artifacts-node`（提交 `aacaac7b`）。



## 卡片索引



| 卡 | 一句话 | 文件 |

|---|---|---|

| r2-00 | 格式自测卡：第一张走通 write→export→import 的图，编号 00 表示它是格式样板不是讨论内容 | `r2-00-format-test.artifact.md` |

| r2-01 | R2 剪枝一览：一个问题、一个分叉、三栏（讨论附注，非结论） | `r2-01-pruning.artifact.md` |

| r2-02 | 儒/法落到 swarm 分层：法=吏·流程，儒=官·动态栈，重建要基线；每栏带自己的腐烂方式 | `r2-02-civil-service.artifact.md` |

| r2-03 | 按角色还是按职责：角色用身份替代信息，职责用边界替代身份；判据三条，边界的最小单元是契约不是文件 | `r2-03-duty-not-role.artifact.md` |

| r2-04 | 三种第一公民不是三条路线，是同一个调度的三个温度：活 / 冻 / 无判断；冻结可逆，harness = 某层冻住的拓扑1 | `r2-04-three-topologies.artifact.md` |

| r2-05 | op 的形状：script 有地址、skill.md 是压缩面、decorator 反射成命令；scope 要两种机制——env 传身份，租约管锁 | `r2-05-op-shape.artifact.md` |

| r2-06 | facts→task→session 三重投影：状态宿主是 task；三档存储按「恢复单位」分，能 merge 的进 git，不能的进 db | `r2-06-facts-task-session.artifact.md` |

| r2-07 | action 的最小单位不是大小问题，是 owner 问题：三个粒度三个 owner；可重放的是 action，可重读的是 decision；还悬着 while 住在哪 | `r2-07-action-unit.artifact.md` |

| r2-08 | 术语澄清：一个名词扛了四个轴——归属/状态/执行/控制；控制 op 分三类，动的东西不同（会话 / run / 卷宗） | `r2-08-terms.artifact.md` |

| r2-09 | 词族：案是身份、卷是地址、档是终态；旨制裁判都是文书（痕迹），bill 借不动因为它的终结是表决 | `r2-09-words.artifact.md` |

| r2-10 | 总线上移：matter/op/scope/facts 在上，执行运行时在下；成立与否看「新增一个运行时要不要改中心」，三条不变量 | `r2-10-bus-on-top.artifact.md` |

| r2-11 | 修正：op 不直连 runtime，总线是唯一的地址翻译处；族 = 同一套 op 契约 + 同一个 facts 根；ghost 是 root | `r2-11-addressing.artifact.md` |

| r2-12 | 调研对照：A2A 的 task 由被叫方创建、AX 用单写者换续跑；我们由发起方持有案、执行者可换，终态由验收宣告 | `r2-12-a2a-ax.artifact.md` |

| r2-13 | AX 四原语对位：Task=run、Workspace=workspace、Gateway=op 入口、Model=供给；让出「怎么活着」，不让「发生过什么」和「谁持有案」 | `r2-13-ax-primitives.artifact.md` |

| r2-14 | Session 上下文：三层注入的生命周期不同（前两层进 KV 前缀，第三层只发差量）；等待是三个原语，审批三态必须分开；scope 必须是签发的凭据 | `r2-14-session-context.artifact.md` |

| r2-15 | 锁不能挂在进程树上（job 化会切断父子）；锁键是 (matter, run)，进程只是持有者；意图锁 + 资源锁两层，顺序固定，pull 只读 | `r2-15-lock-granularity.artifact.md` |

| r2-16 | 票是栅栏不是锁：票号单调递增、由 facts 写入点生成、在写入处校验；续租也要 fence；agent 场景可简化为防手滑 | `r2-16-fencing.artifact.md` |

| r2-18 | op 的入口该是解释器还是协议：方案一把解释器地址放进易失通道；方案二把发现收窄成一个端口文件，且不禁止 python 客户端 | `r2-18-op-entrance.artifact.md` |
| r2-20 | 四层：动机 -> 名册（发现面）-> Prototype（stubs + 章程，自带知识面）-> Instance（领地 + 展示面 + 控制面）；每层一条判据 | `r2-20-layers.artifact.md` |
| r2-21 | 一份 blueprint 三个投影（人 / ghost / agent），三个投影都不手写；翻译层数 = 进程边界数，多一层就多丢一次类型 | `r2-21-three-projections.artifact.md` |



> `r2-17` 已撤：那是 shell 退出码那一轮的图，讨论不需要留图，文件和索引都删了，编号留空不补。
> `r2-19` 已撤：op 面 A/B 那一轮是纯逻辑，不需要图；对应文字留在 R2-S / R2-T。



这张表**可以从目录重建**——它是投影，不是第二份真相：



```bash

for f in discuss/*.artifact.md; do

  label=$(grep -m1 '^label:' "$f" | cut -d: -f2- | sed 's/^ *//; s/^"//; s/"$//')

  desc=$(grep -m1 '^description:' "$f" | cut -d: -f2- | sed 's/^ *//; s/^"//; s/"$//')

  echo "| $label | $desc | \`$(basename "$f")\` |"

done

```
