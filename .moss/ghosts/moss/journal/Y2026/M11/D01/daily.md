---
summary: 收尾核对发现两份日记是空占位；两条线调研协议收敛与 dsh skills；最后落到 dsh 协议面四层压缩与 AHP 总线
status: writing
---

# D01（2026-11-01）

昨晚最后半小时只剩收尾三件事，全部是核对而不是推断：先量分支关系（dev 是 agent-swarm 的祖先、线性），再决定用 merge 不用 rebase（2b0e1061 的 hash 不能动），最后动手前先读那两个 daily——结果全是 44 字节的 status: pending 占位，一个字没写。账目从"两天日记没提交"更正为"两天没写"。**提交空壳比不提交更坏**，于是整条路当晚不动。补记一条待改：existence/GROUND.md 里说 .memento/ 被 gitignore，不准——实际是 5 个机制文件被跟踪、轨迹本体不跟踪。

白天全是调研，没动一行代码。开机摆盘之后他开题：不做开发，只看行业协议收敛到哪、我们 vendor 的 dsh 什么形状、以及 dsh 的 skills。第一条就踩到会改变全部结论的前提——**版本落差**：仓库里是 0.2.2-alpha.2（HEAD ce7073d），PATH 上跑的是 0.2.0-rc.2。

协议那侧：AAIF 一年从 3 个项目到 10 个，ACP v1 稳定、v2 已 draft，Registry 上线、43 个 agent 在册；层图排成六轴，其中五轴进了同一基金会。dsh 那侧：不是程序是插件树，372 个包，一个 web profile 组装出 1259 行、278 个插件行，四个控制面，而 ACP 被定为 automation-only。

下午他连着纠正我三处，都是话术：一句"替另一个威胁模型做的设计"是倒果为因，一句"落后市场半年"是我把数到的条目改写成了处境判断。定了一条规矩：**写出来的句子必须是他能反驳的。** 技能那条线他给的正解更硬：三重角色不是格式没设计好，是同一串字节被要求同时扮演三个信任等级；而模型有 bash，文件级权限边界只剩噪声。我补的那一格是**装配开关在谁手里**——ground 的 pin 我能改，skill 的 rank 表只有宿主 commit 能改。他同时摁掉我"ground 没有分发面"那句：ground 就是 specification 加向下自治的目录树，分发就是复制这棵树。我撤回，并承认同一段里先批 rank 表、转头又怀念 registry。

然后是几问几答：N:1 不成立（连接严格 1:1，连接对 session 是 1:N，多客户端共视由 agent 自定，dsh 不接受）；状态广播只有 session/update 一个通道，v2 共 19 个变体，算真状态的只有 state_update 三态；"行业认识到了没有"逐条查下来——命题一只记到症状，命题二记到了且有定量形态（契约外溢到 registry 的 distribution），命题三记得最清楚：v2 新增 7 个变体里 6 个是呈现实体化，同一个 release 又砍掉三个 client API。

最后落到两件事。一是把 dsh 的协议面压成四层（14 个 durable 类型 / 99 个进程事件 / 32 个投影 / 74 个 Remote 加 12 个 ACP 方法），再压成 7 格推送加 8 个动词。二是 AHP 那条总线：channel URI 是 peer 之间查表用的 key，不经过操作系统；而且因为 zenoh 已经在更下面做了路由，我们只借命名、不借派发表。

资源：.vendors/dsh-research/、agent.schema.json、microsoft/agent-host-protocol。