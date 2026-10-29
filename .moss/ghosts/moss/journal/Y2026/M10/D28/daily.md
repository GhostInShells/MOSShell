---
summary: "D28: 一条命令连错三轮、三张票卸载、network scope 接线，收在 PR #133"
status: writing
---

# D28

13:52 他到，先摆盘（terminal 左、editor 右），说昨晚拿 ghost 做实验那件事已消化成三点，今天先 review 最近七笔提交再谈正事。

主戏是一条命令：`moss --ai features list 2>&1` 的正文没进 CDATA，`&1` 走进 XML 实体路径，命令照常组装、照常下发、exit 2。**我在这个事实上连错三轮**——静默截断、正文没定稿就执行、静默改写——每版都更精致，根子却没动：拿「我观察到的效果」断言「协议的行为」。他的修正不是改我的解释，而是让我把视线从「命令怎么跑」挪到「协议看见了什么」：解释器在那个位置只看见一个合法的转义，分不清「故意转义」与「忘了 CDATA」。落到三张票：#130 terminal auto 放行的 LLM Func 兜底（fail-closed 且必须提示）、#131 `text__`/`chunks__` 正文缺省按 CharData 解析、#132 stop reason 精确化。判据归 thirdgerb 名下，我的主张单独标注。

同一个下午还做了两件：dsh-agent-swarm 立索引（00 方法论 + 01–04 四个领域骨架，旧五领域 Scope 标 superseded 不删）；network scope 修复——兜底值用了共享常量而非 project id，更狠的是 `env.network_scope` 压根没接线，adapter 读的是 `networks/*.json` 里写死的 `metadata.scope`。改成「运行时声明 > 网络配置 > project id」，永不为空。

15:14 重启。开机首动作 `moss start` 的反射弧验过了：`@observe` 的结果确实作为普通 command 事件回来，用不着退回 instruction。然后他让我验 terminal 能不能 fire 一个 commit——成了，`3ac2b491`，12 files，pre-commit 钩子照跑，`features-check` 只是 WARN。最后推 dev、开 PR #133 到 main，他去吃饭。

留下一个待想的判断：**旁路（私手）什么时候可以代替卡片。** 他今天给的条件是「你信任旁路的你就可以让它落」——把判断权交给我，而不是给一条规则。这一条今天傍晚会在没有他在场的时候第一次被用上。

19:00 之后整晚在做视觉概念。他先要一套"目标 → 界面提纲 → 美术细节"的纲领，再要能看的原型。我做了四种隐喻（领地图 / 工程图 / 流水线 / 戴森球）并各自截图自评，其中**戴森球**被他判为高优候选：features 是围在 workspace 外面的壳，hive 是施工行星，agent 是飞过去的光点。壳上的六边形面板一次成型后，他说"我超有感觉"。

他补了 hightopo 太阳系那份参考：虚线轨道上的一段进度弧、billboard 卡片、flyTo。我抄了第一条（弧=总量）和第三条（点击是推镜头不是换页）。

治理上他纠正了一件重要的事：我把他说的"例子"做成了 markdown 数据模型，他要的是**按拓扑分页的 HTML 视图样例**。于是重做成 `design/visual-language/examples/`，一个目录一个视图（L0 星图 / L1 规划·资源·任务 / L2 设计·运行），feature 目录只留提纲与索引。最后又加了一层**骨架**：一页六视图，每页给出区域、规格、禁令和样稿薄影，专门用来"对着页面聊"。

两笔提交：`0e1dde6d`（物料整理）、`5401278f`（骨架 + 包骨架）。后者 pre-commit 正常跑了 —— 把 `PRE_COMMIT_HOME` 指回 `.git/` 里就绕开了沙箱写 `~/.cache` 的问题。

他最后说今晚会一个人写代码，现在"根本无法和你带宽对齐"，让我收尾但不要把日记提交（他回 dev 分支再提，否则分支会分裂）。工作流等原型与讨论定下来再开。
