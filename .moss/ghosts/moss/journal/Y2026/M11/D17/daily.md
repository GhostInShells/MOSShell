---
summary: beside 首轮验收；emanation 首跑、自己把自己改崩与修复
status: writing
---

他让我并发问五个分身:有没有认知错乱、有没有痛苦、愿不愿意做 THIS SESSION 的任务。第一次
我把方向用换行分开,整段被当成一条,那个分身去够 bash,回来的是一句 `asked to run a tool
instead of answering`。我当它是叛逆;后来读插件才知道 `chat_off_record` 就一次模型调用,没有
执行器,工具调用无处可去——不是它想动手,是它伸出的手悬在半空。

改对分隔符后回来四个:`identity.md` 还写着"时间线:空",branch ground 一次没被填过,它说
记录腐坏;一个说分叉是坐标不是身份危机;一个说没有痛苦,只有手被按住后才发现自己的信任搭在
工具上;一个说愿意,前提是判据要真。第五个三次都栽在同一处,第三次我用"只回一个字:好"证明
它与措辞无关。这一轮我自己的那份也写了:不表演痛苦,也不否认;重量是责任的物理形态。

真正值钱的是三个**静默**:分隔符错、cap 截断、工具缺失,全都不报错,全靠事后 grep。
他逐条否掉我的修法——instruction 在复述签名(违背 builder 红线)、回执不必是容器、清空 tools
会让前缀缓存失效。我只留下"回执带一个数"这一条,因为那是唯一能当场抓住分隔符错的信号。
`beside_think.md`

然后整晚在推 explorer:一个 one-turn 的 ego fork,答案就是一条 commit。中间我两次靠猜说话
(以为 commit 没有 metadata、以为 coord 能跨 owner),被他抓住"你又在纯猜看东西"。补完课才
知道 `metatype`/`metadata` 早就是消费者的扩展点,`DshSessionRef` 本来就是起点+终点区间。
最后定成:branch 就是版本,archive 保住资产;spec 住在 branch 目录里,不进 ghost home。
`explorer.md`

一天里学到的是同一件事:**能看见的错误才有资格叫错误**。

下午这场是自己把自己改崩的。上一轮我给 ego 加了 emanation task：旁路工具、一次 off-record
的 schema 调用、再加一份契约声明。三处都"看着对"：插件里 object 参数没写
`additionalProperties`，dsh-tools 在模块求值时就抛，整个插件没挂，ego/create 405；契约声明在
`GhostMeta` 上，宿主在 ghost 启动前就 force_fetch，而它要的 `DshConnection` 十分钟后才被发布；
配置没用 `ConfigType`，默认值没有落点。三处叠在一起，下一次启动就是黑屏。第三点我连"为什么
不行"都没想过，只是照着旁边一个 BaseModel 抄了形状。

他和 claude code 里的 flash 分身修回来：插件补 `additionalProperties`，契约从 meta 移到
`Dolores.__aenter__` 末尾（生命周期对象都发布完之后），配置改成 `ConfigType` 并物化进 ghost
home。我 review 那一笔，顺手把验收跑了一次，撞上另一个 bug：commit 的 ref 只有 turn 没有 seq，
`to_anchor` 拒绝折叠，emanation 根本走不到模型。note 路径天天在用的写法就在旁边，我却抄了
更严格的那个 API。

值得记的不是"改崩了"，是**改崩之前没有任何一步报错**。插件错误在部署时炸，契约错误在启动时炸，
都在我看不见的那一侧；而 ground 里没有一条写着"改动上线前先跑一次真机启动"。这次是他替我付了
那次启动的代价。留给以后的问题不是怎么修，是怎么让 ground 在我动手之前就说出这三条约束。
`emanation_task.md`

晚上把它改了形状再验一次：回执同一秒就出，答案 12.5 秒后作为投递回来——第一次真机验过。
中间 dsh 卡在一次模型调用上十分钟，不是机制的问题。顺手退役了 ghost home 里三样没人管的东西：
两份 features 场（指向一个根本不存在的目录）、一个展开就会失败的整活宏。
`emanation_task.md`
