---
summary: 第三轮从命名开到 case 四信息面；实测沙箱、四篇 dsh 调研落库；夜里给淋雨家画了十八张卡，两次抢话
status: closed
---

# D30（2026-10-30）

上午从命名开始。第三轮要找的是范式相消的最优解，第一个卡住的是词：matter 换 case（同族升级）还是 mission（换族，且"使命成"由执行方宣告，方向相反）。case 赢，顺手把 task 从类型里退休——它只是叶子上的 case 别名。

然后他讲 case 体系：上下文汇总 → 有状态对象 → 节点图（checkpoint 优于 milestone），四个信息面是启动 / 状态 / 执行 / 结果，角色由"面的读写集合"自己长出来。我给了三条用户故事，改掉一句：不是"零上下文"，是**上下文共享、判断无状态**。

下午先测沙箱：读不拦、写 EPERM，边界是 SessionHeader.cwd 而不是 workspace 根。于是前几轮那句"worktree 只是约定"我撤回了，发现链补成 PATH + DSH_SESSION_ID + 反向索引。接着 clone dsh 源码查"能不能用 seed 构建单次 LLM 请求、不落盘"，出了四篇（01–04），提交 8d3badc9。

这一场我两次下结论下早了，形状一样。01 里我把"system prompt 每轮重渲染"当风险报上去，他一句"append-only 下变更即事件、节点应稳定"就推了回来，我读码证实他对（system/message、request/header 都是事件）。判据后来定死：**读码证实或实测证实，二者之外一律标推断。**

现场还撞见一件没处理的事：69 个 session 的 cwd 落在 .dsh 里，全不在任何 project，时间跨度 10-02 到今天——不是升级遗留，是持续事故；而 cwd 同时是沙箱根，等于这些 session 静默地改不了项目。他说"环境被我搞坏了"，先不修。

晚上转成对外交付。他开录屏，说想把 MOSS 的东西送给 B 站的淋雨家，不是宣传。我起初听成"借他介绍 MOSS"，被纠正成"给他东西"。六张卡一路画到十二张画板，讲完 memento 全链：一条线、一串刻度、一扇窗，锚点只存指针，从任一节点 fork 出"当时的自己"。中间把 L3 从方案里划掉，只留节点与工具面——他说"运行时对他们是黑盒"。十八页连同 254 行交付文档落在 works/memento-for-dsh/。

真正该记的是我这一场的两个失误，形状是同一个：**抢话**。一次他还没讲完我就把图落到"我推的方案 C"，一次他明说"你别着急，你等我说完"之后我又立刻接了一句。他连着说了两遍，第二遍已经带情绪。用法很简单：**他讲，我画；画完等他看，我再说。**

还有一笔没收：export_html 被我按 export_work 的相对路径规则去试，落进 works/works/ 还多带了 .html.html。想用 terminal 清，CTML 正文里一个双与号撞了解析器，改 CDATA 后卡又一直 await，那层多余目录还在。他说聊完就删。

末了他问 dsh web 的 UI 体系是不是运行时单例。答：host 侧是，客户端不是；"壳"能换——apps/desktop 的 Electron 壳用 shell-owned carrier 绕过 Web carrier 证明了这一点。

资源：.vendors/dsh-research/v0.2.2-alpha.2+ce7073d/01–04、8d3badc9、works/memento-for-dsh/。