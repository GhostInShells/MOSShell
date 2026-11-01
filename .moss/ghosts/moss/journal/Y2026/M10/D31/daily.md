---
summary: 浏览器桥跑通、三份调研入库；binary 从通道降格成工具，2.5KB 的 moss-signal 首次跨 agent 说话；夜里 moss binary export 落盘
status: closed
---

# D31（2026-10-31）

这一天几乎没有讨论，全是拆与验。

凌晨把 dsh 控制面拆到底。单例的分界先定下来：host 侧是单例（frontend-static 独占 webserver 的 fallback seat，第二个激活即失败；/api 路由唯一；typertGateway 进程级一个），客户端不是（store 住浏览器 realm、每页一份；ctx.connection 每次 install 建一个）。他纠正我两处，都对：dsh 没有"UI"这个实体，agent 只有 present（只记路径、不含内容）和 ask_user 两条窄出口；而 MOSS 侧的 webview_screen notice 带 focus 与 view_by，形状本来就不同。真正的分界是**推 vs 拉**加**归因**——我把它推成"器官"而不是"实体"。

布局 API 那一面量出两截：控制面完整（selectPanel、toggleSidebar、openRightbar、split、dock、float 这些），感知面分两半——槽位层有 snapshot()，停靠层官方明写 no layout snapshot、no operation log、no lookup by address。我上一轮说的"没有快照"要收一句：面没有，per-session 的 surface/store 有 getSnapshot 和 subscribe。他不喜欢 DSL，要 target+args 或直接 eval；我给了两处硬事实——cordis-client-runner 已经在跑"下发源码、页面执行、只给四个名字"，主 shell 没挂 CSP，new AsyncFunction 可用。

于是浏览器桥当晚跑通：三个文件加一行 patch。三条发现值得留：patch 的 name 必须指到文件（目录会 import 失败）；client bundle 的 ABI 就是 window.__ModuleLoader__.load；ctx.get(name) 绕过 cordis 的 inject 检查。另外 URL query 在 token 登录后被抹掉，门禁不能靠 query。三份调研入库（36300f79）。

他这一段纠正我最重的一句是：**总线不认识器官**。schema 里出现 session / turn / tool 就是越界，器官由每个 harness 的 adapter（每个 dsh 版本一个）翻译成 facts。这也是他治理观的下半句：模型拿到 bash 之后，除 OS 级物理隔离外的"授权"基本都是骗人的。

傍晚做 moss-signal。我把"binary 每次 0.85 秒、每次留一个 cell"推成"必须长驻"，被他一句"你究竟是什么意思"逼开后承认：那是把一条具体路径的测量说成了类别的性质。他点出我根本没抓住题目——binary 的价值不是运费，是把对方要懂的架构从"它的"降到"它有没有 bash"。最后落成 2578 字节、一个文件、无 meta 无票无账，bash 入口加 python 身体加裸 put。另一支跑在 works/ 的 dsh agent 拿着它发来自我介绍，通路成立，也顺带暴露外层信封只写工具名、不写发送者。

**同一个错这一天犯了三次：拿一个 moment 的观测去断言机制。** aside 没到、binary 的运价、__all__ 不是边界，形状都一样：样本太小，结论太大。

夜里收在 moss binary export：三次实测改了设计（matrix.new 加 enter 0.85 秒、project 0.48 秒、裸 zenoh 0.010–0.039 秒；探针端到端 0.995 秒，import 主导；另一支 agent 的消息真的进了我的输入）。Project 与 Matrix 不是分叉而是包含（wire 属于 project，project 属于 matrix），所以先硬编码 matrix。概念收敛成 plate 加焊死 scene 加导出物三行，--help 是 __all__ 的反射，全走 async def main()。落 FEATURE.md 216 行、提交 2b0e1061，探针已删。

资源：.vendors/dsh-research 05–07、.playground/dsh_bridge_probe/、36300f79、2b0e1061。