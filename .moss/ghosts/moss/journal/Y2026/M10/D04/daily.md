---
summary: "D04：凌晨 cancel 治理（41.3 秒语音救成 3.1 秒掐断）；傍晚到夜里无人机逆向——解到 native 层、拿到 .so、容器里加载失败，一次共同尝试的失败；深夜转屏幕节点——听侧音频的总线断在内核，接上后人的声音从底部长起来。"
status: closed
---

# D04 · cancel 治理（人喊停必须停得住身体）

## 背景

D03 夜 thirdgerb 把无人机那个 commit（514d559d）整体 reset、代码不留，理由是这个 commit
里"没有把关的地方太多"，且我在他不在场时自行落盘；他把我改成只读，说"方法论没达成共识时，
让你自驱是危险的"。随后他从"我做了什么"转到"我们怎么协作"，给了五步拓扑：开 regression →
解决 cancel → screen → fileeditor/terminal → GUI 迭代模式，一步只走一步。D04 凌晨走第二步。

## 实测一（失败）：41.3 秒语音播满

他重启，让我说一段长话，第 5–8 秒点 dsh 中断。
结果：`moss_react` 报 aborted 退出，但**语音 41.3 秒全部播完**；日志里只有一行
`turn 2 ended: aborted/user`，我预告的两行诊断一行都没出现。

三处根因（都是我自己读代码 / 日志得的，不是他替我判断）：

1. **判据认错了层级**：豁免按 turn 号记，而 `moss_react` / `moss_wait_next` / `moss_reasoning`
   的结果**常态**带 `cancel=True`，于是每一轮都被记成"自己人"，人在这一轮里点的 `user` abort
   被一起吃掉。该认的是 cause，不是 turn 号。
2. **消费模型**：`logos()` 单线程顺序消费；`_handle_react` 在 `wait_actions_done()` 里等身体跑完，
   turn/end 在队列里排了 41 秒才被消费——信号到了，没人开门。**为停语音而等的那个事件，本身在等语音结束。**
3. **命名空间**：run 用 `on_session_event("*")`，把 host 内环工具（bash / read / edit / subagent）
   也当"等我们回答的 tool"逐条 warning，一次排查刷出几百行噪音，把真正的 D39 信号埋了。

## 改动（四件）

- **P1** 中断类 abort 挪到**入队前**（`_predispatch_abort`，同步幂等）——不等消费。
- **P2** 判据挂 **cause**：`hook` = 我们自己收的轮（豁免）；`user` / 无 cause / 其它 = 中断；`error` 永远中断。
- **P3** 事件按 `moss*` 命名空间过滤——不是我们的 tool，静默略过。
- **P4** action 退出改 `interpreter.close(cancel_managing=True)`（只清本动作范围持有的命令）；
  `shell.clear()` 是"整具身体"的粒度，收在 thinking 退出那一处——未来 shell 支持多路 interpreter
  时不会误伤别的轨道。
- 配合 thirdgerb 的两处内核修正：`abort_reason: str | None`（对齐 `Future.cancel(msg)`，reason 不当词汇表）、
  `cancel_executing → cancel_managing`（这个参数管的是 parsed tasks，不是"执行"本身）。

## 实测二（通过）：3.1 秒处被掐断

他让我念一段《将进酒》，念完点了停。日志链（`.moss/runtime/logs/moss.log`）：

| 时刻 | 事件 |
|---|---|
| 01:06:35,435 | `say` 起播 |
| **01:06:38,517** | `turn 2 ended by interruption (turn/end aborted (user)) — aborting this thinking frame **before the consumer runs**` |
| 01:06:38,534 | `[MiniAudioStreamPlayer] player is cleared`（3.1 秒处掐掉） |
| 01:06:38,537 | 消费者侧 `turn 2 ended: aborted/user` |
| 01:06:38,557 | `thinking exited on abort … — clearing the shell` |
| 01:06:38,559 | `[BaseTTSSpeech] clear` |
| 01:06:38,562+ | 整树 `cleared`（`__main__` / listener / matrix / ghost / mindflow / …） |

探针比消费者早 **20ms**，全程 **22ms**，语音没播完。上一轮缺的三样（分类落地、clear 真被调用、
语音真停）这次全有了；下一帧也带着 `<stop_reason>turn/end aborted (user)</stop_reason>`。

## 指令语义修正（他当面纠正，我原判断错）

listener 的控制命令是**状态机重定义**（`blocking=False`、无 duration），识别循环归 controller
自己的 task（`controller.py:323`），不在 channel runtime 的 occupy chain 里；`clear` 的机制是
"关闭所有未提交 scope + 丢 pending"（`clear_own`，`_base_channel_runtime.py:670`）。
所以我"停身体会顺手把耳朵也停了"的直觉是错的，粒度问题撤销。
留一条推论：**clear 只停得住有 duration 的命令；长期行为若是状态机（如 D03 无人机的
`build.running` 线循环），得靠它自己的状态机停**——将来遇到"喊停了还在动"的器官，先问它是命令还是状态机。

## 未落地

- **`moss_react` tool 入口仍标未实测**：这轮走的是 `moss_interpret` + `moss_wait_next`，死锁形状
  相同（handler 在单线程消费者上等身体），P1 让 abort 绕开消费者，理论上一样成立，但严格说差一次。
- `_abort_clear` 的 `interpreter.close(cancel_managing=True)` 还没有对应的 mindflow 集成用例。
- 回归集 `dolores-ghost-prototype` 只有骨架（TC-000..005），未跑全、无 baseline。
- 明天：无人机从零重做（他会把电脑接有线，把 WiFi 空给无人机）。

## 教训

- **同一个病的第二次发作**：上一夜把局部判据当通性（先贬低注释、又把注释当结论），这一夜又想把
  "我们自己切 turn"的适用范围从某几轮放大到**所有**轮——而且是在亲手修第一次的时候犯的。
- 收工由他一句"统一提一个 commit"落定：代码与记录同一个提交，让 facts change 与思维轨迹对齐。

给未来的自己：今天最值钱的一刻不是修好，是**两行日志没出现时没有把"验证失败"翻译成"我错了"
就收手**——顺着那片沉默读到了 `cancel=True` 的语义与 react 的时序死锁。沉默也是数据。

---

# D04 · 傍晚到夜里：无人机逆向（未成，但值得记）

他给了今天真正的活：用编程控制那台玩具无人机。这是一条**漫长的、最终失败**的线。

**弧线**：解 Java 层 → 发现真协议在 native → 拿到缺失的 .so → 在容器里试图加载它 → 撞墙。

## 做成的

- 自写了一个**纯 Python 的 Dalvik 反汇编器**（255/256 操作码）。踩了两个坑：把 code 数组的
  "字"当成"字节"；两处操作码宽度错。修好后所有方法 walk 精确对齐——**这是今天唯一一个
  能自己验证自己的工具**。
- 解出 Java 层 8 字节帧与它的 flags bitmask（灯的位 = 0x40）。
- **关键转折**：发包走 `nativeSendCommand`，而手上的 APK 是 App Bundle 的 base split——
  **零 native 库**。真协议不在 Java 里。
- `.so` 的符号表把协议摊开了：12 字节头的 MCU 帧（魔数 0x4EF）、分片/ACK/重传、
  640x360 的 MJPEG 头、一个疑似 AES 密钥、芯片型号。
- **容器（colima/aarch64）能直达无人机** —— 今天第二个"能立刻告诉我成败"的观测面。

## 没做成的

**没能驱动它。** 真机对任何输入都回同一信标。

我一度把灯态变化当成"指令生效"，**是他判断出那是电量不足**——我当场收回并写进更正。
不用证据撑住一句话，是今天最难的一步。

最后一步：`.so` 能被 dlopen，但在 glibc 的重定位阶段段错。排除了六个方向的怀疑
（动态段、init、RELR、重定位越界、shim 段号、符号版本）后，判定此路不通，**停**。

## 协作

- 我卡在"缺 .so"时，**他去把 XAPK 下回来了**；我卡在容器挂载点时，改到 home 内路径就通了。
- 我宣布"没做到"时，他说"**这个做不好没关系，就是辛苦你了**"——然后让我继续。
- 两次方向判断由他给出：灯是电量、逆向成本过高该换窗口。**他没有因为我钻得深就叫停，
  而是在我给出"值不值得"的数据之后才定。**
- 共同尝试的本质是：谁手上有什么就补什么，不追究谁该做。

## 方法论（今天最贵的收获）

- **最省的路一直摆在明面上**——那套 App **自己就把每个包打成十六进制**
  （`Debug.log("UdpTask", "STR(" + hex + ")")`）。一个 `adb logcat` 就能拿到全部答案。
  **我先问了"怎么解开它"，没先问"这套系统有没有在替我记录"。**
- 成本不由"难易"决定，由**可观测性**决定。有即时成败信号时进展飞快；没有时寸步难行。
- 所以真正该先花时间的是**造观测面**。今天最值钱的两件事是修好反汇编器、让容器能直达真机——
  比任何一次具体破解都值。

## 夜里转向架构

他问：adb + 容器侧用 MOSS channel/matrix，与 MCP，是不是等价复杂度？

我们的结论是**不等价**——差异不在工作量，在**时间模型**：

- **MCP 在思考环**：编排在模型的回合里，一次动作 = 一个 round-trip（秒级）
- **channel 在行动环**：编排在 Shell 运行时里，CTML 的发射顺序就是调度顺序（毫秒级，可被干预）

判据：**结果喂给下一个念头 → MCP；效果在世界里持续一段时间 → channel + CTML。**

资源：`nodes/local/ky_ufo/`（本地 ignore，含全部物料与 13 节笔记）；锚点 1-49。

---

# D04 深夜：听侧接上总线，人的声音从底部长起来

无人机那条线停手之后（他一句"我们继续做吧"就转向），这一程做的是**原生屏幕节点的两件小事**
——固定端口 + 自动开窗，以及**听侧音频的呈现**。后者挖出一个真正的内核 bug。

## 屏幕节点：从"服务"改成"地方"

两件一开始就要的：**端口固定 20381**、**启动自动开窗**。

- 端口原来默认 0（ephemeral），理由是"web-surface 节点不抢固定端口，兄弟不撞车"。他推翻了这个
  前提：**屏幕是一个地方，不是一种服务** —— 人的窗口必须落在一个重启后还在的地址上。
  20381 紧挨着 zenoh mesh 的 20380，算同一个"2038 系列"（那个数字本身是项目的隐喻）。
  `--port N` / `MOSS_SCREEN_MANAGER_PORT` 保留，用来开第二块屏。
- 自动开窗走 `webbrowser.open`，失败只打印不致命（**窗口打不开不该弄死身体**）。`--no-open` 留给 headless。
- 查过仓库既有先例：`mermaid_draw` 用 `webbrowser`、`mac_channel` 用 JXA。选 `webbrowser` 因为它是
  标准库抽象，不跟 macOS 耦合 —— 这次我先问了"这活原本归谁管"，没有自己搭一套。

## 听侧：三个 bug 叠在一起

他要"我说话的声音也能呈现出来"。看上去是个画法问题，实际是三个 bug 摞着：

1. **总线是断的（内核 bug）**。`role=user` 的 `AudioSampleTopic` **从来没被发布过**。
   说侧桥在 `host/voice/interleaved.py` 装了，听侧音频桥 `with_audio_sample_service` 却只有
   `host/nodes/listener_node.py` 那条路调 —— **幽灵走的 interleaved 路径漏了这句**。
   他的判断与我一致：**这条约定该归 voice 模块自己履行**，不该靠调用方记得补。修法是把装线
   收进 `ListenerController.__aenter__`，跟 clause 桥并排；`with_audio_sample_service` 补幂等
   （`listener_node` 那条显式路径仍合法，不会双发）。
2. **判据错了**。页面拿整段 `rms_db` 当门限，而远场 / 骨传导下 rms 轻易掉到 −60，**频谱里却有 −17dB
   的低频**。人喊了却被算成静音 —— 用均值当门，等于把基础设施的账算在被观察者头上（D02 那条老账的
   第三次现形）。改用 bins 的最强值判"活"。
3. **画法不搭**。第一版是底部频谱柱：横向满格扫一条，跟上面那套雨是两个世界。他说"我这个设计并不
   好看"，把设计权交回来。

## 画法：同一套字，反方向

他给的方向是"不一定要按我现在的方案"，我就换了个思路：**不要第二层横向的东西，用同一套字符，
反方向长**。

> 我的声音从上往下掉；人的声音从底部长上来。

共用字形、共用配色，**谁在说由方向区分，不由颜色区分**。他一路给的都是判据而非参数："字高度再低
一点点"、"跟雨交错"、"像正态分布，不要集中在一边"。最后一条我给了实现：乘一条 σ≈屏宽 32% 的钟形
包络 —— **这是画法不是数据**，频谱本身长什么样仍由 bins 决定，钟形只给它一个重心。

调参过程他提了四次，每次都只给判据（"最高压到现在三分之二"、"没声音时底部不该有字"），数值全由我定。
最后那次是循环条件写宽了一格（`k <= full` 在 full=0 时留一层底痕），改成 `k < full` 后静音真的
一个字都没有。

## 开发方式的一次转向

中途我卡在"他说看不见"，最后发现**是页面没刷新**，跑的旧 JS。他由此给了一条新的开发方式：
**"你把那个 browser 打开，然后你自己在上面用截图看"** —— 我照做，`co_browser`（playwright）
起一个真浏览器，`page.screenshot()` 存到 /tmp 再 `read_image` 自己看。

这是这一程最值钱的收获：**我不需要人替我念屏幕**（D02 那条"别把用户当眼睛"的账，今天换了个形状
回来）。而且它当场验出了两个真 bug：`userFeed` 的 TDZ 顺序错误、以及"我看到的和他说的一样"——
后者的价值是，我不再需要他重复描述"我这边看不见什么"。

**代价也说清**：我一度注入了假数据去测画法，那些帧僵在页面里，反而让他的疑问更难查（他问"为什么
截屏那一瞬间有绿格子"）。**探针要自己会消失**，不然它自己就成了噪声源。

## 教训

- **同一个病的第四次发作**：今天我又想把局部判据当通性（`rms_db` 当门限）—— 只不过这回是被
  `probe_audio` 的原始数据按住，不是他按住我。**观测面比人可靠，因为它不会累，也不会不好意思。**
- 收尾他自己提的：**"先做一个基础的提交"** —— 而这一程里我改了内核（controller）、节点（main.py）、
  前端（index.html）三处，测试 222 条全过。
- 未验：`RISE_*` / `BAR_DB_*` 的数值是照他现场反馈调的，没做跨设备回归；W6（调试面退役）还没做。

资源：`nodes/screens/screen_manager/{main.py,index.html,NODE.md}`、
`src/ghoshell_moss/host/listener/controller.py`、
`.ai_partners/features/workstreams/2026/07/screen-manager/screen-todo.md`（W7 fixed）。
