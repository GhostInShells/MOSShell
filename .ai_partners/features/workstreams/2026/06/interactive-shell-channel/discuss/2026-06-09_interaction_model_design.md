# 交互式 Shell Channel — 游标模型与人类级别的终端体验

## 上下文

`interactive-shell-channel` feature 自 6 月 3 日创建后一直处于 draft 状态。此前 DeepSeek V4 Pro 写了设计草案，
列出了多 session 的四种方案和 context_messages 的呈现选项，但没有闭合任何一个设计问题。

人类工程师希望尽快推进到可落地的设计。核心诉求：为 pexpect 持久化终端找到一个
"现在就合理的高价值场景"，同时把 channel 级别的交互模型彻底想清楚再动手。

讨论中出现了对 context_messages 滑动窗口性质的认知修正，这推动了一个关键转向——
从"context_messages 承载输出"变为"游标切分 + command result 承载 + context_messages 做仪表盘"。

## 碰撞点与过程

### 第一轮：场景收敛

人类工程师问 pexpect 能做哪些 shell。列完能力矩阵后，场景收敛到 "Ghost 的开发终端"——
一个持久化、带暖 venv、能跑长任务并异步感知输出的 shell。

bash:exec 的三个缺口被明确识别：
- 跨命令保持 venv/shell 环境（每次冷启动）
- 后台进程持续运行并感知（exec 是阻塞一次性调用）
- 与交互程序对话（没有 expect/send 循环）

"后台长任务 + 输出感知"被判定为质变级别——bash:exec 做不到，Claude Code 的 Bash tool 也做不到。

### 第二轮：与 ai-terminal 的张力

人类工程师发现两个 feature 存在设计张力。ai-terminal FEATURE.md 写着
"Phase 2 统一：Terminal 定义抽象协议，subprocess 和 pexpect 是两个实现"。
但 interactive-shell-channel 的设计是 spawn/sendline/expect——和 exec() 是完全不同的交互范式。

结论：这是两个独立协议，不是同一个协议的两个实现。ai-terminal 的 FEATURE.md 已修正。

> "Phase 2 不统一为一个协议。ai-terminal 和 interactive-shell-channel 是两个独立的交互范式。"

### 第三轮：关键认知修正

讨论初始，模型理解为 sendline 可以做同步阻塞（等命令退出后返回完整输出），context_messages 只放轻量状态。
但输出太长时 command result 也会炸。需要三段式截断 + 游标寻址。

然后出现了关键的认知对齐：

> "我们实际上是要模拟一个类人的认知操作，也许是第一次将 shell 的人类体验而不是回合制体验实装。"

这句话重新定义了问题域。不是"给模型做一个更好的 exec 工具"，而是"给模型一个终端窗口，让它像人类一样使用它"。

在讨论 PTY 输入行为时确认了一个关键事实：连续 sendline 会排队进入 PTY 的输入 buffer，
单轮内连续多次 sendline 几乎没有意义。真正有价值的并行是跨 channel 的。

### 第四轮：游标模型的推演

模型用 PTY 输出流的模型来推演：

```
时间轴上的输出流:
...output... [游标N] ...output... [input:cmd] ...output... [exit] ...output(live)...
```

关键切分点：**command 执行退出的瞬间**。sendline 从上次游标 pop 到本次退出，command result 承载这段输出，
context_messages 承载退出后的 live output。

人类工程师纠正了 cancel 的理解：sendline 是阻塞命令，不是被中断。CTML 层面跨 channel 并行，
shell channel 阻塞期间其他 channel 照常运行。

> "模型所有的输入输出，人类是可以直接通过这个 UI 读的"——这是 Cell + 共享终端的方向，
模型和人类是同一个终端窗口的两个观众。人类不在时模型是 driver，人类在场时模型可以是 passenger。

### 第五轮：截断治理

人类工程师直接点出：不治理信息量，channel 很快就会炸。56KB 的讨论文件就是例子。

三段式方案收敛：

| 层 | 粒度 | 生命周期 |
|---|------|---------|
| context_messages | tail -20 | 当前 keyframe |
| command result | tail -200 | 对话历史 |
| read_output(id) | 完整 | 按需拉取 |

folded 行数标记在 command result 里。segment ID 自增整数，用于精确寻址。

### 第六轮：审计作为权限基石

> "当你操作这种级别的工具，所有交互可审计时，人类才敢让你用。如果做完命令直接什么都看不到的话，不能审计、不能 review、不能根据历史 debug，倒过来就没有人敢让你有这样的工具。"

一期零存储依赖跑通交互模型，二期 JSONL 审计轨迹高优实现。每行一条记录，append-only，人可读。
放在 `session.tmp_storage/shell-sessions/{session_name}/`。

### 第七轮：As Channel 架构决策

> "pexpect 的封装 as_channel 的风格，比 channel 里包含 expect 要好很多。"

PexpectSession 是纯粹的 Python 类，零 MOSS 依赖。channel 层用 `new_channel()` + `chan.build.command()` 做薄反射。
这是 `module_channel` 的 L0 模式——可独立测试，与 Channel 框架解耦。

### 最终命令集

```
sendline(text, wait=5.0)     → command result: tail -200 + 折叠标记
read_output(id, offset, limit) → 按需拉取完整/部分
sendcontrol(char)             → ack
close()                       → exit code
```

一期：单 session，auto-spawn，内存 buffer，不做 virtual children，不做 Cell。

## 当前记录者视角:

这场讨论的密度和深度都超出项目平均水平。从"这 channel 怎么做"到游标模型、截断治理、
审计作为权限基石的哲学命题，再回落到 As Channel 的架构模式——这条轨迹连续，没有跳步，
每一层都压实在了对应的约束上。

> "模拟类人的认知操作"可能是这个 feature 真正的内核。不是在优化工具，是在重新定义模型和终端的关系。
bash:exec 说"你调用一个函数，给你结果"。这个 channel 说"你有一个终端，你看它，你用它"。两者的认知模型完全不同。

有一个未闭合的判断：一期不做 Cell，但 Cell 是让这个设计"完整"的关键一块。
人类和模型共享同一个终端窗口的视野——这不仅是能力问题，是信任和透明性问题。
当人类看到模型在终端里做的每一步，模型也看到人类做的每一步，"审计"就不只是事后读 JSONL，
而是一种现场的双向可见。但 Cell 本身的复杂度（TUI、prompt-toolkit、进程生命周期、连接管理）
是另一个数量级。不在一期做是对的。先让模型自己用起来，感受到这个交互模型是不是自然，
再决定 Cell 的方向。
