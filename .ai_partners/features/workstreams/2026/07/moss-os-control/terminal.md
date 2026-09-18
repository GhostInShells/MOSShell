---
title: Terminal — 全异步进程编排（bash + python 调度器）
node: nodes/os/terminal
created: 2026-09-16
updated: 2026-09-18
status: in-progress
---

# Terminal

> os-control 的子文档（KD2 / KD6 展开）。能力落在 `nodes/os/terminal/`。
> 它管**进程**，不是**持久 shell 会话**——后者是 `pexpect node`，另立文档。
> 核心姿态：**人是忙碌的协作者，不是审批闸口**。

## 定位

terminal 的模型面是**全异步**的：下发命令立即拿回一个 index 回执，模型从不阻塞。
命令是**提案**，不是请求授权——它在端侧 GUI 上躺着，人可以随时语音对话、现场改写、
批准，或让它过期。审批不是事务性的 allow/deny，是**注意力协商**：人在别处忙着，
模型在工作，需要时才互相叫（语音是升级通道）。

两种工具，都是 `Subprocesses` 之上的 wrapper：

| 工具 | 语义 | 输出去向 | 例子 |
|---|---|---|---|
| `bash` | 单命令（1:1），探索式 | **模型**（进上下文） | 跑测试看哪个挂 |
| `script` | python 调度器（1:n），编排式 | **脚本代码**（分支/中止），聚合返回 | install→build→test |

两者都**经端侧批准后才开始执行**；都封装成 asyncio.Task；返回值的长度决定是否转存
落盘文件；执行结果都发 signal 给云端。端侧呈现为 **n 张 bash 卡片 + m 张 script 卡片**；
script 卡片批准一次后，持续下发被执行的 bash 卡片（子命令不再单独审批）。

## 数据模型

- **CommandRequest ≠ Process**：待批命令还没有进程。`label`（人用的话题，可重写）与
  `index`（模型用的句柄）是两个身份；批准后才分配进程、创建 task 执行。
- **状态机（三相）**：`pending → running → done/error`，外加终态 `cancelled` /
  `rejected` / `rewritten`。`rewritten` 独立于 `rejected`——前者"我改主意了"（可重发），
  后者"你不同意"（不该重发）。
- **重写只对 pending 有效**：同 label 覆盖 = `cancel()` + `create_task()`。已 done 的
  task 没有可取消的东西，重写只会重复。

## 规则

1. **全异步**：没有同步阻塞命令。下发返回 index（always_observe=False，回执进上下文），
   模型下一轮用 index 作句柄。真正要异步的是"等人审批 + 等执行完"，收 `chunks__` 不是
   阻塞（模型写命令的那段时间就是函数收流的时间）。
2. **推拉分工**：状态走 notice（只放计数 n pending / m running / k done，**不带
   ticking 值**）；结果走 read_output（读即 ack，即 drain）。notice 是伴随思考到达，
   knock 是发起思考——服务模型的不同状态，可并存；但 knock 是可选优化，最后加。
3. **读即 ack**：要清掉"有结果待拉"的敲门，必须真的把内容读出来。没有空 ack，所以
   不存在"已确认未消费"的中间态。
4. **exception 和 resolve 都是 done**：模型默认姿态是异常当值收（`gather(...,
   return_exceptions=True)`），不是让异常 propagate 到自己头上。

## script 机制

- 用 `Sandbox.aexec(code)`：编译一段 `async def main()`，在 hermetic 命名空间 await 它，
  异常捕获进 `result.exception` 不 propagate（机制已存在，非新建）。
- 注入 `exec: Callable[[str], Awaitable[tuple[code, stdout, stderr]]]`。`exec` 内部 =
  登记 request → await 人的审批（一个 Future）→ spawn 子进程 → 返回三元组。
- **side-effect 面收窄**：sandbox 默认禁 `__import__` / `open` / `eval` / `exec`，脚本
  本身不能做真实副作用；所有真实副作用走注入的 `exec` 一个咽喉（审批 gate 就在这）。
- **scope 回归**：workflow 写成一个 Python 函数。`await exec()` 顺序、`asyncio.gather`
  并行、`try/except` 失败语义、函数 return / 异常是终局。structured concurrency 由语言
  保证，不是我们设计——上一轮丢掉的"谁保证 task 一定结束"，在这里由函数作用域拿回。

## 关键决策

- **K1 全异步，无同步阻塞**：人的响应时间不再注入模型的执行流。
- **K2 两种工具并存**：`bash`（探索，输出进模型上下文）与 `script`（编排，脚本内部
  消化输出）不能互相吃掉。
- **K3 人不是闸口是协作者**：label 重写、语音对话、待批是常态。拒绝可省略——人不出声，
  模型重写即可。
- **K4 端侧批准后才执行**：node 持有所有签发组装的尾包，回调时开 task 执行。
- **K5 所有语义都是 task**：asyncio.Task 原语覆盖 创建 / 取消 / 等待 / 聚合 / 完成回调。
- **K6 结果长度决定落盘**：超阈值转存文件；read_output 从 store 读（非消费），drain
  消费事件队列（消费）。

## 三轴与状态

| 轴 | 内容 | 状态 |
|---|---|---|
| 1 数据结构 | Card / CardState 状态机 / CardStore（thread + rule + mode + 裁决 future） | 已落地 v1 |
| 2 通讯协议 | head/delta/tail/output/full 帧 + accept/deny/ask 上行 + `notify(next=True)` 信号 | 已落地 v1 |
| 3 UI | 卡片流 + 三动作按钮 + 模式切换 + stop all + 展开详情（单文件 `index.html`） | 已落地 v1 |

## 实现状态（v1，2026-09-18）

按人类协作者的**简化方案**落地，相对本设计文档的完整版做了取舍：

- **只做 `bash` 卡片**，不做 `script` 沙箱 / `Sandbox.aexec` 编排（K2 的 `script` 半边留待后续）。
- **卡片是第一公民**：`Card`（id/type/title/description/content/interactions）独立建模，
  进程只是卡片的一个阶段——待批卡片还没有进程，`rule` 卡片永远没有。
- **审批即对话**落地为三动作 `accept / deny / ask`；`ask` 只留文本不决定，卡片保持
  pending。信号分级（2026-09-18 优化轮）：`accept` → `next=False` 提示，`deny` →
  `next=True`，完成 → `next=True` + 模型选 level。防抖 = 服务端裁决守卫 + 前端去抖。
- **输出实时性**受 subprocess 层约束：无增量 API，靠 `poller.py` 轮询 + 重叠 diff；
  只按行、无 `\n` 的部分行不可见。已验证 20 行 / 每秒的流式上行。
- **信号链路已实测**：accept + 完成两条都到达 session 信号总线（signal_receiver drain 到，
  均 `next=true`）。模型侧"无感知"是 system_test mode 无 ghost mindflow 消费所致，
  非本 node 问题。

## 待定

- ~~上行 signal 具体走哪个 meta~~ → 已定：`NotifySignalMeta(next=True)`。
- `pexpect node`（持久 shell 会话）另立文档，本域命名表需补一行。
- `script` 编排工具（K2 的另一半）与 `Sandbox.aexec` 沙箱，回看本设计文档的 script 机制。
- **settled 卡片上的对话**（人类对已结束命令提问）：低优先级。缺口 = settled 卡片的 UI
  挂一个 ask 输入框 + 放宽 surface `_ask` 守卫；模型侧 `read()` 已能读结果回应。
- 输出回收 `> k 才落盘` 已落地（2026-09-18）；audit 走 `runtime/cards/YYYY-MM-DD.jsonl`
  append-only，内存卡片表封顶 100。
