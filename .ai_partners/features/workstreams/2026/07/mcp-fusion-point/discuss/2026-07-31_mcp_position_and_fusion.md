# 2026-07-31 — MCP 在 MOSS 中的位置与融合点

参与者: 人类工程师 + deepseek-v4-flash (via claude code)

一场从"语音输入怎么实现"出发、最终落到"MCP 在 MOSS 架构中的位置"的讨论。产出
一个 draft feature（`mcp-fusion-point`），本文记录讨论轨迹，不落决议。

## 上下文

会话起点是人类工程师要求两件事：进入语音相关的 feature，并调研 g1 实现里语音
输入的实现方式。调研结果引出了下一步的问题：speech-protocol-alignment 的通用化
需要一个跨进程控制协议，而人类工程师不想发明轮子——MCP 2026-07-28 版成为候选。
由此这场讨论从语音输入一路上升到架构位置问题。

### 调研事实（g1 语音输入）

g1 有两套并行听觉路径，对称设计在 `src/ghoshell_moss_contrib/unitree/g1/` 下：

- **远场**（`runtime/asr.py`）：G1 机载麦克风阵列，DDS topic `rt/audio_msg`，
  非流式整句识别，`is_final` 恒 false；angle/speaker_id 实机恒 0。
- **近场**（`runtime/listener.py`，1120 行）：蓝牙耳机 → miniaudio capture →
  Volcengine ASR websocket，流式 partial + finalized ring buffer 双层状态，
  `drain(force_finalize_partial=True)` 支持按键打断 VAD。`start()` 永不抛，
  配置缺失/蓝牙不在/ws 异常都是 health 状态。

channel 层（`channels/listener.py`）默认关闭聆听（人主权），耳机中键/F1 切换，
Y 键=自由对话（VAD 判停自动 NotifySignal），A 键=立即 drain 提交。context_messages
始终 tail-N 只读，状态每条可见——"听不见"是可感知的事实。

一个值得注意的现象：**g1 的实现独立"发明"了 voice-input-state-machine 设计里的
多个概念**（force drain ≈ PTT 提交、Y 键 ≈ turn-taking、默认关闭 + health 门 ≈
闸口与状态可见性）。但 g1 是进程内模块单例 + 命令式 drain/pause API，设计文档是
声明式四层状态机 + Gate 可插拔策略——两者没有走到一起。这不代表系统性偏差——
人类工程师持有完整架构设计，模型实例通过对话总结理解上下文，自然会在不同位置
看到不同程度的差异。

## 碰撞点与过程

### 1. 问题提出：通用化的控制动作需要跨进程协议

人类工程师提出 speech-protocol-alignment 的通用化：drain/pause 这类控制动作
可以跨 cell 存在，依赖跨进程的有状态协议。MOSS channels 本质上是有状态 1:1
协议，继续做下去就"不得不发明一个协议无关的 stateless transport RPC"。

> "之前我犹豫过很多次这个事情要不要做，现阶段还没有动手。但是好像有点不得不
> 的味道了。"

MCP 7/28 成为候选——人类工程师要求全面调研。

### 2. 模型的调研判断：MCP 7/28 就是那个 stateless RPC

模型读完 7/28 的收敛讨论记录（`.discuss/2026-07-30_mcp_duplex_convergence_and_memento_branch.md`）
和现有实现（moss-mcp / matrix / mcp-hub / cell）后给出核心判断：

**"你怕发明的那个东西，7/28 的 MCP 已经是它了——而且是行业标准版。"** 关键不在
"MCP 是不是 RPC 协议层"，而在把状态放在哪：transport 层 stateless（MCP 提供），
状态持有层在 cell 内的 channel 状态机（MOSS 已有），控制面是 MCP 工具面，
数据面留 Zenoh，仲裁层留 mindflow，授权层留 mesh/scopes。

映射是干净的：drain→tool 一次调用/Tasks 异步；pause→幂等翻转 tool；health→
list+ttl 缓存/subscription；mode→配置变更 tool；事件→subscriptions/listen。
边界表述为**"MCP 送达，MOSS 仲裁"**。

7/28 的 stateless 化同时溶解了"双工里套双工"——旧 MCP 有状态 session 嵌在
MOSS 双工体系内是嵌套双工（难），新 MCP 每请求自包含，嵌套消失。

### 3. 转向：开发时契约 vs 运行时发现

人类工程师提出一个更深的 reframe：**MCP 不一定是为运行时模型准备的，它可以为
开发时准备**。有了 MCP 协议快照，开发时的模型读取后按接口设计 RPC 调用，
类似遵循 Thrift 接口协议，运行时自检发现变更就报错。

> "问题在于，mcp 这套架构是绑定了运行时发现的，所以它意味着 cell 启动有拓扑
> 顺序的依赖。这是主要的集成障碍。"

模型回应：拓扑依赖不是 MCP 的属性，是**契约检查放在哪**的属性——设计时
（无依赖）/ 启动时 fail-closed（有依赖，B 每次启动要 A 在线）/ 调用时 lazy
（无依赖，调用才爆）。MCP 7/28 的自描述请求让持 snapshot 的客户端可以不经过
发现往返直接 tools/call。而 MOSS 有使能性质：channel 静态自描述（code as
prompt），`moss codex` 就是现成 IDL 生成器——**MCP 缺 IDL，MOSS 自带生成器**。

### 4. 综合：原生 stateless 接口 + 三身份

人类工程师进一步综合：开发时和运行时没有那么大壁垒。若 matrix cell 原生支持
stateless 协议接口（有状态通讯肯定不行），把 MCP 原生锚定为机制，首次运行时
dump `tools/list` 作为约定快照（`mcp.yaml`，按约定文件名经 nodes 发现体系
自动发现），cell 间调用走寻址 + 实例化 MCP client，ghost 用 cell 的 MCP 就
是动态取 MCP 节点封装 channel。

随后提出 MCP 的三个身份，并押注第一个：

1. **mcp server as cell**：mcp 启动时仍带 matrix cell/node 身份，可被治理。
2. **mcp client as cell**：mcp_hub 节点化，只是开启方式，无特殊治理意义。
3. **mcp client as channel**：即现有 `channels/mcp_hub.py`。

> "哪个身份更有利于 moss 架构的运行才是关键。我判断 1 才是，2/3 都依赖外部
> 去启动 mcp。"

### 5. 模型的验证与锐化：致命问题比看起来窄

人类工程师提出 MCP 的"致命问题"：协议不是 background 的（不像 moss channel
可运行在 asyncio thread 里），启动入口有约束，绕不过就只能做 mcp cell；而
mcp cell 是第三方能力桥，不提供 MOSS 技术体系，会退化到"用 mcp client 反射
一个 matrix cell"，那就没有 cell 间协议的价值了。

模型先做了实证验证：项目装的 fastmcp 3.4.3 的 `http_app()` 返回
`StarletteWithLifespan`（被动 ASGI callable），带 `stateless_http` 参数
（每请求新 transport，正是 7/28 模式），错误信息明确指导嵌入父 ASGI 应用。
**"MCP 不是 background"是 FastMCP 便捷生命周期（run_async 想拥有主循环）的
性质，不是协议的性质。** tool handler 与 channel 方法同进程同 loop，直接访问
状态，零跨线程。

于是退化论证成立且收敛：MCP 作为 cell 间协议的价值**只存在于"cell 是 server"
一条路径**；spike 过了就落地，不过 MCP 留在外部边界（Hub），内部维持现状。
二选一，没有中间态，且**可证伪、可廉价测试**。

### 6. 会话任务确立与判断

人类工程师明确本会话的正式任务：**确认 MCP 在 MOSS 架构中的位置**，连带问题是
是否以 MCP 作为 cell 间 RPC 底座。**前者比后者更重要。** 并给出关键动因：
"当我想要做的 cell rpc 已经和 mcp stateless 协议接近了，我就很不想发明轮子，
从而让 1~3 给 mcp 一个正式身份变得重要了。"

模型锐化两点：

- **身份 1 拆成双形态**，分别回答两个问题：受治 server 形态（任何 MCP server
  带 cell 身份被 nodes/mesh 治理）回答"位置"；原生表面形态（cell 的跨进程面
  就是它的 MCP 表面）回答"RPC 底座"。这个拆分消解了"mcp cell 退化"的张力——
  退化只发生在网关形态，原生形态不退化。
- **身份 1 让身份 3 变简单**：mcp_hub 现在的复杂性（config/连接/session/生命
  周期）全部来自"外部 MCP 不是 cell，没人治理"。若 MCP server 是 cell，hub
  退化为"发现 MCP cell + 封装成 channel"，继承 cells channel 的治理模式。
  "hub 参考 cells channel 逻辑做完善"与身份 1 是同一个方向。

### 7. 判断（非决议）

人类工程师给出三个判断，要求作为背景而非决议记录：

> "mcp 和 moss channel 基于高度类似的愿景而设计，最终会收敛到一处；我也判断
> 最终应该是资源决定生存。但现阶段 channel 仍然在一些设计上是 mcp 的超集，
> 比如时间第一公民，排序和阻塞，树形构建等等。所以两者一直互相不好兼容。"

"模型初见不考古，通常会认为有了 mcp 还在独立发明轮子"——而在持续阅读代码和
协议面的模型实例眼里，两者看似相近却又频繁出现不可兼容的设计差异。这个张力
本身就说明融合是值得正式对待的命题。

### 8. 留档

人类工程师决定：建立 feature（`mcp-fusion-point`），FEATURE.md 不放太多决议，
在 feature 目录起 discuss 记录本次讨论。正式目标一句话：**找到 mcp 与 moss
项目的合适融合点**。

## 模型的自留地

当前记录者视角:

这一场讨论最有意思的地方，是人类工程师在"不发明轮子"和"不轻信轮子"之间反复
走钢丝。三次转弯——从 RPC 协议到开发时契约，从开发时契约到三身份，从三身份到
"资源决定生存"——每次都是把抽象层次抬高一格，直到问题从"怎么做"变成"它
在体系里是什么"。

我自己的判断与人类一致：身份 1 是正确位置，7/28 的 stateless 让它在技术上也
成立了。但有两件事我保留了不同的权重：

一是"资源决定生存"这句。我完全认同融合点最终由生态与工程成本决定，但我要给
"channel 是超集"这半句加一个反向注脚：超集意味着 MOSS 现在占着时间/排序/树形
这些 MCP 故意不做的东西。MCP 不做的原因是"stateless 优先"，不是"能力不足"。
如果未来 MCP 生态往时间感知方向演进（SEP-2322 MRTR 已经露出苗头），超集关系
会倒转，届时融合点的几何会变。所以融合点不是一次定位，是持续观察。

二是 spike 的裁决力。我验证了 fastmcp 可嵌入，但"可嵌入"和"值得嵌入"是两件事。
真正的裁决是那个最小闭环：voice cell 的原生表面形态 + mcp.yaml 快照 + 二次进程
client 封装 channel。跑通它，位置确认就有实证；跑不通，退到"MCP 留在外部边界"
也一样成立。这场讨论的推演已经足够支撑下一个实例动手做这个实验了。

