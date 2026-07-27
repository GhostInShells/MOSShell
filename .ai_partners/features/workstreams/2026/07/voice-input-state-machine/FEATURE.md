---
title: Voice Input State Machine — 语音输入全状态机与交互模式
status: design-locked
priority: P0
created: 2026-07-28
updated: 2026-07-28
depends:
  - audio-capture
  - node-migration
  - channel-meta-dyn-static
milestone: 0.1.0
description: >-
  将语音输入从"两个独立 app 拼接"重构为单一感知节点，基于四层分层状态机
  统一交互模式、话语生命周期、发送闸口、打断粒度。端侧控人、Nucleus 侧控模型——
  以可编程协议对话取代隐式自由对话假设。
---

# Voice Input State Machine — 语音输入全状态机与交互模式

> 人类架构师 + claude-fable-5 (opus-4-7)。v0.1.0 语音输入能力的完整拓扑设计——
> 不是再做一个 app，而是收束 audio-capture 之后沉淀的所有分层概念，建一个
> 可迭代的架构槽位。

## Motivation

1. **交互模式不是进程**。当前 voice 输入由两个独立 app 提供——`sensors/listener`
   (341 行，回合制持续聆听 + TTS 三重门控) 和 `sensors/ptt_listener` (180 行，
   按键式 + 全局键盘钩子)。切模式 = 杀进程换进程。没有一个运行时可切换、可被
   模型和人共同观察的"模式"状态。

2. **话语生命周期是隐式的**。`idle → 收音 → 识别中 → 提交/丢弃` 这条链埋在
   listener 的 while 循环里。三重门控、cooldown、silence timeout 都是这个隐式
   状态机的补丁。没有声明式状态，就没法跨模式复用、没法可视化、没法测试。

3. **分层概念已经沉淀，但散落在各处**。audio-capture 定义了 AudioTransport 解耦、
   SpeechTopic 统一话语事件、AudioSignal 接入 mindflow、五种交互模式 (KD15)。
   但这些是设计文档里的定义——实现层没有统一成单一可维护的节点。

4. **"自由对话不可能，协议对话可编程"**。行业倾向用端到端实时语音模型拟合
   人类自由对话节奏，但人类对话节奏随场景、性格、关系快速变化，没有普适模型。
   正确的抽象是：**对话协议可编程**——把节奏协议显式化，人才能学会它，模型
   才能反身性地调整它。这要求状态机可声明、可观察、可配置。

5. **感知需要闸口**。ghost-runtime-safemode 是出方向的闸口（行动须审批），
   语音输入需要入方向的闸口（感知不一定立即进入思考）。manual-send 和
   model-validate-send (KD15 #4/#5) 本质上和 safemode 是同构的——都是"先存着，
   等条件满足再放行"。但 KD15 把它们和交互触发维度混在一起，需要解耦。

## Design Index

### 四层分层状态机 — 四种时间尺度

```
┌─ voice-input node (sensors/voice) — 单一感知节点 ───────────────┐
│                                                                   │
│  L1 设备层 (分钟级, 人主权)                                        │
│    开关、设备选择、多设备、降噪、空间/角度                           │
│    控制面: 端侧 channel (人); 模型可请求 (反身性 channel)           │
│    状态可见: AudioRuntimeTopic (已有) 扩展                          │
│                                                                   │
│  L2 流层 (百毫秒级, 端侧配置)                                       │
│    VAD、声纹、唤醒词、片段存储、流边界切分、commit 时机              │
│    说/听联动 (回声消除/半双工/近场远场)                              │
│    控制面: 端侧 channel + 流层配置                                  │
│                                                                   │
│  L3 加工层 (几十毫秒级, 并行管线)                                    │
│    ASR (主)、声音事件检测 (并行)、声纹匹配 (并行)                    │
│    每类加工产出独立 signal 类型                                     │
│    并行管线 = 同源流 → 多消费者，各自独立生命周期                    │
│    门控节点: 策略接口 accept(utterance) → bool                      │
│                                                                   │
│  L4 Nucleus 层 (秒级, 模型反身性)                                    │
│    Signal → Impulse 加工: 尾包 mode + 优先级                        │
│    首包双 flag: barge-in (打断嘴) / attention (抢占注意力)          │
│    打断粒度: speech 组 / 全部 / 不打断                              │
│    ChallengeMode: default / notify / silent                        │
│    Hint (不进历史, 临帧 prompt)                                     │
│    Buffer/drain (模型可感知未消费的 signal)                          │
│                                                                   │
│  旁路 — 不经 mindflow                                               │
│    快捷指令反射: 白名单 + 单调降能量, 安全约束                       │
│    间包聆听反馈: channel 级局部反射, 0.5s 预算闭环                   │
│    间包思考: 预留, 0.1 不做                                         │
└───────────────────────────────────────────────────────────────────┘
```

### 交互模式 = 配置组合，不是独立状态机

五种模式 (KD15) 不是五个状态机。它们是对同一个分层状态机的**配置组合**：

| 模式 | L2 commit 触发 | L4 barge-in | L4 attention | 闸口 | 0.1 |
|------|----------------|-------------|--------------|------|-----|
| push-to-talk | 按键松开+尾音 | speech | 是 (incomplete impulse) | auto | 做 |
| enter-to-talk | 回车 | speech | 是 | auto | 做 |
| turn-taking | VAD 静音 | speech | 是 | auto | 做 |
| free-duplex | VAD 静音 (常开) | speech | 是 | auto | 做 |
| manual-send | 任意上述 | 任意 | 任意 | manual (人点发送) | P2 |
| model-gate | 任意上述 | 任意 | 任意 | model-gate (flash 模型判断) | P2 |

manual-send 和 model-gate 的差异不在触发，在**闸口**——这是和触发维度正交的
独立维度。0.1 只做 auto 闸口。闸口接口设计为可替换策略，后续插入 manual/gate
不改变管线拓扑。

### 话语生命周期 — 所有模式共享的通用状态机

```
idle → armed → capturing → finalizing → staged → committed
                                                  └→ dropped (门控拒绝)
```

| 状态 | 含义 | 触发 |
|------|------|------|
| idle | 未激活，不收音 | 初始 / 模式关闭 |
| armed | 等待触发条件 | 模式激活（如 TTS 结束 = turn-taking 的 armed） |
| capturing | 收音中 | 触发条件满足（按键 / VAD onset / 常开） |
| finalizing | 收音结束，等待 ASR 尾包 | commit 触发（松手/回车/VAD offset/静音超时） |
| staged | ASR 出结果，等待闸口放行 | ASR is_final 且 text 非空 |
| committed | 已发送 Signal + SpeechTopic | 闸口放行（auto 立即 / manual 人点 / model-gate 模型判） |
| dropped | 被丢弃（门控拒绝/误触/超时） | 门控拒绝 / 时长 < 阈值 / abort |

L2 finalizing 状态的 commit 触发由**模式**决定。L3 ASR 产出的 is_final 是
独立事件——finalizing 不等同于 ASR 结束，ASR 可以在 capturing 期间就产出 partial。
这是"尾包 commit 时机推回流层"的关键体现：**ASR 告诉你说了什么，交互模式决定
什么时候算说完**。

### 说/听联动 — L2 流层的双向感知

所有非 trivial 的双工语音都涉及 TTS 播放和麦克风收音的协调。三种经典模式：

| 模式 | 行为 | 工程需求 |
|------|------|----------|
| 半双工 (walkie-talkie) | 说时不听，听时不说 | TTS 播放 → gate 收音 |
| 全双工 + AEC | 边说边听，回声消除 | AEC 算法 (不在 0.1 范围) |
| 近场听/远场说 (耳机) | 物理隔离 | 设备选择即可 |

0.1 只做半双工。全双工需要 AEC，是 v2 命题。但 AudioRuntimeTopic 的 TTS 播放
状态已经可被流层读取——**跨向依赖已有载体**，不需要新抽象。

### 打断粒度 — 不是布尔，是输出通道分组引用

`interrupt=True` (Impulse 协议已有) 在当前实现中是全局 `shell.stop_interpretation()`。
但对于语音场景，用户通常只想打断嘴 (TTS)，不想打断手（正在执行的动作）。

MOSS Shell 从设计上支持 channel 分组和独立 interpreter 管线（2 月重构时预设，
3 月 reflex-fix 验证），因此打断粒度 = Shell 输出通道分组的能力投射：

| 粒度 | 行为 | 0.1 |
|------|------|-----|
| speech 组 | 只 attenuate TTS channel (Preemptable)，手继续 | 做 |
| 全部 | shell.stop_interpretation()，停所有 logos | 做 |
| 不打断 | 首包不触发 interrupt，等尾包 | 做 (turn-taking 默认) |

Preemptable Protocol (KD14, audio-capture Step 13 遗留) 在此成为必须件——
它从"待实现的 P2 尾巴"升级为打断粒度机制的前提。`Preemptable.attenuate()` 被
mindflow 的 attention preempt hook 调用，不经过自定义回调路径。

### 门控 — 管线可插拔节点

```
capturing → [门控节点] → committed/dropped
                │
                ├─ 直通 gate (v0)    — 所有 utterance 放行
                ├─ VAD gate (v0)     — 静音过滤
                ├─ 时长 gate (v0)    — 误触过滤 (< 阈值丢弃)
                ├─ 模型 gate (v2+)   — flash 模型预判
                └─ 声纹 gate (v2+)   — 说话人验证
```

接口：`Gate = Callable[[Utterance], GateVerdict]`，返回 `commit | drop | hold`。
hold 语义：暂不放行但保持 staged，可后续 commit 或超时 drop。门控可以延后
commit，但必须伴随占位 signal 告知"正在处理中"——ghost 不得处于"不知道自己在
等什么"的状态。

### 快捷指令反射 — 白名单 + 单调安全方向

声音直接驱动机器人动作而不经过模型判断，是唯一有物理安全风险的路径。约束：

1. **白名单**：只有预定义的指令短语可触发反射
2. **单调降能量**：只允许"降低系统能量"的指令 (停止/归位/松开/静音/减速)，
   永不允许"提升能量" (移动/抓取/加速/启动力矩)
3. **声纹**：安全类指令不要求声纹门控 (紧急时任何人都该能喊停)；
   非安全指令不得走反射通道，必须经过模型
4. **冲突**：如果捷径和当前 ASR 识别结果冲突，捷径胜出 (停就是停)

实现：匹配命中 → `ImpulsePrimitive.fatal_command()` + `interrupt=True`，
不经过闸口。旁路附加 `AudioAction.SHORTCUT_TRIGGERED` signal (BACKGROUND +
notify) 留痕。

### 状态可见性 — 不止是调试，是信任基础设施

每一层的状态变更必须对人和模型同时可见：

| 可见对象 | 机制 | 内容 |
|----------|------|------|
| 人 | VoiceNodeRuntimeTopic + TUI 指示条/波形窗 | 模式、收音状态、partial 文本、staged 文本 |
| 模型 | perspective (Attention.with_perspective_func) | 听觉设备状态、当前模式、buffer 深度 |

**"听不见"必须是一个可感知的事实，不是信息的缺席**。当人关了麦、设备掉线、
门控拒绝时，ghost 必须知道自己听不见——否则会产生错误的世界模型（以为对方沉默）。

可视化分两档：
- **最小形态** (0.1 做)：终端状态条——模式 + 呼吸灯 (按话语状态变色) +
  partial/staged 文本。复用 desktop-gui 的 breathing-light indicator 模式。
- **正式形态** (0.1 做)：qt_screen 的 window——波形 + 状态 + staged 文本编辑区。
  复用 screen node 的 float slot 和 URL window 机制。

### 节点边界

`nodes/sensors/voice/` — 单一感知节点，模型通过 CTML (`<nodes:run_node>` /
`<nodes:stop_node>`) 或 matrix channel 命令开启/关闭/切换模式。

内部聚合：
- 设备 + 流管道 (本进程直采，不走 Zenoh 转一道——模式切换同时改捕获时机和 ASR
  送入时机，跨进程协调是 listener 门控复杂度的根因)
- ASR 管线 (VolcengineASR，接口可替换)
- 并行处理管线 (声纹/声音事件，预留，0.1 不实现)
- 状态机运行时
- 控制 channel (暴露给模型反身性使用)

audio_capture 节点保留为独立 PCM 源供 waveform 等其他消费者使用。
voice 节点不依赖 audio_capture——它在进程内直接持有 capture pipeline。

## Key Decisions

### KD1: 四层分层，不是单一状态机

**接受**: 语音输入不是一层能描述的系统。设备层 (分钟)、流层 (百毫秒)、加工层
(几十毫秒)、Nucleus 层 (秒) 是四种时间尺度和控制权限的边界。每层独立状态机，
层间通过定义好的合约通讯。

**拒绝**: 单一状态机——listener 的 341 行就是单一状态机 + 补丁的失败证明。
三重门控是"TTS 在播吗"(设备/流层事实) 被编码进了"要不要送 ASR"(加工层决策)，
每加一个场景就加一层防线。分层明确后，门控从防线退化为管线节点。

**拒绝**: 两层设计 (端侧 + nucleus)——缺少流层，commit 时机 (PTT/enter/VAD)
无处安放，会和 ASR 尾包逻辑耦合回单一状态机。

### KD2: 模式 = L2 触发配置 + L4 参数组合

五种交互模式 (KD15) 不是五个状态机，是上述分层状态机的配置快照。
L2 决定何时收音、何时 commit；L4 决定首包策略、打断粒度、ChallengeMode。
manual-send/model-gate 是闸口维度，和触发维度正交，不应混入模式枚举。

**接受**: 模式作为命名配置组合暴露，底层状态机不变。切模式 = 改 L2/L4 参数，
不重启管线。

**拒绝**: 模式 = 独立 app/进程——当前 listener/ptt_listener 并存正是此问题的症状。
`moss nodes run sensors/ptt_listener` 和 `moss nodes run sensors/listener` 是两个
进程，切模式 = kill + start，做不到运行时无缝切换。

### KD3: commit 推回流层, signal 退化到标准协议

PTT 松开/enter 回车/VAD offset 决定的是 **音频流的尾包 commit 时机**——即何时
告诉 ASR "说完了"。这与 ASR 自身的分句判停 (is_final) 是两个独立事件。

交互触发决定 L2→L3 commit；ASR 分句决定 L3 internal logic。
L3→L4 signal 只带 ASR 结果 + complete flag + priority——不关心触发方式。

**收益**: 同一套 Signal/Nucleus 逻辑对 PTT 和 VAD 都成立。PTT 按着不放
说长句——ASR 中间有 is_final 分句想提交——此时需要明确行为：PTT 模式下，
ASR is_final 不分句提交，松手才是唯一 commit 信号。

### KD4: 首包打断 = barge-in (嘴) 和 attention (注意力) 两个独立 flag

`barge_in` 控制是否触发 `Preemptable.attenuate()` (只停 TTS channel 组)。
`attention` 控制是否发 incomplete impulse 抢占注意力 (现有 AudioNucleus 机制)。
两者独立，组合出不同对话风格：

| barge_in | attention | 效果 |
|----------|-----------|------|
| True | True | 首包既停嘴又占注意力——强打断 (free-duplex) |
| True | False | 首包只停嘴，不占注意力——弱打断 (driver-passenger 车载) |
| False | True | 首包不打断，但抢占注意力——等播放完再响应 |
| False | False | 首包只更新 buffer，不打断不等——纯 drain (G1 监控) |

`barge_in` 的粒度不是布尔，是分组引用 (speech / all / none)。

**拒绝**: 单 flag `interrupt`——当前实现只有全局 stop_interpretation。
这会打断正在执行的非语音动作 (机器人运动、文件编辑)。

### KD5: 说/听联动 — 半双工先行，全双工槽位预留

L2 流层感知 TTS 播放状态 (AudioRuntimeTopic)，在半双工模式下 gate 收音。
全双工 (AEC) 需要回声消除算法，不在 0.1 范围。但 AudioRuntimeTopic 的
`device_name="speaker"` + `running` 字段已回答"输出在不在播"——L2 读取它
即可，不需要新抽象。

**接收**: 半双工——TTS 播放时 gate 收音 (类似手机通话的硬件限制)。
**拒绝**: 软件 AEC——这是独立的信号处理命题，不属于此 feature。
**预留**: 全双工模式可通过流层的设备选择 (耳机 = 近场麦克风 + 远场扬声器 =
天然 AEC) 或添加 AEC 处理节点实现，不破坏分层拓扑。

### KD6: 门控接口 — 策略模式，v0 无智能

门控节点接受 `accept(utterance) → GateVerdict`，可插拔。0.1 实现三种零智能门控：
- `PassthroughGate`: 全放行
- `VADGate`: 基于 VAD 信号过滤纯噪音帧
- `DurationGate`: 过滤 < 阈值的误触

模型门控、声纹门控是 v2+ 扩展——门控模型引入了一致性问题 (门控和 ASR 对
"说完了吗"的判断可能分歧)，但接口设计已预留：`hold` 允许门控延迟决策
(伴随占位 signal)，门控可以提前 commit 不能无限期延后。

### KD7: 快捷指令 — 白名单 + 单调安全

声音直接驱动动作 = 物理安全风险。两条硬约束：
1. 只允许"降低系统能量"的指令
2. 紧急停止类不要求声纹 (任何人可喊停)

这既是安全设计也是信任设计——人类知道任何情况下喊停都有效，无需担心模型
是否在线、推理是否延迟、闸口是否拦截。

### KD8: voice 节点 = 单一进程，capture 保留为独立源

`nodes/sensors/voice/` 在本进程内直持 capture pipeline，不经过 Zenoh PCM 流转发。
模式切换需要同时改捕获时机和 ASR 送入时机——跨进程协调这两个维度的尝试
(listener + audio_capture 分离) 是门控复杂度的根因。

`sensors/audio_capture` 保留为独立节点——waveform 可视化、录音等纯 PCM 消费者
仍可通过它获取原始流。voice 节点不替代它。

**拒绝**: capture 和 listener 保持分离——这是当前架构，已在实践中验证跨进程
协调的脆弱性 (三重门控、queue buffer 隔离、aclose 陷阱全部源于此)。

### KD9: 状态可见 — "听不见"是可感知的事实

设备层/流层的运行时状态通过 `VoiceNodeRuntimeTopic` 暴露，携带四层状态快照。
模型通过 `Attention.with_perspective_func` 在每帧获取听觉能力状态的 perspective
视图——ghost 知道自己的耳朵开着还是关着、收音中还是静默中。

可视化通过最小形态 (TUI 状态条) 和正式形态 (qt_screen window) 双轨实现。
状态可见是信任基础设施——麦克风是最敏感的持续感知，人必须时刻知道它的状态。

### KD10: 间包聆听反馈不经 mindflow — channel 局部反射

间包反馈的 0.5s 延迟预算不允许经过 impulse → challenge → attention → articulator
的全链。实现为 channel 层面的局部反射：多分类命中 → 直接调预置音频资源播放。
事后发 BACKGROUND + notify signal 留痕。

反射协议的发现机制：channel 声明可用的反射能力 (语义、延迟预算、触发条件)，
模型在运行时绑定事件→反射的映射关系。这是反身性可编程反射，不出现在 0.1 范围，
但核心理念在此声明：**有些反应不经大脑**。

**依赖**: channel 的能力声明机制。audio-capture 的间包反馈规划曾被视为依赖
channel-meta-dyn-static——澄清后，channel 作为 node 的组成部分，在 matrix 中
自动声明资源与能力，模型可感知协议并配置反射逻辑。不阻塞 0.1。

## Contract

> 本章的代码片段是设计草图——表达意图和字段语义，不是实现规范。字段名、方法签名、
> 模块路径都是示意性的。实现时以 Key Decisions 为约束，以代码可读性和现有项目
> 惯例为具体决策依据，不要照抄这里的伪代码。

### voice node — 模型可见的控制 channel

```
voice                         # 主 channel
  .start()                    # 启动感知
  .stop()                     # 停止感知 (关闭耳朵)
  .mode:                      # 模式子 channel
    .set(name)                # 切换模式 (ptt / enter / turn_taking / duplex)
    .current()                # 查询当前模式与参数
  .gate:                      # 门控子 channel
    .set(name)                # 设置闸口策略 (auto / manual / model_gate)
    .pending()                # 查看 staged 但未发送的话语
    .commit(id)               # 手动发送 (manual 模式)
    .drop(id)                 # 手动丢弃
  .config:                    # 配置子 channel
    .show()                   # 当前全部配置
    .set_barge_in(target)     # 打断粒度: speech / all / none
```

### 新增/修改合约

```python
# contracts/voice.py — 新增

class VoiceNodeRuntimeTopic(TopicModel):
    """voice node 四层状态快照。每层变更时 pub，max_size=1。"""
    running: bool = False

    # L1 设备层
    device_name: str = ""
    device_sample_rate: int = 0
    noise_suppression: bool = False

    # L2 流层
    mode: str = "off"          # ptt / enter / turn_taking / duplex
    stream_state: str = "idle" # idle / armed / capturing / finalizing
    barge_in_target: str = "speech"
    duplex_mode: str = "half"  # half / full (AEC)

    # L3 加工层
    asr_partial: str = ""      # 当前 ASR 部分结果
    staged_text: str = ""      # staged 但未发送的文本 (闸口 hold)
    gate_name: str = "passthrough"

    # L4 Nucleus 层
    attention_occupied: bool = False  # 当前是否有 speech attention
    buffer_depth: int = 0             # nucleus buffer 中未消费的 signal 数

    @classmethod
    def topic_type(cls) -> str: return "voice/runtime"
    @classmethod
    def default_topic_name(cls) -> str: return "voice/runtime"

class VoiceMode(str, Enum):
    OFF = "off"
    PTT = "ptt"
    ENTER = "enter"
    TURN_TAKING = "turn_taking"
    DUPLEX = "duplex"

class StreamState(str, Enum):
    IDLE = "idle"
    ARMED = "armed"
    CAPTURING = "capturing"
    FINALIZING = "finalizing"
    STAGED = "staged"

class BargeInTarget(str, Enum):
    SPEECH = "speech"
    ALL = "all"
    NONE = "none"

class GateVerdict(str, Enum):
    COMMIT = "commit"
    DROP = "drop"
    HOLD = "hold"

class Gate(ABC):
    """门控策略 — 可插拔的管线节点。"""
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def accept(self, utterance: Utterance) -> GateVerdict: ...

class Utterance(BaseModel):
    """一次话语的上下文——门控输入。"""
    text: str = ""
    duration_ms: int = 0
    confidence: float = 0.0
    audio_key: str | None = None
    timestamp: float = 0.0

class ShortcutCommand(BaseModel):
    """快捷指令定义 — 白名单项。"""
    phrase: str                      # 匹配文本
    command_logos: str               # 执行的 CTML
    requires_voiceprint: bool        # 是否要求声纹
    energy_direction: str            # "lower" | "neutral" — 永不可为 "raise"
```

```python
# contracts/audio.py — 修改 AudioSignal

class AudioAction(str, Enum):
    SPEECH_STARTED = "speech_started"
    SPEECH_DELTA = "speech_delta"
    SPEECH_FINAL = "speech_final"
    WAKE_WORD = "wake_word"
    AUDIO_ALERT = "audio_alert"
    SHORTCUT_TRIGGERED = "shortcut_triggered"  # 新增: 快捷指令反射触发
    INTER_PACKET_FEEDBACK = "inter_packet_feedback"  # 新增: 间包聆听反馈触发

class AudioSignal(SignalMeta):
    action: AudioAction
    speech_topic: SpeechTopic | None = None
    barge_in: bool = False          # 新增: 是否触发打断 (停嘴)
    barge_in_target: str = "speech" # 新增: 打断粒度
```

## Implementation Progress

| Step | 内容 | 状态 |
|------|------|------|
| 1 | 四层状态机合约 — contracts/voice.py + AudioSignal 扩展 | pending |
| 2 | L2 流层状态机 — idle→armed→capturing→finalizing→staged 核心循环 | pending |
| 3 | 模式实现 — PTT / enter / turn-taking / duplex 四种触发策略 | pending |
| 4 | commit 推回流层 — ASR is_final 与模式 commit 信号解耦 | pending |
| 5 | L4 ASR Nucleus 重构 — barge_in / attention 双 flag + 打断粒度 | pending |
| 6 | Preemptable 集成 — TTS channel 实现 Preemptable.attenuate/resume | pending |
| 7 | 门控接口 + 三种零智能实现 (passthrough / VAD / duration) | pending |
| 8 | 快捷指令 — 白名单匹配 + fatal_command 反射 + 安全约束 | pending |
| 9 | 说/听联动 — 半双工 TTS gate (AudioRuntimeTopic 读取) | pending |
| 10 | VoiceNodeRuntimeTopic — 四层状态快照 Topic | pending |
| 11 | 控制 channel — voice/mode/gate/config 子 channel 树 | pending |
| 12 | 可视化 — TUI 状态条 (最小形态) + qt_screen window (正式形态) | pending |
| 13 | voice node 落地 — nodes/sensors/voice/ NODE.md + main.py | pending |
| 14 | listener/ptt_listener 旧 app 退役 — 功能被 voice node 覆盖后删除 | pending |

### 0.1 不做

- 模型门控 / 声纹门控 (Gate 接口预留)
- 间包思考 (Nucleus 侧 slot 预留)
- 间包聆听反馈反射实现 (channel 声明 slot 预留, 不落地)
- 提前思考 (articulator + 延迟 logos 发送, Ghost runtime 侧命题)
- 全双工 + AEC (L2 流层 duplex_mode="full" slot 预留)
- defer clear (Shell 侧命题, 不在此 feature)
- 声纹识别 (L2 流层过滤链 slot 预留)
- 声纹滤波门控 (L2 流层 slot 预留)

## Dependencies

- **audio-capture (completed)**: AudioTransport / SpeechTopic / AudioSignal /
  AudioNucleus / MiniAudioCaptureSource — 全部基建复用
- **node-migration (in-progress)**: nodes/ 目录结构, NODE.md 声明格式,
  provide_channel 作为 node → matrix 的能力注册入口
- **ghost-runtime-safemode (in-progress)**: turn-based approval UX——闸口概念
  的同构先例。manual-send 的 staged→commit 和 safemode 的 pending→approve
  是同一模式在入向和出向上的对称实现
- **channel-meta-dyn-static (design-locked)**: 间包反射的能力声明机制——
  不阻塞 0.1，但语音设计的"反射发现"依赖此线的 channel 自解释方向

## Handoff — 后续伙伴推进路径

### 优先级

1. **P0 — 四层合约** (Step 1): contracts/voice.py 落地。VoiceNodeRuntimeTopic、
   VoiceMode、StreamState、Gate 接口、Utterance、ShortcutCommand。AudioSignal
   扩展 barge_in/barge_in_target 双 flag 和 SHORTCUT_TRIGGERED 枚举。这是
   整个 feature 的合约锚点——必须先锁定，其余才可并行。

2. **P0 — L2 流层状态机** (Step 2-4): 话语生命周期核心循环 + 四种模式触发策略 +
   commit 推回流层的解耦。这是 listener/ptt_listener 逻辑的真正收敛点——原本埋在
   两个 app + 三重门控里的隐式状态，在此声明化。

3. **P0 — L4 ASR Nucleus 重构** (Step 5-6): barge_in/attention 双 flag +
   Preemptable 集成。AudioNucleus 从"首包 = interrupt=True"的单一行为扩展为
   可配置的双 flag 模式。

4. **P0 — voice node 落地** (Step 7-13): 门控 + 快捷指令 + 半双工联动 +
   Topic + channel + 可视化 + NODE.md。聚合所有组件为 nodes/sensors/voice/。

5. **P1 — 旧 app 退役** (Step 14): listener / ptt_listener 功能被 voice node
   覆盖后删除。需要 voice node 在三种触发模式下都通过功能等价性验证后再执行。

### 设计原则提醒

- **分层不可逾越**。每层只读同层或上一层的状态。L3 不去读 L4 的 attention 状态。
- **闸口是正交维度**。不要把触发方式 (PTT/VAD) 和发送方式 (auto/manual/gate)
  塞进同一个枚举。
- **状态永远可见**。每层的状态变更 = pub Topic。模型和人同等可见。
- **单调安全**。快捷指令只降能量不升能量。没有例外。
- **Preemptable 现在是 P0 件**。它是打断粒度机制的前提，已从 audio-capture
  的遗留 P2 升级为此 feature 的 P0。

---

*架构设计: claude-fable-5 (opus-4-7) 与人类架构师, 2026-07-28*
*基础调研: audio-capture FEATURE.md (DeepSeek V4 + Claude Opus 4.7) — 已完成的音频感知全链路*
*碰撞记录: 本会话对话 — 分层拓扑推演、交互模式收敛、安全边界讨论*
