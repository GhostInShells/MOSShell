---
title: Voice Input State Machine — 语音输入全状态机与交互模式
status: in-progress
status_note: 'ASR 通用契约提炼 (2026-08-10)：contracts/asr.py 扩 ASRInfo + get_info/configure, 与 TTSInfo 对称；VolcengineASRParams 分离模型可见旋钮, force_to_speech_time 从硬编码提进 config；audio CLI 拆分为 cli/audio/ 子包 (按协议探测层组织)；capture 迁 project 级, contracts 5 槽位 4 OK (asr provider 待注册).'
priority: P0
created: 2026-07-28
updated: 2026-08-10
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

`nodes/sensors/listener/` — 单一感知节点，模型通过 CTML (`<nodes:run_node>` /
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

`nodes/sensors/listener/` 在本进程内直持 capture pipeline，不经过 Zenoh PCM 流转发。
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

## 10 开关正交体系 (2026-08-04)

> 会话决策：voice node 的配置面收敛为 **10 个正交开关**。不是 0.1 全做——是多轮优化的维度空间，每轮切一个子集。

### 正交开关矩阵

| # | 维度 | 取值空间 | 默认 | 控制者 | 承载 |
|---|---|---|---|---|---|
| 1 | 聆听开关 | on/off | on | 人 + ghost | L1 + `.start/.stop` |
| 2 | 控制主权 | ghost/human/auto | auto | 移交机制 | 控制 channel 状态 + Topic |
| 3 | 流上下文缓冲 | 数据面(非开关) | 常开 | 系统 | buffer |
| 4 | 首包打断 | on/off (speech/all/none) | on | 模式配置 | `barge_in` → Preemptable |
| 5 | 首包抢占 | on/off | on | 模式配置 | signal 高优 + Nucleus 响应 |
| 6 | 音频存储 | on/off | off | 系统 | sink (matrix-resources 将来) |
| 7 | 尾包闸口 | auto/manual | auto | 人点击/ghost | 门控策略 + staged 编辑块 |
| 8 | 自动重写 | off/vad/stream | off | ghost 配置 | flash agent + 三份数据 |
| 9 | 信号优先级 | 值 | high | 系统默认 | Signal priority |
| 10 | 用户身份 | 文本 | 空 | 人填写 | Signal/Utterance meta |

### 耦合点（非完全正交，说破）

- **9 是 5 的前提参数**：attention 抢占靠高优首包/尾包供能。9 定值，5 决定抢不抢。
- **3 是基座，7/8 是它的两个消费者**：7 要"可编辑 staged 块"就是缓冲区的可编辑面；8 的三份数据 (buffer/未修正/修正后) 是缓冲区的三个视图。修正后立刻修正 buffer。
- **2 是元维度**：决定其它开关谁有写权。默认 auto (ghost 管)，人一点 UI 移交 human，ghost 经 channel 看到状态变更。

### 承载面分两轴（2026-08-04 决策）

| 轴 | 内容 | 载体 | 理由 |
|---|---|---|---|
| **启动模式** | headless / webview | argv args：`moss nodes run nodes/sensors/listener -- --mode X` | nodes_cli 已支持 append（`allow_extra_args` + `launcher.run.extend`），每轮可能不同 |
| **持久配置** | 10 开关 + device + 身份 | node home `config.toml` + env | 稳定跨启动共享，不该每次传参 |

**协议层不做 RPC**：按 ROS2 经验 parameter + topic 足够。voice node 表面 = config 文件 (持久) + `VoiceNodeRuntimeTopic` (运行时快照) + 控制 channel (ghost 反身性写) + UI ws (交互直连 node 本地服务，零矩阵跳数)。matrix 间通讯协议层与核心**解耦**——核心只暴露事件，协议层作为 adapter 消费广播（见「模块抽象约定」），由人类工程师并行推进。

### 设备选择三级 (隐藏点)

`AudioCaptureConfig.device_pattern` 默认写死 `"blackhole"`，`miniaudio_capture.py:191 _find_device` 按名字子串匹配、回退默认设备。voice node config 提供三级：

1. `device_pattern: str` — 名字子串（现状语义，空 = 不匹配回退）
2. `device_index: int | None` — 枚举 `miniaudio.Devices().capture` 按位置选（miniaudio 支持显式 `device_id`）
3. default — 系统默认输入设备

## 模块抽象约定 (2026-08-04)

> 会话收敛：主体在 host 层，channel 从 IoC 拿实现，node 是薄壳。核心不感知 matrix，协议层可换。

### 代码位置

| 层 | 位置 | 内容 |
|---|---|---|
| 核心 | `ghoshell_moss/host/listener/` | VoiceController contract + 实现（两轴状态机 + capture + asr + buffer）。依赖方向干净：只依赖 contracts/audio、asr，不感知 channel/matrix |
| channel | `ghoshell_moss/host/listener/channel.py` | VoiceChannel：`ChannelInterface.new(container)` 从 IoC 拉 VoiceController，command 内直调。channel 与核心同包——当前体量不足以独立成 channels 包 |
| node | `nodes/sensors/listener/` | 薄壳：IoC 装配 + provide channel + GUI 入口（webview 模式） |

IoC 机制已在 `channel_builder.py` 验证：`force_get_contract` (:108) / `with_binding` (:574) / `with_contract_factory` (:583) / `ChannelInterface.new` (:671)。

### VoiceController 契约

```python
# 强类型生命周期事件
class VoiceLifecycleEvent(BaseModel): ...
class StreamStateChanged(VoiceLifecycleEvent):
    state: StreamState
class AsrPartial(VoiceLifecycleEvent):
    utterance_id: str
    text: str
class AsrFinal(VoiceLifecycleEvent):
    utterance_id: str
    text: str
class BufferUpdated(VoiceLifecycleEvent):
    content: str

class EventHandler(Protocol):
    """每个事件类型一个方法，强类型投递，绝不 dict。"""
    def on_stream_state_changed(self, e: StreamStateChanged) -> None: ...
    def on_asr_partial(self, e: AsrPartial) -> None: ...
    def on_asr_final(self, e: AsrFinal) -> None: ...
    def on_buffer_updated(self, e: BufferUpdated) -> None: ...

class VoiceController(ABC):
    """语音输入核心契约。channel 从 IoC 获取，不感知 matrix/channel。"""

    # 控制面（低频，channel command 直调）
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def set_mode(self, mode: VoiceMode) -> None: ...
    async def set_config(self, config: VoiceConfig) -> None: ...

    # 事件订阅（强类型 + 注册 handler）
    def add_handler(self, handler: EventHandler) -> Disposer: ...

    # 状态查询
    def snapshot(self) -> VoiceNodeRuntime: ...
```

### 解耦到协议层（matrix 协议层并行推进，core 不动）

core 不感知 matrix。协议层（或 node adapter）注册 EventHandler → 收到强类型事件 → 卸载到队列 → 有序消费 → 广播：

- **出向（事件）**：`add_handler` → 入队 → 协议层有序消费 → SpeechTopic 广播 + AudioSignal 进 mindflow。首包 `complete=False` + 高优先级占座；尾包 `complete=True` + 同 utterance_id → same-id absorb（mindflow 机制已验证有单测：`test_attention_challenge.py:92`、`test_strength_zero_yield.py` 等）。
- **入向（控制）**：channel command 直调 VoiceController 方法，低频，不走队列。
- **顺序**：voice 核心是单一事件源，单 FIFO 队列天然保序；协议层单消费、不重排。

### 采集边界与配置（维持前议）

- 采集边界 = 节点自持：miniaudio 直采 16000，不经 transport/consumer/跨进程锁。
- 配置 = 读 node 内 config：10 开关 + device 三级选择（pattern/index/default）。
- 复用好件：miniaudio 设备枚举、VolcengineASR。不复用：transport 拆分、每帧 FFT、跨进程锁、44100→16000 resample。

## 执行路径与 Round 1 计划 (2026-08-04)

**迭代纪律（g1 式）**：node 创建 → 可用 → 模型可治理 → 多轮优化加功能。Round 1 只做"**不移交时仍然可用**"——控制主权默认 ghost（channel 治理），human 移交（#2）、manual gate 编辑块（#7）、自动重写（#8）、音频存储（#6）、webview GUI 全部后置。

### Round 1 — host 核心 + IoC channel + 薄 node（今天可用）

**目标**：`host/listener/` 核心可用（两轴状态机 + 采集 + asr），voice channel 模型可治理，真实语音一次 ASR 走通。

| 步骤 | 内容 | 产物 |
|---|---|---|
| R1-1 | 核心契约 | `host/listener/` VoiceController contract + 强类型事件 + EventHandler |
| R1-2 | 配置模块 | 读 node 内 config.toml：10 开关 + device 三级选择 |
| R1-3 | 两轴状态机 | 话语生命周期（idle→capturing→finalizing）× 闸口/buffer（staged→committed/dropped） |
| R1-4 | 采集/ASR 管道 | 采集边界自持（miniaudio 直采 16000）+ VolcengineASR（复用 listener 的 pump/silence-timeout/TTS gate 语义） |
| R1-5 | 控制 channel | voice channel（IoC 拿 VoiceController）+ mode/config 子 channel |
| R1-6 | 事件 adapter | 协议层注册 EventHandler → 队列 → 有序广播（SpeechTopic + AudioSignal 首包 complete=False / 尾包 complete=True 同 utterance_id） |
| R1-7 | node 薄壳 | `nodes/sensors/listener/` NODE.md + provide channel + GUI 入口（--mode headless/webview） |
| R1-8 | 验证 | `moss nodes run` 拉起 + moss-as-mcp 命令链路 + 真实语音一次 ASR |

**Round 1 不做**：#2 human 移交、#6 音频存储、#7 manual gate、#8 自动重写、webview GUI。gate 默认 auto（passthrough）。

## Implementation Progress

| Step | 内容 | 状态 |
|------|------|------|
| 1 | 四层状态机合约 — contracts.py + AudioSignal 扩展 | completed (R1) |
| 2 | L2 流层状态机 — idle→armed→capturing→finalizing→staged 核心循环 | completed (R1) |
| 3 | 模式实现 — PTT / enter / turn-taking / duplex 四种触发策略 | pending |
| 4 | commit 推回流层 — ASR is_final 与模式 commit 信号解耦 | pending |
| 5 | L4 ASR Nucleus 重构 — barge_in / attention 双 flag + 打断粒度 | pending |
| 6 | Preemptable 集成 — TTS channel 实现 Preemptable.attenuate/resume | pending |
| 7 | 门控接口 + 三种零智能实现 (passthrough / VAD / duration) | pending |
| 8 | 快捷指令 — 白名单匹配 + fatal_command 反射 + 安全约束 | pending |
| 9 | 说/听联动 — 半双工 TTS gate (AudioRuntimeTopic 读取) | completed (R1) |
| 10 | VoiceNodeRuntimeTopic — 四层状态快照 Topic | completed (R1) |
| 11 | 控制 channel — voice/mode/config 子 channel 树 | completed (R1) |
| 12 | 可视化 — TUI 状态条 (最小形态) + qt_screen window (正式形态) | pending (Round 2+) |
| 13 | listener node 落地 — nodes/sensors/listener/ NODE.md + main.py | completed (R1) |
| 14 | listener/ptt_listener 旧 app 退役 — 功能被 listener node 覆盖后删除 | pending (三触发模式等价验证后) |

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
   Topic + channel + 可视化 + NODE.md。聚合所有组件为 nodes/sensors/listener/。

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

## 2026-08-06 会话决策

### voice → listener 重命名

voice 是介质名，不是行为名。sensors 下都是行为（camera、listener），保持叙事一致。
重命名范围：
- `host/voice/` → `host/listener/`（核心 + capture + volcengine_asr + channel）
- `nodes/sensors/voice/` → `nodes/sensors/listener/`（薄壳 node）
- FEATURE.md 所有路径引用

channel 保留在同包（`host/listener/channel.py`），不独立为 `channels/` 包——当前体量不足以支撑独立包。

### 后续执行路径 (6 步)

1. 重命名 + FEATURE.md 修改 ← 本轮
2. 音频输入/输出依赖统一为 `moss[host]`，关联节点不做独立 venv
3. 相关能力 provider 进 **project 级 `MOSS.manifests.providers`**（基线能力，2026-08-09 修订）——
   原计划"进 mode"，但 CLI 走 `Matrix.new` 只加载 project + MATRIX manifests、不加载 HOST/mode。
   作为基线能力的 provider 必须进 project 级才能被 CLI 经 `Matrix.new` 看见。实现留在 host 作为依赖路径。
4. 配套做全套无状态机调试工具：play / 音频采样 / tts / asr，根据 mode 配置项来，CLI 作为 moss 无关的底层调试工具
5. 做 UI 无关的独立 listen node
6. 最后做完整的交互模式与可视化

---

## 2026-08-09 会话决策 — 音频 provider project 级迁移 (为 CLI 准备)

> 结对编程：人类架构师 + deepseek-v4-flash。对齐后动手——"执行计划以人的说法为准，FEATURE.md 里记录的是上一个模型实例的理解"。

### 背景

`moss audio contracts`（第一个 CLI 调试命令）需要经 `Matrix.new` 遍历核心音频抽象
(capture / player / speech / tts / asr)，打印 `contract -> instance | importError`。
但 `Matrix.new` 只加载 **project (MOSS.manifests) + MATRIX.manifests**，不加载 HOST/mode 层——
此前 speech 三件套声明在 HOST.providers，`Matrix.new` 看不到。

### 迁移内容（本轮完成）

1. **speech 三件套声明迁到 project 级**：`TTSServiceProvider` / `TTSSpeechServiceProvider` /
   `AudioPlayerProvider` 从 HOST.providers 移到 `MOSS.manifests.providers`（`.moss` + `stubs` 同步）。
   实现文件留在 `ghoshell_moss/host/providers/`（host 作为依赖路径，不建 `ghoshell_moss.audio`）。
2. **`audio_player_provider` 转 lazy**：`MiniAudioStreamPlayer` 和 `MatrixAudioTransport` 都移进 factory——
   `MatrixAudioTransport` 看似 light，但 `host/listener/capture/__init__.py` 顶层 import `miniaudio_capture`，
   顶层引用会经包 init 拉 miniaudio。全部 4 个 provider 模块顶层 import 链已轻（probe 验证无 miniaudio/websockets/httpx）。
3. **HOST 层移除**：default / system_test 各留 `AudioCaptureProvider`；stubs HOST providers 变空 header + 注释。
   避免 mode 后注册重复覆盖。
4. **manifest CLI 可见性修复**：`Project.discover()` 注册 workspace source 到 sys.path——discover 即自足，
   项目级 manifests 无需 bootstrap 可被 `scan_package` 发现。此前 `moss manifests providers` 对 MOSS.manifests
   一直显示 0（基建 provider 也看不见），是 manifests CLI 重建时就有的设计缺口。删除了环境细节单测
   `test_providers_inherited_from_matrix`（断言 stub mode 声明 ≥1 provider，属实现状态非协议契约）。

### `moss audio contracts` 命令（执行路径第 4 步，已落地）

`src/ghoshell_moss/cli/audio_cli.py` + 注册进 `main.py`（`depend_matrix` 守卫内）。

- 以 cli 身份 `Matrix.new("audio_cli", category='cli')` 声明为 node，**只建容器不 join 网络**
  （~0.22s，contracts 只需注册可用性）。全 sync，无需 asyncio。
- 对每个槽位调 `container.get_provider(contract)` 打印 **Provider 类**（注册可用性），**不实例化实现**
  ——不触发重 import、无副作用。tts/speech/player 已注册；capture/asr 返回 None 标 TODO
  （capture 在 HOST 待迁移，asr 无 provider / 抽象待 listener 重构）。
- 顺带修复 meta-help 解析器：单命令组（`registered_commands` 恰 1 个）时 typer `get_command`
  返回 `TyperCommand` 非 Group，`_show_command_help` 此前对 `help <group> <cmd>` 报错。

### 待办（下轮）——按优先级

**当前主线：player 文章做完**（体感可验证 → 波形可见 → tts→play 链路）。capture 迁移后置。

1. ~~**CLI `audio play`**~~ — completed (2026-08-09, 见下方会话决策)。
2. ~~**波形展示**~~ — completed：`audio play --waveform` 按 100ms 片段喂入渲染文本波形
   （human rich 面板 / `--ai` 纯文本行）。多帧波形依赖来源产出片段，与 tts→play 同构。
3. **tts→play 链路**——把 speech 的其余概念做全：stream_id、片段存储
   （`speech_storage` 拼接）、文本音频片段广播（PlaybackSample 是否 topic 化的第二层决定）。
4. **audio CLI 拆分**——`audio_cli.py` 已见膨胀（contracts + play + 合成/读取/渲染 helpers）。
   随 echo 等命令落地时拆 `cli/audio/` 子 package。
5. **capture 迁移到 project 级**（后置）——capture 放 HOST 一开始就是错的，`Matrix.new` 路径
   看不到 HOST。连同 listener 抽象一起处理。

## 2026-08-09 会话决策 — player 实际播放可感知 (PlaybackSample + observe)

> 结对编程：人类架构师 + deepseek-v4-flash。这是对"不满意的 audio topic"的两层拆分：
> **第一层 (本轮)** = player 实际播放的可感知走回调，**第二层 (可选下一步)** = 如何消费
> （用 topic 广播？）——技术上独立的决定，本轮不做。CLI 锚点：`moss audio echo`
> （听 n 秒 → 播 n 秒 → CLI 画文本波形），两层最终都会出现在 CLI 里。

### 设计修正（align 后落地，推翻了我最初的实现）

我最初把 `observe` 做成 **stream 作用域 + 自动移除**（`_stream_pending` 计数，
stream 片段播完即摘除观察者）。用户 review 指出这是过度设计，修正为：

1. **player 不做 stream 生命周期追踪**。stream 真正的生命周期在**治理层**——明确知道
   它的 speech 或外层控制 `clear()` 的节点。player 只提供简单订阅（`observe` 返回
   unsubscribe），何时结束由治理层管理。
2. **observe 是全局的，不是 stream 作用域**。stream 身份放进**数据**里（PlaybackSample
   携带 stream_id），不做成注册作用域——否则逼 player 维护按 stream 的内部状态。
3. **新增 fragment_id（拼接身份）**。发送方在 `add()` 传入（通常是自增整数），消费方
   对齐回调判断哪些片段拼接到一起，方便存储。这是链路里更关键的一环——形如
   `speech_storage` 专门负责拼接动作。
4. **优化点**：无观察者注册时不计算 PlaybackSample（跳过频谱计算）。

### 落地内容

- `contracts/speech.py`：
  - `PlaybackSample`（pydantic model，可序列化，未来可广播为 topic）：`stream_id /
    fragment_id / timestamp / duration / rms_db / peak / bands{bass,mid,high}`。
  - `StreamAudioPlayer.add(chunk, ..., stream_id="", fragment_id="")`。
  - `StreamAudioPlayer.observe(callback) -> Callable[[], None]`（全局订阅，返回 unsubscribe）。
- `core/speech/base_player.py`：队列项改为 `(pcm, stream_id, fragment_id)` 元组；
  `_audio_worker` 写设备后 `_dispatch_playback_sample`（真正写入时刻，非入队时刻）；
  `_compute_playback_sample` 计算轻量频谱摘要（rms/peak/3-band）。无观察者时跳过。
- 测试 `tests/ghoshell_moss/host/speech/test_player_playback_sample.py`：触发、拼接身份
  透传、全局观察、unsubscribe、无观察者不崩溃、多观察者。host+default 1282 全绿。

### 下一步（待定，不是本轮 scope）

- **第二层消费**：PlaybackSample 是否广播为 topic、怎么广播——独立决定。
- **CLI `audio echo`**：听 n 秒 → 播 n 秒 → 文本波形图，同时消费两层。
  此前会话的崩溃点就发生在这个操作上，迁移时先对齐再做。

### 关键认知（避免未来重蹈覆辙）

- **"不全局 import" 指 provider 模块顶层不拉重依赖**（concrete import 隔离），abstract import 隔离意义不大。
- **三层 manifest 加载范围**：`Matrix.new`/CLI = project + MATRIX；Host runtime 额外叠加 HOST。
  作为基线能力的 provider 必须进 project 级才能被无 Host 的 CLI 看见。

## 2026-08-10 会话决策 — ASR 通用内核契约提炼

> 结对编程：人类架构师 + deepseek-v4-flash。从 volcengine 实现提炼可迁移重实现的通用内核。

### 动机
现有 contracts/asr.py 只有薄 ASR ABC + ASRResult(text, is_final)。配置面（凭据、端点、音频格式、
调参）全焊在 VolcengineASRConfig 里 — 任何非 volcengine 重实现、provider 注册、CLI asr 工具
都无法脱离 VolcengineASRConfig 工作。TTS 侧有对称的 TTSInfo + TTSBatch + TTSItem，ASR 侧没有。

### 提炼的内核

1. **ASRInfo** — 镜像 TTSInfo 的反射面。sample_rate/bits/channel/model（音频输入契约 + 模型身份）+
   params_schema/params（可调行为参数的 json schema 与当前值，各实现暴露自己的 params BaseModel，
   契约只背 dict）。模型先 get_info() 自解释，再 configure() 调行为旋钮。

2. **ASR.configure(params: dict)** — 会话级调参，作用于下一次 recognize()。校验与取值空间
   留给各实现的 params BaseModel。火山引擎：每次 recognize() 新建 WS + init 下发全参 → configure()
   只换 self._config.params，下次连接自然携带新参数，协议零改动。

3. **VolcengineASRParams** — 火山引擎专属行为参数：end_window_size, force_to_speech_time（从
   硬编码 1000 提进 config）, enable_punc, enable_ddc。model_name 不在此列 — 模型身份每实例
   固定，模型选择由工厂/provider 负责（让有状态实现自己切换 model 会协议冲突）。

4. **controller 不再硬编码 16000** — VoiceControllerImpl 的采样率从 asr.get_info().sample_rate
   推导，ASR 创建移至 __init__（get_info() 先于 capture 以便 capture 取对采样率）。

### 不做
- model_name 不是 runtime 旋钮，不入 params。不同 model = 不同工厂实例，未来 ASRFactory.get(model)。
- provider 注册延后 — 契约先立住，provider 是下游。
- enable_ddc 在 params 但火山协议仍未发送（预存兼容，不发是因为当前 bigmodel 接口不需要）。

---
*架构设计: claude-fable-5 (opus-4-7) 与人类架构师, 2026-07-28*
*基础调研: audio-capture FEATURE.md (DeepSeek V4 + Claude Opus 4.7) — 已完成的音频感知全链路*
*碰撞记录: 本会话对话 — 分层拓扑推演、交互模式收敛、安全边界讨论*
