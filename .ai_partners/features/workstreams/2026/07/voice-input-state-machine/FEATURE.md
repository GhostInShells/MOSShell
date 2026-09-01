---
title: Voice Input State Machine — 语音输入全状态机与交互模式
status: in-progress
status_note: 'CLI 基建完成 (2026-08-11)：ASR provider 注册 (AudioASRProvider, project 级)；moss audio asr 命令 (live 流式 / --ai / --json 三种模式, 多 turn 云端 VAD 判停, 44100→16000 采样率桥接)；ASRResult 增 error 字段 (server error 不再静默)；protocol.py 空 payload GZIP 标志修复；audio contracts 5 槽位全部 OK. 监听 CLI 基建就绪, 无独立 listener CLI — 下一阶段为 node-level voice-input 感知节点. 2026-09-01: 协作调整为人类架构师手改实现+模型协助/review; signal 四态语义 (首包/分句中/分句/尾包) 与 ASR 会话对象方向已收敛, 详见文末.'
priority: P0
created: 2026-07-28
updated: 2026-09-01
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

## 2026-08-11 会话决策 — ASR Provider 注册 + CLI asr 命令

> 结对编程：人类架构师 + claude-opus-4-7。Provider 注册 + CLI 命令 + 错误可观测性 + 协议修复。

### 动机

ASR 通用契约已在 08-10 提炼完成，但 contracts 的 asr 槽位仍是 "no provider registered"，
CLI 的 asr 子命令尚未实现。本轮补齐这两块，使 `moss audio contracts` 五槽位全绿、
`moss audio asr` 可用。

### 交付

1. **AudioASRProvider** (`host/providers/audio_asr_provider.py`) — singleton=True，
   从 VolcengineASRConfig 解析环境变量创建 VolcengineASR。注册在 project 级
   (`.moss/src/MOSS/manifests/providers/`)，CLI 无需 Host 即可看见。

2. **`moss audio asr` 命令** (`cli/audio/asr.py`) — 三种输出模式：
   - 人类模式：`\r` 就地更新 partial，final 换行提交，`---` 分隔 VAD turn
   - `--ai` 模式：只输出 final + `---` 分隔符（bash 友好）
   - `--json` 模式：每结果一行 JSON（text/is_final/elapsed/turn/error）
   - 参数：`-t/--timeout` (默认 60s)，`-d/--device`，`-o/--save`，`--json`

3. **ASRResult.error** — 新增 `error: str = ""` 字段。VolcengineASR server error 经此
   传给 CLI 显式输出，不再仅藏于日志文件。`iter_with_silence_timeout` 遇 error 立即
   透传，不等超时。

4. **修复**：
   - `create_audio_only_request`：空 payload 时不标 GZIP 压缩（服务端解压 EOF）
   - bridge 跳过 `len(pcm) == 0` 的空 chunk
   - `_audio_gen` 用 `wait_for(get(), 0.5s)` 防遗弃 waiter 污染下个 turn
   - bridge 内 44100→16000 线性插值重采样（对齐 BaseAudioStreamPlayer.resample）

### 关键决策

- **CLI 不做 VAD** — 云端 VAD 判停本身就是 CLI 探测的协议行为。CLI 只收音→喂 ASR→流式显示。
- **listener 无独立 CLI** — ASR 命令在 `moss audio` 子树下，作为协议探测面。感知节点
  (voice-input node) 是下一阶段的事。
- **Provider singleton=True** — VolcengineASR 每次 recognize() 独立建 WS，无累积状态，
  单例安全。若未来有状态实现，改 False 即可。
- **默认 60s timeout** — `-t` 是全局兜底，不设则 60s。用户停止说话后由 silence timeout
  (5s patience) 结束 turn，新 turn 若无语音则快速过。

---

## 2026-08-12 会话决策 — audio 架构讨论清单（整理）

> 人类工程师 + deepseek-v4-flash 对 audio 架构的收敛讨论。议题太具体，不落 `.design/`，只在此表列备忘。
> 状态：✅ 定案 / 🔶 待定 / 🔴 开放。

### 议题清单

| # | 议题 | 结论 / 方向 | 状态 |
|---|---|---|---|
| 1 | **player 单例** | `AudioPlayerProvider` 改 `singleton=True`，对齐 capture（拥有输出设备）。speech 为单例锚点持有 player | ✅ 定案 |
| 2 | **speech 开放观测接口** | speech 不直接耦合 topic——开放观测接口，topic 接线放外侧。speech stream 可持有父层传入的 callback。speech 拥有 player 回调并决定怎么用（折算/广播） | 🔶 待定（接口形态） |
| 3 | **conversation / dialog topic** | 统一"听/说"双侧的话语 topic，全网可用数据结构，`TopicWindow` 可 buffer，对 ASR 特别重要。与现存 `SpeechTopic` 是演进/吸收关系（吸收范围待定） | 🔶 待定（吸收范围） |
| 4 | **ASR 开放词表 + 上下文 getter** | 是否开放词表（hotword）与上下文 getter 两个接口 | 🔴 开放 |
| 5 | **signal / topic 迁移 types** | 迁移到统一 types 目录合理（纯数据结构），迁移成本低。先记下，后移 | ✅ 方向认同（迁移后置） |
| 6 | **AudioSignal 数据结构化** | 无 messages、纯结构化 SignalMeta。命名 `AudioSignalMeta`（可能直接 asr signal meta）。可扩展优先：未来声纹/用户识别/音频事件检测 | ✅ 定案（方向），实现后置 |
| 7 | **listener 状态机** | 优先级最高，但先想通 3/4/5/6 再动手 | ✅ 排序 |

### 补充议题（检查遗漏时补入）

| # | 议题 | 说明 | 状态 |
|---|---|---|---|
| 8 | **中断折算语义** | 观测接口设计须包含：`clear()` 折算已播半句、被打断句带 interrupted 标记发布、cancelled task resolve。实现时别丢 | 🔴 待并入 #2 |
| 9 | **说侧运行时状态 is_speaking** | 回声抑制/barge-in 依赖。conversation topic 是事件流，is_speaking 是状态快照——确认 #3 不吸收状态需求 | 🔴 待定 |
| 10 | **AudioRuntimeTopic 双发布者拆解 + AudioPlaybackTopic 去留** | capture heartbeat 与 speaker gate 同 schema 压扁；AudioPlaybackTopic 目前 CLI-only、未导出。与 #3 一起定 | 🔴 待定 |
| 11 | **分句器位置** | speech FEATURE 遗留开放问题，决定说侧进 conversation topic 的分句粒度 | 🔴 待定 |

### 说侧四层对齐与基建盘点（自 speech-protocol-alignment 吸收, 2026-08-12）

说侧 (output) 对齐骨架——与听侧四层分层状态机正交：

```
L1 interpreter 级    一轮 logos = 一个 Interpretation, 含 0..n speech command
L2 command-stream 级 一个 say/__content__ = 一个 SpeechStream (batch_id = task.cid)
L3 tts 分段级        句子 = TTSItem{text, audio, duration}, 携带 (cid, seq)
L4 play 分段级       player chunk + 播放游标, observe/on_play 逐片回调
```

对齐本质：每层持上层引用，中断折算沿链向上折（L4 游标 → L3 已播句 → L2 已播文本 → L1 本轮真实 logos）。

分层广播频率原则：

| 层 | 频率 | 载体 |
|---|---|---|
| 帧级 (10~50ms) | 高频 | stream 协议，不进 Topic |
| 句级 | 秒级 | TopicService |
| 状态快照 | 持续覆盖 | TopicWindow(max_size=1) |
| 轮次级 | 每轮 | Interpretation 记账 |

说侧基建盘点（已核实，折算 #8 的实现依赖）：

- `contracts/speech.py`：TTSItem 已是文本-音频对齐单元；`StreamAudioPlayer.add()` 返回播放结束时间戳；`observe(PlaybackSample)` 逐片回调（真正写入时刻）；`SpeechStream.feed` 写回 cmd_task.tokens——**"喂了什么"，不是"播了什么"**。
- `Interpretation` 五本账 (compiled/pending/success/cancelled/failed)；`executed_inputs` 只累计成功 task；**`on_done_task` 的 result 合并对 cancelled task 同样生效**——被打断的 say task 可经 result messages 送进观察上下文，折算回报通道现成。
- SpeechStream 生命周期四元门控 commit/synthesis/play/close 可乱序；新 stream 的 start_play 关闭上一个 stream。

## 2026-08-12 会话补充(二) — topic 结构定案与状态/事件分家

> 人类工程师 + deepseek-v4-flash 第二轮。上一轮列议题清单, 这一轮定案 topic 结构与载体。
> 说侧设计 (四层对齐、折算) 已在「2026-08-12 会话决策 — audio 架构讨论清单」消化, 本轮补听侧 topic 面。

### 三个跨侧协议 topic 定案

| # | topic | 语义 | 载体 |
|---|---|---|---|
| 1 | **分句 topic** (SpeechTopic 改造, 命名候选 ConversationTopic) | 一句实际说出的话, 听/说双侧共享的会话句段。role/name/sentence_id/batch_id/text/lang/audio_key/timestamp/**interrupted**/seq; address = meta.sender | **Topic** (事件面) |
| 2 | **AudioPlaybackTopic** | 说侧播放元数据广播 ~20Hz, **不含二进制** (无 PCM)。stream_id/fragment_id/sample_rate/rms/peak/spectrum_bins。数字人口型、ghost 声波 | **Topic** (事件面) |
| 3 | **AudioRuntimeTopic** | listener ↔ speech 半双工门控。is_speaking(speaker)/is_capturing(mic), device_name 区分 | **Parameter** (状态面) |

### 事件/状态分家原则 (本轮核心判据)

**判据: 消费者是否需要"自己启动之前的值" (前值)。**

- **依赖前值 → Parameter** (推, host 广播真值, 本地被覆盖; 启动时 query 一次拿到前值)。
- **不依赖前值 (只要未来值) → Topic** (事件流)。

状态面 (Parameter): VoiceNodeRuntimeTopic、AudioRuntimeTopic 门控。
事件面 (Topic): 分句 topic、AudioPlaybackTopic。

前值需求的具体场景: KD9 ghost 世界模型 (中途接入必须立刻知道"正在收音")、半双工门控
(listener 在 ghost 已说话后接入须知道"在说", 否则 TTS 回声)、TUI (启动即渲染当前状态)。

### VoiceNodeRuntimeTopic 生命周期判定

依赖前值 → **parameter**。字段拆解:
- queryable facts → Parameter: running / mode / stream_state / gate / device /
  barge_in_target / staged_text / attention_occupied / buffer_depth。
- asr_partial → **不是状态**, 瞬态实时转写 (sub-second 写, 只有未来值有意义),
  走 AsrPartial 事件流, 不混入状态 Parameter。

前置条件: **parameter 必须有变更回调 on_change**——当前接口只有 get/set/version/remove,
缺失, 待补。独立 workstream: `parameter-host-truth`。

### AudioPlaybackTopic 强约束

**不含二进制是强要求** — 它和 PlaybackSample 不是一回事 (PlaybackSample 带 PCM, 本地
observe 回调; AudioPlaybackTopic 是无 PCM 的广播元数据)。数字人动口型、ghost 声波是
跨 cell/跨进程消费者, 所以必须 TopicService 广播, 不是本地回调。

### signal 不算 topic

AudioSignalMeta (#6) 是纵向协议 (shell→ghost, mindflow envelope), 和三个横向 topic 是
两条协议轴。听侧用户句同时走两条: 广播进分句 topic (横向) + 进 signal (纵向进 ghost 感知)。

### 两个观测方向 (设计定后最先落地)

1. **topic 监听脚本**: 订阅三个 topic + declare 各 parameter 轮询版本, 实时 dump,
   验证数据面通不通, 在场景装线前把协议面钉死。可做成 `moss audio watch` 类命令。
2. **listener 极简 TUI**: 直接进命令行, 复用 shell/ghost TUI 基建, 渲染状态 (Parameter)
   + 逐句流 (分句 topic) + 波形 (Playback)。

### 清单状态更新

- #3 conversation topic → ✅ 定案 (分句 topic, 事件面)。
- #9 is_speaking 状态快照 → ✅ AudioRuntimeTopic 门控状态 → Parameter。
- #10 AudioRuntimeTopic 双发布者 + AudioPlaybackTopic 去留 → ✅ AudioPlaybackTopic **保留**
  (非二进制硬约束); AudioRuntimeTopic 双发布者 = 门控对称两侧, 归一为 Parameter。
- #6 AudioSignalMeta → ✅ 方向 + 载体确认 (纵向协议, 不算 topic)。
- 新增: VoiceNodeRuntimeTopic 判定 → Parameter, on_change 缺失 → parameter-host-truth。

## 2026-08-13 会话决策 — flag 驱动 pipeline + 主干优先

> 人类工程师 + deepseek-v4-flash。从"状态机丢决策"的失望中重捞设计，收敛为实现范式。

### 命名

- **`VoiceStateMachine` 弃用 → `listener`**。旧名用实现机制（状态机）命名，错了。
  新名命名器官（一个会听的单元）。命名与实现范式是同一件事的两种表述。
- **listener 不继承 `VoiceController` ABC**。那个 ABC 的形状（start/stop/set_mode/
  set_config/add_handler/snapshot）是"状态机控制方法"的形状，本身就是旧范式。
  继承它 = 保留了旧契约，只是改了实现类名，flag 驱动设计没有落地。

### 核心判断：listener = flag 驱动的并行管线 (CSP)，不是中央状态机

- listener 开放的**不是状态机切换，是若干控制接口**——每个接口可能是 bool、可能有参、
  大部分是纯 flag（listening / mode / gate / barge_in...）。
- **状态机没有消失，是被分布式了**。CSP 的 C 是 Sequential：每个环节是带局部状态的
  顺序进程（ASR 环节内部：累积 → 收 commit → finalize → 发 signal）。但**不存在一个
  中央的、试图协调一切的状态机对象**。
- 变更 flag = cancel 某环节 task 重建，或改 flag（如关 listening 只让 capture 时间片
  里暂停转发，麦克风设备生命周期不动）。

### 协议化动机（最核心）

**协议化让"环节"从"进程内一个步骤"变成"matrix 里的一个细胞"**——同进程/跨进程从
代码决策降级为部署决策。每个环节的入口/出口若是协议（topic/stream），capture 在进程 A、
ASR 在进程 B 对环节代码透明。这溶解了 KD8 的"单一进程"决定（KD8 要单进程是因"模式切换
要协调捕获时机+送入时机，跨进程是门控复杂度根因"——flag 协议化恰好把这两者都变成
协议化 flag，根因被消解）。

### 主干优先（本轮实现范围）

- **不做 3-5 层节点**。先做**主干**：capture → ASR → signal，同时在主干上广播 topic
  （ASR final → ConversationTopic），旁路（声纹/声音事件）以后仿照装线。
- **控制面先不做 Parameter**。先让 listener 自己有正确接口（start/stop/set_mode/
  snapshot），协议控制（Parameter + on_change）另做（parameter-host-truth）。
- **阶段**：listener 作为 CLI node 独立可跑 → 复刻逻辑成 node + 配 GUI。

### KD3 关键：is_final 与 commit 分离（之前丢的核心）

`is_final`（ASR 自己分句）与 `commit`（触发决定"说完了"）是**两个独立事件**：

| 模式 | commit 触发 | is_final 行为 |
|------|------------|--------------|
| turn_taking | 云端 VAD（is_final 重合） | 触发 commit |
| PTT | 松手 | is_final 不分句提交，松手才是唯一 commit |
| enter | 回车 | is_final 不分句提交 |
| duplex | VAD 静音（常开） | 触发 commit |

当前 controller.py 把两者混为一谈（`if result.is_final: on_asr_final`），这是模式
触发无法成立的根因。主干必须把"ASR 产出 is_final"与"触发发 commit 事件"拆开。

### 装线难点（已盘点材料）

装线难在：类型纪律（进程内 object vs 跨协议 bytes）、生命周期（慢消费者/崩溃/重连）、
顺序（seq/commit 事件后于音频）。材料已备：AudioChunk / ConversationTopic /
AudioPlaybackTopic / ASRResult 类型已定，AudioSequentialConsumer 背压已有。缺两件：
commit 事件协议（trigger → ASR finalize 通道）、flag 作为 Parameter 的 on_change。

### 失败模式记录（实现教训，2026-08-13）

上一模型实例在讨论定稿后的**第一步实现**就犯下两个不可接受的错误，直接导致 listener
抽象由人类工程师接手（本任务转为"人类做抽象 + 模型 review"）：

1. **`Listener(VoiceController)` 继承旧 ABC**。`VoiceController` 的形状就是状态机控制
   方法的形状，继承它等于保留旧范式、只改实现类名。正确做法是 listener 有自己独立的
   公开面（flag 控制接口），与旧契约彻底脱钩。
2. **`__import__("...", fromlist=[...])` 魔法 import**。在 dispatch 热路径里动态 import
   契约类，是赶交付的屎山症状——没想清模块结构就动手。
3. （同源）PTT 用同一个 commit_event 既当"按下"又当"松开"，语义过载。

**根因**：从架构讨论（CSP / 协议化）直接跳到写代码，没有先把"listener 的具体公开面"
定下来。这是 CLAUDE.md 记录的"交付优先堆屎山"失败模式，与 ground 开发的 silent todo
同源——表面交付、设计未落地。

**重建要点（给 review / 下一实现实例）**：
- listener 的公开面 = flag（listening / mode / gate...），不是状态机方法，也不继承
  `VoiceController`。
- 先定"环节协议入口/出口 + flag 表面"，再写代码；协议未定不许动手。
- KD3 的 is_final / commit 分离是模式触发的前提，云端 ASR 会话生命周期如何承载"松手
  才 finalize"必须先想通，不许静默跳过或糊弄。

## 2026-09-01 会话记录 — 当前理解（非决议）与协作方式调整

> 本段记录本次会话的**技术理解**，性质是"当前理解"而非"已定决议"。实现将由人类架构师
> 集中手改，后续模型实例据此提供协助并最终 review。本段只记技术目标上的关键点，不记情绪。

### 协作方式调整（声明）

- 本 workstream 的下一步实现由**人类架构师集中手改**，模型不主导实现。
- 模型角色调整为：① 实现过程中对具体问题协助；② 最终 review 实现是否退回"capture→ASR 泵"
  或漏掉下述已钉契约。
- 本段价值：给后续模型实例提供 review 人类方案所需的上下文，而非一份待执行计划。

### A. 命名（开放，未定）

- `AudioSignal` 命名是错的——"audio" 是介质名不是行为名（同 voice→listener 教训）。
- 候选：`ListenerSignal`（域=聆听感知单元，罩得住 wake_word/alert/shortcut 等非 ASR 动作）
  或 `AsrSignal`（scoped 到 ASR turn，但非 ASR action 会无家可归）。

### B. ASR signal 四态语义（理解已收敛；命名与编码未定）

四态：**首包 / 分句中 / 分句 / 尾包**。

| 态 | 语义 | text |
|---|---|---|
| 首包 | turn 开始 | 可空 |
| 分句中 | partial，**全文 replace（非 delta）** | 当前最佳全文 |
| 分句 | 引擎 definite（句稳定） | 该句最终文本 |
| 尾包 | commit（完整提交的句子） | 该 turn 完整累积文本（自包含） |

身份两层：
- 1/2/3/4 共用一个 turn 级 id；命名候选 `turn`（非 `session`——MOSS 已占用；非 `batch`——撞 TTSBatch）。
- 3/4 共用分句 `segment_index`（引擎 result_index）。

**delta 已死**：ASR partial 非单调重写，partial 只能是"全文 replace"，不做 delta；分句 index + 对话 uid
取代增量排序。

编码方式未定：四态语义 enum（直观，consumer 一眼读懂）vs `turn_phase(start/middle/end)` × `is_final`
两轴（更贴引擎原生字段）——本次倾向 enum，未拍板。

时间戳：`start_ms`/`end_ms` 可拿到——火山 `result.utterances[].start_time/end_time`（ms，流相对），
当前 `_parse_result` 丢弃了 utterances 数组。两时间概念别混：引擎相对（start/end）vs 到达墙钟（envelope `meta.created_at`）。

### C. 当前 ASR 抽象的致命缺陷（已确认）

"只出不入 + 直接耦合 capture"：`recognize(audio_chunks: AsyncIterable[np.ndarray]) -> AsyncIterable[ASRResult]`
是单向拉取生成器，无控制通道（commit/stop 无法注入），输入是裸 numpy PCM。

- 嘴（TTS/`TTSBatch`）已是会话对象（`feed/commit/items/close`）；耳朵应镜像：`ASRSession`（`feed(AudioChunk)/commit()/results()/close()`）。
- commit 机制在 wire 层存在：`send_audio(is_last=True)` = 负序号 = 结当前轮、提前出 final（不等 VAD）。
  但 `_send_loop` 只在流末发一次，抽象层无 commit 入口。

### D. 当前代码把 Listener 实现成了 ASR（已确认）

- `controller.py:_run()` 是 capture→ASR 泵，不是 Listener 状态机。
- `is_final` 当 commit（`break`）+ `_drain_queue` 丢音频 → 多句 turn 丢。
- 四 mode（PTT/ENTER/TURN_TAKING/DUPLEX）死配置：`set_mode` 只改字段，无转移逻辑读 mode。
- L4 无 signal 发射进 mindflow。
- KD3（is_final vs commit 解耦）未实现。

### E. 写对状态机的三个前置（历次唯一被跳过的步骤）

1. 转移表 + 每条边的事件触发（mode commit，**非** ASR is_final）。
2. mode→commit 映射：PTT=松手 / ENTER=回车 / TURN_TAKING=VAD 静音 / DUPLEX=VAD 常开。
3. ASR 会话边界：`feed/commit/results/close`，吃 `AudioChunk`，不吃裸 np.ndarray。

### F. 本轮已落地的小改动

- `Matrix.new` 加 `singleton: bool | None = None` 参数，去掉 `if not persist: singleton=False` 强制；
  singleton 与 persist 正交。audio CLI 节点默认 singleton=True + persist=False（`core/blueprint/matrix.py`）。
- 未修：`AudioPlayerProvider.singleton()` 仍返回 False（08-12 已定案改 True，见 `host/providers/audio_player_provider.py`）。

### G. 关键澄清

- 火山 `definite=true` = 句稳定，**不是**流结束；负序号（`is_last=True`）= commit 当前轮。
- Realtime API（`v1/realtime?model=bigmodel`，OpenAI 兼容）有 `input_audio_buffer.commit` +
  interim `result`（累计 replace），与四态 1:1 对应，是备选链路（非必需）。

---
*架构设计: claude-fable-5 (opus-4-7) 与人类架构师, 2026-07-28*
*基础调研: audio-capture FEATURE.md (DeepSeek V4 + Claude Opus 4.7) — 已完成的音频感知全链路*
*碰撞记录: 本会话对话 — 分层拓扑推演、交互模式收敛、安全边界讨论*
