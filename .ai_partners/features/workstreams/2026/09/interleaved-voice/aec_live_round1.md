# AEC live 第一轮：生效了吗 —— 没生效，但问题在接线不在算法

- 时间：2026-09-22 05:40–05:46 (+8)
- 场景：单进程听+说（host runtime 自带 capture + MiniAudioStreamPlayer），
  macOS，capture 16k/1ch/50ms，player 16k/1ch（两者同率，无重采样介入）。
- 触发：人类用户实时验证 "回声消除生效了没有"。

## 现场证据

| 时刻 | 事件 |
|---|---|
| 05:42:08 | 我 emit `<say>`，`TTSSpeechStream` 起播（player 16k） |
| 05:42:13 | 我的 say 被 `CLEARED`（interpreter cancelled 1）——**被自己的回声打断** |
| 05:42:16 | listener 收到 ASR：`有回声消除，是有的，Web RPC、AEC。` ≈ 我原句开头逐字 |
| 05:40:59 | 更早一轮同现象：`我在，回声测试 —— 一、二、三。` → ASR `123。` |

结论：live 端到端回声抑制 ≈ 0 dB，而离线 `aec_alignment_probe.py` 实测稳态 ~16 dB。
两者差别只在 **far/near 对齐** 这一层 —— 所以先查接线，别动算法。

装线本身是通的：`moss audio contracts` 显示 speech = `TTSSpeechServiceProvider`（实例是
`BaseTTSSpeech(TTSSpeech)`），`_wire_aec_far` 的 `isinstance(speech, TTSSpeech)` 门因此打开
（`host/moss_runtime.py:658-690`）。

## 两个根因（都能落到行）

### 1. far 环是「取最新」而不是「按消费推进」——render 时间轴被压平

`host/listener/capture/webrtc_aec.py:79-95` `_far_window()` 每次返回环里**最新的** 10ms；
而 `process()`（同文件 64-77）把一帧 near 按 10ms 切开逐块调 AEC。

- capture 帧 = `AudioCaptureConfig.frame_duration_ms = 50`（`contracts/audio.py:167`）
  → 一帧 800 样本 = **5 个 10ms 块**；
- 这 5 次 `_far_window(160)` 取到的是**同一段** far（环只追加、不消费、期间无新 push）
  → 每个采集帧喂给 AEC3 的 render 是 5 份重复的最新 10ms，下一帧再整跳到新的最新 10ms。

AEC3 要靠 render/capture 两条时间轴做 delay estimator + 自适应滤波。render 轴被这样
「碎帧化 + 停滞」之后，滤波器不可能收敛 —— 这就是 live ≈ 0 dB 的直接原因。离线探针
按 10ms 严格对齐喂帧，所以它拿到 16 dB。

### 2. far 的推送时刻不是出声时刻，且按片段突发

`core/speech/base_player.py:289-298`：

```python
self._audio_stream_write(audio_data)      # 只是放进 player._data_queue
for callback in self._on_play_callbacks:  # far 在这里被 push
    callback(audio_data)
self._wait_consumed(audio_data)           # 只是按片段时长 sleep（201-214），不读设备时钟
```

- far 在「写进队列」时刻推，真正出声还要经过 miniaudio 设备缓冲；
- `_wait_consumed` 是模拟播放时钟（`time.sleep(duration)`），没有设备消费计数；
- 于是 far 相对 near 有一个 ≈ 设备缓冲 + 片段粒度的常量前置 + 突发抖动。

常量前置本可由 AEC3 的 delay estimator 吸收；但叠加根因 1 之后，estimator 看到的是
「重复 + 跳跃」的 render，连常量都无法锁。

## 修法（已实施，2026-09-22 05:55）

只动实现，不动 surface，不动装线：

- `src/ghoshell_moss/host/listener/capture/webrtc_aec.py`
  - `_far_window()`（每次取环里最新的 10ms）→ `_take_far()`（**按序消费**：每个 10ms
    near 块取走 far 行首的 10ms，不足补零）。
  - 依据：near 的消费量本身就是时钟 —— render 与 capture 因此严格同速前进，AEC3 的
    delay estimator 只需吸收一个常量偏移（它本来就是干这个的）。原来的「取最新」
    让 50ms 采集帧的 5 个 10ms 块拿到同一段参考，render 轴碎帧化，estimator 无从锁。
  - `push_far` 只追加不覆盖；`far_capacity_s` 仍是防御上限（现在溢出会丢**未消费**的
    参考，属异常工况，已在 docstring 标明）。
- 未改：`AcousticEchoCanceller` surface（`push_far` / `process`）、`_wire_aec_far`
  装线（`stream_delay_ms=0` 保持不变）、capture/player 的 I/O 样貌。
  这些先不动 —— 修完看真机结果再决定要不要碰 far 的推送时刻（根因 2）。

### 证据

`tests/ghoshell_moss/host/listener/test_webrtc_aec.py` 新增 live 节拍用例
（far 250ms 片段突发 + near 50ms 采集帧 + 60ms 延迟回声 + 两个早期反射）：

| 实现 | live 节拍 ERLE | 严格对齐喂帧 ERLE |
|---|---|---|
| 改前（取最新） | **2.9 dB** | 9.6 dB |
| 改后（按序消费） | **10.0 dB** | 9.6 dB |

- 改前的 2.9 dB 与真机现象吻合（自说自话被 ASR 逐字抄回、自己的回声打断自己）。
- 改后 live 节拍与理想对齐喂帧同级 —— 断言写成「live 节拍不差于对齐喂帧 1dB 以上」，
  而不是硬编码 dB 魔法值。
- `pytest tests/ghoshell_moss/host` 141 项全过；ruff 干净。

> 附带发现：`aec_alignment_probe.py:489` 的 live 探针里是同一个 `_far_window` ——
> 生产实现是从它移植的，所以这个"取最新"的假设一路带进了运行时。探针的 offline
> 分支按 10ms 严格对齐喂帧，所以它当初测出 16dB、没测出问题。

## 待办 / 残余风险

1. **真机验证**（下一步）：重启 runtime 后自说一段，看 ASR 还抄不抄回来、我还会不会
   被自己的回声打断。这是唯一的最终判据。
2. 根因 2（far 在**写队列**时刻推，不是出声时刻）这次没碰。设备缓冲是个常量偏移，
   estimator 应当吸收；若真机还有残余泄漏，下一步就是把它挪到消费侧
   （`_dispatch_playback_sample` 已经是消费时刻）或给 `stream_delay_ms` 一个粗 hint。
3. 播放被 `clear()` 打断时，环里未消费的 far 其实从未出声；修好后它会被当作"参考"
   喂进去（对应一段不存在的回声）。影响应该温和（滤波器只会不更新），但真机若发现
   **句尾**泄漏，这是首选嫌疑 —— 那时给 canceller 加一个 stream flush。
4. 建议的 ERLE 探针（把"生效没有"变成数字）仍未做；真机判据目前只能靠耳朵 + ASR。

## 附：关于「兜底方案 C」的取舍（本轮讨论结论）

说时闸打断、不闸听：能立刻止住「自己的回声打断自己」，但副作用有 —— 播放期间人类
真打断失效、误差窗口双向（估短了漏、估长了吃掉真打断）、礼仪语义变脏（`onset.interrupt`
变成依赖对方状态）、且治不了回声**文本**污染（我还是会听见自己）。

人类架构师定调：**别搞复杂了，先修 AEC**。C 因此不做。相关分析留给以后再需要时看。
