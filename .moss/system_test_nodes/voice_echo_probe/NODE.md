---
name: 'voice_echo_probe'
description: '单进程 听+说 共存与回声实测 — 跑完即退的纯观测节点: 启动 → 说一句 → 观测 → 退出'
singleton: true
exec:
  command: python
  args: main.py
---

单进程 听+说 共存与回声实测 node — **跑完即退的纯观测**。一个进程里同时拉起耳朵
(`listener`: capture + ASR) 和嘴 (`Speech`: player + TTS), 说一句, 观测一个窗口, 然后退出。
不做常驻 idle。

回答两个问题:

1. **共存** — 麦克风与扬声器同进程起会不会打架 (device 争用 / stream 冲突 / 生命周期).
   `[boot] both up` 出现即两者共存成立.
2. **回声** — 嘴说一句, 看耳朵有没有听见自己说的话. 听见了就是回声, 事件打印里的 Δ
   就是回声延迟 (相对说话起点). 每个事件带 `seg=` 后四位, 区分"切段"和"只认一半".

耳朵全程 `always` 礼仪 (纯 segment_vad), 不让 LLM 进观测链路.

## 打印

```
[boot] listener: assembling (capture + asr, device=<default>) ...
[boot] speech: starting (player + tts) ...
[boot] speech: started — player=MiniAudioStreamPlayer 44100Hz/1ch
[boot] capture: miniaudio capture, 44100Hz, 1ch, pcm_s16le
[boot] both up                        ← 两个器官都活着
[boot] listener: listening            ← 麦克风真正被占用
[say] "你好，这是一句测试。..."         ← Δ 轴的原点
[first   + 2.63s seg=abcd] 你好        ← 耳朵听见了 → 回声
[clause  + 8.55s seg=abcd] 听一听。
[tail    +10.18s seg=abcd] 听一听。
[say done] played 5.02s over 14 samples (播放窗口 = Δ 0.00s → +6.29s)
[observe] 再听 8s 收尾, 之后自动退出
[done] observation complete — exiting
```

判决: `[say]` 之后耳侧出现 `[first/clause/tail]` 且文本与所说内容对得上 → 有回声;
说完之后耳侧安静 → 这轮没测到回声.

## 测试方法

**外放跑** — 要测的就是扬声器 → 麦克风这条路. 戴耳机测到的是"没回声".

    moss nodes install .moss/system_test_nodes/voice_echo_probe
    moss --mode system_test nodes run .moss/system_test_nodes/voice_echo_probe/ -- \
        <device_pattern> "要说的一句话"

两个参数都可省. 节点跑完自动退出, 不占麦克风锁.
