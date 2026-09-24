---
created: 2026-09-24
depends: []
description: Mac 上 say 最后一个字被吞 — miniaudio 在自己的 wait_play_done 里补设备排空,
  core/contracts 不动.
milestone: null
priority: P1
status: completed
status_note: fix lives entirely in MiniAudioStreamPlayer; core/contracts untouched
title: Speech Tail Drain
updated: '2026-09-24'
---

# Speech Tail Drain

> Use `moss features set-status speech-tail-drain <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

Mac 上 `say` 的最后一个字必被吞. 根因不是 clear 本身, 而是"播完"的判据:

- `BaseAudioStreamPlayer` 的 `_play_done_event` 在 worker 输入队列一空就置位, `wait_play_done`
  因此提前返回.
- `MiniAudioStreamPlayer._audio_stream_write` 只把 PCM 放进 `_data_queue`, 设备回调还要一个
  周期才交给 CoreAudio; 加上设备自己的输出缓冲, "队列空"比"真播完"至少早一块.
- 核心路径随后 (`_play_loop` finally、`close()`) 调 `player.clear()`, 而 clear 第一件事是
  `_playback.stop()` —— 还在设备缓冲里的尾音被切.

## Design Index

- 代码: `src/ghoshell_moss/host/speech/player/miniaudio_player.py` (唯一改动)
- 测试: `tests/ghoshell_moss/host/speech/test_miniaudio_drain.py`

## Key Decisions

- **修在 miniaudio 实现里, 不动 contracts/core.** 设备输出缓冲是 miniaudio/CoreAudio 的实现
  细节, 不该让 `BaseAudioStreamPlayer` 为它长出新接口. 修完 core 仍在自然播完时调 `clear()`
  —— 但那时设备已排空, `stop()` 切掉的只是补的静音, 于是不需要改 core 的 clear.
- **用现成的契约方法 `wait_play_done` 当接缝, 不新加内部钩子.** 一度试过在
  `BaseAudioStreamPlayer` 加 `_wait_device_drained` 钩子 + `_playback_finished` 标志来跳过
  clear; 已否决并回退: 那是把设备关注点塞进共享基类. `wait_play_done` 本来就是"等播完"的契约
  方法, 哪个后端违反承诺就由哪个后端覆写 —— 这才是抽象方法存在的意义.
- **不把基类 `_wait_consumed` 改成设备真播完.** 它必须留作节拍时钟: worker 写完 chunk k 后
  sleep 它的时长, chunk k+1 才能在 k 播完前交给设备 (read-ahead). 若改成等设备播完 k 再写
  k+1, 每个块边界都会掉一个设备周期的静音.
- **miniaudio 用回调推进的播放头当设备时钟, 不引设备 API.** `PlaybackDevice` 的 cdef 只暴露
  `sampleRate/state` 等少数字段, 没有 played-frames. 但 generator 每次回调都知道: 回调被触发
  的时刻 = 上一块刚播完, 故播放头 == 已交付帧数. 补的静音从不进 `_buf`, 故
  `_last_real_frame_pos` 精确标出真实音频终点. 拒绝"用 `buffersize_msec` 估算缓冲深度": 那个
  值只是 hint, 实测 CoreAudio 回调周期与它不完全一致.
- **顺带修 `is_playing()`.** 基类只看内部事件会提前报停; 覆写为"设备缓冲里还有声音就算在播".

## Implementation Notes

- `wait_play_done(timeout)` 覆写: 先 `super()` (预估 sleep + 等基类事件), 再
  `_drain_device_output`; 返回即设备已排空, timeout 到期仍未排空返回 False.
- 排空判据只用设备回调给出的播放头 (不乐观外推提前返回); 回调是离散的, 所以用回调时刻外推
  下一次复核时间点 (5–50ms), 避免空转轮询.
- 设备停摆兜底: 播放头超过 `_DRAIN_STALL_TIMEOUT` (0.5s) 不推进就放弃, 不让 `wait_play_done`
  挂死 (基类 timeout 只约束预估 sleep, 不约束等事件那一段).
- 残留边界: 基类 `_play_done_event` 本身 (以及 `on_play_done` 回调) 仍是"队列空"语义. 在不改
  core 的约束下这是明知的取舍 —— 真正被 `say`/`wait_played` 读到的是 `wait_play_done` 的结果,
  它已是设备排空语义.
- 实测 (Mac, 16k mono): 0.35s 片段旧路径 0.4505s 返回, 新路径 0.5662s —— 差的正是原本留在
  CoreAudio 缓冲里、随后被 stop() 丢掉的那段.
