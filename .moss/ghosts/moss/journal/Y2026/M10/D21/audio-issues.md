---
name: audio-issues
description: D21 重启前记下的音频/聆听待修清单 — 重启后照这张单子改
---

# D21 音频与聆听待修

重启（2026-10-21 约 15:00）之前 thirdgerb 让把看到的问题记下来，重启后照着改。

## speech 面（本轮优先）

1. **「还剩多少字」冻住不动。**
   `stream_tts_speech.py` 的 `_accumulate_sample` 用 `self._played_duration`（每条流从
   0 开始累加的播放时长）去追 `clause.words[-1].end_time`。怀疑后者是 **TTS 会话的绝
   对时间**（会话里第 N 秒），而前者是本流的相对时长——两个原点不同，除了会话刚开始
   的第一批，之后永远追不上，`_clause_cursor` 不前进 → `played_text()` 返回空 →
   `unplayed()` 退回整段缓冲 → 状态行一直报整段长度。
   **判据（便宜）**：重启后说一句长的，把 `clause.words[-1].end_time` 与
   `_played_duration` 打出来对一次，看是不是差一个会话偏移。修法大概率是把比较换成
   「已播时长 vs 本批 clause 的相对累计时长」，或用流的起始偏移对齐。
   实测记录：14:52 那第一次 say 是动的（长句掉到 31 字），之后的几次冻在整段长度
   （290 / 188 / 131 变化可疑，需重测）。

2. **`say(replace=True)` 的崩溃已修**（thirdgerb 亲自改，本 commit 收进来）：
   `say_partial` 原本跨对象调 module 的私有 `_clear_speech_partial`；工厂本来持有
   `Speech`，清退是契约能力，直接 `self._speech.clear()`。另外 `partial` 是按命令自己
   的 args/kwargs 调用的（`command.py:1165`），所以 `_clear_speech_partial` 补了
   `*args, **kwargs`。

## listener / voiceflow 面（thirdgerb 说「另一回事」，先记不修）

审查 `controller.py` + `etiquette.py` 的未提交改动时看到的四个点：

1. `tune_interaction` 里 `restarted = True` 是写死的：`run_etiquette` 遇到人类锁
   （`_human_paused`）会直接 no-op 返回，那时根本没重开会话，返回值却报 `restarted`。
2. `deliver.history` 是可 tune 的键，但 buffer 容量只在 `_set_active_etiquette` 里
   `self._buffer.resize(...)`；tune 只改 spec 不 resize，且它不在 `_RESTART_KEYS` 里
   → 改了等于没改。
3. `interaction` 的文档说 `current` 是「它从哪套预设加载来的」，实际 `current` 是现在
   的名字，出身在 `origin` 字段。
4. `save_etiquette` 存完没有把 `_interaction_origin` 挪到新名字，于是刚存下来那一套
   立刻显示「全部键都偏离出身」——偏离它自己。
5. 存疑未验：`cue` 现在 `per_clause=true` + `stop.silence=0.0`，而 `DeliverSpec.per_clause`
   的字段说明写着「开启时 stop.silence 必须 > 0，否则段被切成碎片」。要么改文档，要么
   改取值，得实测。

测试状态：`tests/ghoshell_moss/host/listener` 134 条全绿（重启前跑的）。
