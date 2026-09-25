---
name: 'topic_subscriber'
description: 'generic topic drainer — subscribes to a topic by name and prints each raw Topic as one JSON line'
singleton: true
exec:
  command: python
  args: main.py
---

Generic topic drainer probe. Subscribes to `matrix.session.topics` on an arbitrary
topic name (跨进程经 zenoh), 阻塞 poll 每条原始 `Topic` 并打印一行 `name #n: {json}`.
No channel, no topic model — works standalone, no Ghost needed.

把 topic 名作为 argv 传入 (缺省 `clause`):

    moss nodes run .moss/system_test_nodes/topic_subscriber/ -- audio/sample

它验证任意 topic 的生产装线 (如 `moss_runtime._audio_sample_topic_bridge` 把
player 播放样本广播成 `AudioSampleTopic`).

## 测试方法 (recorded)

1. 起本 node (后台或前台均可), 指定要听的 topic:

       moss nodes run .moss/system_test_nodes/topic_subscriber/ -- audio/sample

2. 另开进程产生该 topic (如 `moss audio listen` 听侧 / `moss-shell` 说侧).

3. 数本 node 打印的 `audio/sample #n:` 行数, 每条带 `meta.type` / `data.role` /
   `data.rms_db` / `data.spectrum_bins` / `data.waveform`.

Run in the same network scope as the producer (default scope from .moss).
