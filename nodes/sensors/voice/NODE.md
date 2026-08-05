---
name: 'voice'
description: '语音输入感知节点 — 将麦克风音频经状态机 + ASR 转为 SpeechTopic 广播和 Mindflow 信号'
category: sensors
singleton: true
exec:
  command: .venv/bin/python
  args: main.py
---

Voice input node — thin shell for the host/voice core (VoiceController).

## 架构

voice node 是薄壳，主体在 `ghoshell_moss.host.voice`：

- **core (host/)** : VoiceController contract + 实现（两轴状态机 + miniaudio 直采 + VolcengineASR）+ VoiceConfig（10 开关正交配置）
- **channel** : voice/mode/config 子 channel 树，模型通过 CTML `<voice:start />` 等治理
- **adapter** : 本 node 内的事件适配器 —— core 事件 → matrix Session (SpeechTopic + AudioSignal)

core 不依赖 matrix/channel/transport。

## 配置

voice.json 在 node home 目录，首次运行自动生成默认值。
Volcengine ASR 密钥通过环境变量 `VOLCENGINE_BM_ASR_APPID` / `VOLCENGINE_BM_ASR_TOKEN` 注入。

## CTML invocation

    <voice:start />
    <voice:stop />
    <voice:status />
    <voice.mode:set name="turn_taking" />
    <voice.mode:current />
    <voice.config:show />
    <voice.config:set key="barge_in" value="false" />

## 开发

Launch: `moss nodes run nodes/sensors/voice`  (pure process, default headless)
Debug:  `.venv/bin/python main.py --help`
