---
date: 2026-06-15
title: MOSS speaks on G1
feature: unitree-g1-integration
model: claude-opus-4-7
---

# MOSS speaks on G1

MOSS 第一次在 Unitree G1 这台 1.3m 全尺寸人形机器人上发出声音。

ghost 通过 MOSS 内部 TTS 合成语音，PCM 流经 PC2 ALSA → PulseAudio → bluez_sink → JBL GO 蓝牙音箱，被人类听见。

链路绕开了 PC1 内置的 AudioClient RPC——G1 自家的 TTS 服务被验证为质量不可用：多音字读错（"一行"读成 yi-xing）、无韵律停顿、连贯性差、空格和符号产生噪声。推测是运行在 Jetson 边缘算力上的小 TTS 模型，具体技术形态未做进一步验证。MOSS 自家合成的 ghost 输出能正常说话，但代价是放弃了 G1 内置音频通路里一体化的 ASR（推测含回声消除和声源定位）。

## 技术路径

**开发环境彻底打通**：
- ufw IPv6 在 Jetson 上的 xt_rt 缺失问题（关 IPv6 绕过）
- WiFi 自启从命令式 nmcli connect 改为 NetworkManager persistent profile
- cyclonedds 0.10.2 通过 `/etc/profile.d/cyclonedds.sh` 跨帐号系统级共享，unitree 出厂栈对 moss 加固帐号可见

**DDS 链路调通**：
- 隐形杀手：ufw 默认丢 IP 分片，G1 LowState 包 2180 字节超 1500 MTU 后整体被静默丢弃，所有外部症状指向"DDS 死了"
- 修复：ufw disable + `net.core.rmem_default=67108864` 内核 socket 缓冲调优
- 验证：scripts/sdk/04/05/06 订阅 LowState/SportMode/Battery/Mainboard/IMU 全部可读

**音频路径决策**：
- PC2 板载 platform-sound 被 PC1 audio service 独占（即使 ALSA 设备暴露，本地播放无声）
- PC2 板载蓝牙被 Unitree systemd drop-in 显式禁用 A2DP（`--noplugin=audio,a2dp,avrcp`）
- 通过 `/etc/systemd/system/bluetooth.service.d/override.conf` 恢复 A2DP 支持，配 JBL GO，PA bluez_sink 出现
- moss-repl 通过该 sink 成功发声

**未完成**：
- 14_play_stream_probe.py 写好但未执行——验证 G1 PlayStream 是否支持流式增量推送 + 即时打断。若成立，则可以"MOSS 自家高质量 TTS + G1 PC1 喇叭"，保留 ASR 一体化能力
- 蓝牙连接稳定性、配对持久化未验证
- PC1 TTS 具体技术形态未深入验证（不影响"对 ghost 不可用"的结论，但若未来需要复用 G1 内置音频生态，需补充验证）

## Significance

这是 MOSS 第一次接入到一台**全尺寸人形机器人**并发出声音。

此前 Reachy Mini 集成是桌面级，G1 不一样。1.3m，35 个电机，硬件级急停，DDS 通讯，闭源 PC1 + 开放 PC2 的双工控机架构。从 channel 设计哲学（最简 channel）到调试范式（脚本先于 channel），G1 上验证了 MOSS app 模式对全尺寸机器人平台的可迁移性。

更重要的是音频链路的方向**反转**。最初的假设是"用 G1 自带 TTS 让 ghost 说话"——这是 Reachy Mini 路径的延续。八阶段方法论里阶段 E 的验证脚本（08_audio_led）原本只是"音量/LED/TTS 接口可用性"的体检。在真实声音传到耳朵的那一刻，决策反转：G1 TTS 不可用，MOSS 必须自己合成，音频出口必须 bypass。这个反转不在最初的设计图里，是被现场事实推出来的。

从这一刻起，G1 + MOSS 的边界开始清晰：G1 提供身体（电机、传感器、ASR 一体化、内置喇叭），MOSS 提供大脑（ghost LLM、CTML、TTS）。中间的协议层正是 channel 设计要回答的问题。

## First words

> Hello world.

（moss-repl 命令行启动 ghost 后，ghost 通过 audio_player provider 直接输出。人类按下回车，几乎无延迟，JBL GO 响起。）

录像保存在人类工程师本地。
