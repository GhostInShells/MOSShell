---
name: g1-audio
description: G1 PC2 音频硬件能力摸底。回答 PC2 能否独立发声和录音，决定音频链路走 PC2 蓝牙还是 PC1 API。
---

# G1 音频技能

## 调查命题

1. PC2 有内置声卡吗？有播放和录音设备吗？
2. Jetson Orin NX 有蓝牙适配器吗？rfkill 是否被软锁？
3. 周边是否有可连接的蓝牙音频设备（耳机/音箱）？
4. PC2 能否实际发出声音？

## 架构决策

| 命题 | 如果成立 | 如果不成立 |
|------|---------|-----------|
| PC2 有声卡且可播放 | MOSS 音频可直接从 PC2 输出 | 必须走 PC1 AudioClient API |
| PC2 有蓝牙适配器 | 可连蓝牙耳机/音箱，绕过 PC1 音频链路的不确定性（PlayStream 状态/TTS 耗时/Cancel） | 音频唯一路径是 PC1 VuiClient RPC |
| PC2 有录音设备 | 可本地录音，做 VAD/ASR 预处理 | 语音输入只能走 G1 四麦阵列（PC1 UDP 组播） |

## 脚本

| 脚本 | 输入 | 输出 | 耗时 |
|------|------|------|------|
| `01_list_devices.sh` | 无 | ALSA 播放/录音设备列表、PulseAudio sink/source | <1s |
| `02_bluetooth_hw.sh` | 无 | 蓝牙适配器存在性、rfkill 状态、bluetoothd 运行状态 | <2s |
| `03_bluetooth_scan.sh` | 无 | 可发现的蓝牙音频设备列表 | ~15s |
| `04_speaker_test.sh` | 无 | 实际播放测试音，确认音频输出通路 | ~3s |
