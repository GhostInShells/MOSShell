#!/usr/bin/env bash
# 调查: ALSA 播放/录音设备 + PulseAudio sink/source
# 决策: PC2 是否有内置声卡？能否独立播放和录音？
set -euo pipefail

echo "=== ALSA 播放设备 ==="
aplay -l 2>&1 || echo "(aplay 不可用或无设备)"

echo
echo "=== ALSA 录音设备 ==="
arecord -l 2>&1 || echo "(arecord 不可用或无设备)"

echo
echo "=== PulseAudio 状态 ==="
if command -v pactl &>/dev/null; then
    echo "--- Cards ---"
    pactl list cards short 2>/dev/null || echo "(无 card)"
    echo "--- Sinks (播放) ---"
    pactl list sinks short 2>/dev/null || echo "(无 sink)"
    echo "--- Sources (录音) ---"
    pactl list sources short 2>/dev/null || echo "(无 source)"
else
    echo "(pactl 不可用)"
fi

echo
echo "=== /dev/snd 设备 ==="
ls -la /dev/snd/ 2>/dev/null || echo "(无 /dev/snd)"

echo
echo "=== 内核音频模块 ==="
lsmod 2>/dev/null | grep -iE 'snd|audio' || echo "(未找到音频内核模块)"
