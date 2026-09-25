#!/usr/bin/env bash
# 调查: 实际播放测试音 — 回答"PC2 自己能出声吗"
# 决策: 如果这里能发声，MOSS 音频可以直接走 PC2，不需要经过 PC1 AudioClient API
set -euo pipefail

echo "=== 音频输出测试 ==="

# 尝试 speaker-test (ALSA 原生)
if command -v speaker-test &>/dev/null; then
    echo "使用 speaker-test 播放 440Hz 测试音..."
    speaker-test -t sine -f 440 -l 1 -c 2 2>&1
    echo "speaker-test 完成"
else
    echo "(speaker-test 不可用)"
fi

echo
echo "=== 尝试播放系统音频文件 ==="

# 常见的测试音频文件路径
test_files=(
    "/usr/share/sounds/alsa/Front_Center.wav"
    "/usr/share/sounds/alsa/Front_Left.wav"
    "/usr/share/sounds/purple/login.wav"
)

for f in "${test_files[@]}"; do
    if [ -f "$f" ]; then
        echo "播放: $f"
        if command -v aplay &>/dev/null; then
            aplay "$f" 2>&1
            echo "aplay 完成"
        elif command -v paplay &>/dev/null; then
            paplay "$f" 2>&1
            echo "paplay 完成"
        fi
        exit 0
    fi
done

echo "(未找到系统测试音频文件)"

echo
echo "=== 结论 ==="
echo "请人类判断: PC2 在播放过程中是否发出了声音？"
echo "  [Y] 有声音 → PC2 可独立发声，MOSS 音频可走 PC2"
echo "  [N] 无声音 → 音频只能走 PC1 AudioClient API"
