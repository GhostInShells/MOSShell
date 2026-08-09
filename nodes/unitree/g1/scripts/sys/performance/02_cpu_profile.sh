#!/usr/bin/env bash
# 调查: CPU 使用率 — 整体负载、per-core 分布、load average
# 决策: MOSS 是否对单核造成压力？多核负载是否均匀？
set -euo pipefail

echo "=== Load Average ==="
uptime

echo
echo "=== CPU 使用率 (top 快照) ==="
# 取两次 top，用第二次（更准确）; batch 模式避免交互
top -bn2 2>/dev/null | grep -A999 '^%Cpu' | head -20

echo
echo "=== Per-Core 使用率 ==="
if command -v mpstat &>/dev/null; then
    mpstat -P ALL 1 1 2>/dev/null || echo "(mpstat 执行失败)"
else
    echo "(mpstat 不可用 — 请 apt install sysstat)"
    echo "替代: /proc/stat 解析"
    grep '^cpu[0-9]' /proc/stat 2>/dev/null | head -10 || echo "(无法读取 /proc/stat)"
fi

echo
echo "=== CPU 频率 ==="
if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
        freq=$(cat "$cpu" 2>/dev/null || echo "?")
        echo "  $(dirname "$cpu" | xargs basename): $freq kHz"
    done
else
    echo "(cpufreq 不可用)"
fi
