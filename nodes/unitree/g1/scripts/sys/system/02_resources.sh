#!/usr/bin/env bash
# 调查: PC2 系统资源摘要 — CPU/内存/磁盘/温度
# 决策: 快速判断 PC2 是否有足够余量运行 MOSS + DDS
set -euo pipefail

echo "=== CPU ==="
nproc 2>/dev/null && echo " cores"
lscpu 2>/dev/null | grep -E 'Model name|CPU\(s\)|Thread|Core|Socket|MHz|BogoMIPS' || echo "(lscpu 不可用)"

echo
echo "=== 内存 ==="
free -h 2>/dev/null || echo "(free 不可用)"

echo
echo "=== 磁盘 ==="
df -h / /home 2>/dev/null | grep -v '^Filesystem' || echo "(df 不可用)"

echo
echo "=== 温度 ==="
if [ -d /sys/class/thermal ]; then
    for zone in /sys/class/thermal/thermal_zone*; do
        type=$(cat "$zone/type" 2>/dev/null || echo "?")
        temp=$(cat "$zone/temp" 2>/dev/null || echo "?")
        # temp 通常是毫摄氏度
        if [ "$temp" != "?" ] && [ "$temp" -gt 1000 ] 2>/dev/null; then
            temp="$((temp / 1000))°C"
        fi
        echo "  $type: $temp"
    done
else
    echo "(无 thermal zone)"
fi

echo
echo "=== 运行时间 ==="
uptime
