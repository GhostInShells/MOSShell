#!/usr/bin/env bash
# 调查: 内存使用 — 系统级 + MOSS 进程 RSS/VSS
# 决策: MOSS 的内存足迹是否在 Jetson 的合理范围内
set -euo pipefail

echo "=== 系统内存 ==="
free -h 2>/dev/null || echo "(free 不可用)"

echo
echo "=== MOSS 进程内存 (RSS 排序) ==="
ps aux 2>/dev/null | awk 'NR==1 {print $0} /moss|ghoshell|python.*moss/ && !/grep/ {print $0}' | sort -k6 -rn | head -20

echo
echo "=== 进程内存 Top 10 ==="
ps aux 2>/dev/null | awk 'NR==1 {print $0} {print $0}' | sort -k4 -rn | head -11

echo
echo "=== /proc/meminfo 关键项 ==="
grep -E '^(MemTotal|MemFree|MemAvailable|Cached|SwapTotal|SwapFree)' /proc/meminfo 2>/dev/null || echo "(无法读取)"
