#!/usr/bin/env bash
# 调查: MOSS 未启动时的系统基线 — CPU/内存/IO 参照点
# 决策: 后续所有性能数据的对比基准。没有基线，性能数据无判断意义
# 使用: 在启动 MOSS 之前执行此脚本
set -euo pipefail

echo "=== Idle Baseline ==="
echo "时间: $(date -Iseconds)"
echo "注意: 确保 MOSS 未在运行"

echo
echo "=== Load Average ==="
uptime

echo
echo "=== CPU 使用率 ==="
top -bn2 2>/dev/null | grep -A999 '^%Cpu' | head -10

echo
echo "=== 系统内存 ==="
free -h 2>/dev/null

echo
echo "=== 磁盘使用 ==="
df -h / 2>/dev/null

echo
echo "=== 进程数 ==="
ps aux 2>/dev/null | wc -l | tr -d ' '
echo "total processes"

echo
echo "=== Python 进程 (应为空或无 MOSS) ==="
ps aux 2>/dev/null | grep python | grep -v grep || echo "(无 Python 进程 — 正常)"

echo
echo "=== 基线记录完毕 ==="
echo "请保存以上输出。启动 MOSS 后，用 01-04 脚本对比。"
