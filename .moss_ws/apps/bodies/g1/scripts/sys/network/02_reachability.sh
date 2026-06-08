#!/usr/bin/env bash
# 调查: 三节点连通性 — PC1(运控), LiDAR, 外网
# 决策: 确认 DDS 通讯路径完整，判断是否需要网络修复
set -euo pipefail

ping_test() {
    local label="$1"
    local target="$2"
    printf "%-20s (%s): " "$label" "$target"
    if ping -c 2 -W 2 "$target" >/dev/null 2>&1; then
        local rtt
        rtt=$(ping -c 2 "$target" 2>/dev/null | tail -1 | awk -F'/' '{print $5}' || echo "?")
        echo "OK  (rtt=${rtt}ms)"
    else
        echo "UNREACHABLE"
    fi
}

echo "=== G1 内部网络连通性 ==="
ping_test "PC1 (运控)"     192.168.123.161
ping_test "LiDAR"          192.168.123.120
ping_test "PC2 (本机)"     192.168.123.164

echo
echo "=== 外网连通性 ==="
ping_test "互联网"          8.8.8.8
