#!/usr/bin/env bash
# 调查: PC2 所有网络接口、路由表、DNS
# 决策: 确认哪个接口连接 G1 内部交换机 (192.168.123.x)，决定 DDS 网卡绑定
set -euo pipefail

echo "=== 网络接口 ==="
ip addr show

echo
echo "=== 路由表 ==="
ip route show

echo
echo "=== DNS 配置 ==="
cat /etc/resolv.conf 2>/dev/null || echo "(无 /etc/resolv.conf)"

echo
echo "=== 接口摘要 ==="
ip -brief addr show
