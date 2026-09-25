#!/usr/bin/env bash
# 调查: 磁盘 IO — 吞吐量、iowait、存储使用
# 决策: MOSS 的日志写入和存储操作是否产生显著 IO 压力
set -euo pipefail

echo "=== 磁盘使用 ==="
df -h 2>/dev/null || echo "(df 不可用)"

echo
echo "=== IO 统计 (1s 采样) ==="
if command -v iostat &>/dev/null; then
    iostat -x 1 2 2>/dev/null | tail -n +4 || echo "(iostat 执行失败)"
else
    echo "(iostat 不可用 — 请 apt install sysstat)"
fi

echo
echo "=== IO 等待 (iowait 历史) ==="
if command -v sar &>/dev/null; then
    sar -u 1 3 2>/dev/null | grep -i iowait || echo "(无 iowait 数据)"
else
    echo "(sar 不可用)"
fi

echo
echo "=== 存储设备 ==="
lsblk 2>/dev/null || echo "(lsblk 不可用)"

echo
echo "=== MOSS 日志目录大小 ==="
for d in .moss_ws/logs .moss_ws/runtime 2>/dev/null; do
    if [ -d "$d" ]; then
        du -sh "$d" 2>/dev/null || true
    fi
done
