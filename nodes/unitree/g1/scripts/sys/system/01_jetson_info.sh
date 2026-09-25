#!/usr/bin/env bash
# 调查: Jetson 型号、L4T 版本、内核版本
# 决策: 确认硬件环境与文档记录一致
set -euo pipefail

echo "=== Jetson 型号 ==="
if [ -f /etc/nv_tegra_release ]; then
    cat /etc/nv_tegra_release
else
    echo "(非 Jetson 平台或文件缺失)"
fi

echo
echo "=== L4T 版本 ==="
if command -v jetson_release &>/dev/null; then
    jetson_release -v 2>&1
elif [ -f /etc/nv_tegra_release ]; then
    head -1 /etc/nv_tegra_release | grep -oP 'R\d+\.\d+(\.\d+)?' || echo "(无法解析)"
else
    echo "(无法获取)"
fi

echo
echo "=== 内核版本 ==="
uname -a

echo
echo "=== 硬件型号 ==="
cat /proc/device-tree/model 2>/dev/null | tr '\0' '\n' || echo "(无法读取)"

echo
echo "=== 模块序列号 ==="
if [ -f /sys/module/tegra_fuse/parameters/tegra_chip_id ]; then
    echo "Chip ID: $(cat /sys/module/tegra_fuse/parameters/tegra_chip_id 2>/dev/null || echo '?')"
fi
