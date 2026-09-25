#!/usr/bin/env bash
# 调查: PC2 暴露的 USB 接口 — 总线拓扑、已连接设备
# 决策: 可用 USB 端口数量与类型，决定可同时外接多少设备（摄像头+音频）
set -euo pipefail

echo "=== USB 设备树 ==="
lsusb -t 2>/dev/null || echo "(lsusb 不可用)"

echo
echo "=== USB 设备列表 ==="
lsusb 2>/dev/null || echo "(lsusb 不可用)"

echo
echo "=== USB 控制器详情 ==="
for dev in /sys/bus/usb/devices/usb*; do
    if [ -f "$dev/speed" ]; then
        speed=$(cat "$dev/speed" 2>/dev/null || echo "?")
        product=$(cat "$dev/product" 2>/dev/null || echo "?")
        echo "  $dev  speed=$speed  product=$product"
    fi
done

echo
echo "=== 可用 USB 端口 ==="
lsusb -v 2>/dev/null | grep -E 'Bus|bNumConfigurations|idProduct|idVendor' | head -40 || echo "(无法枚举)"
