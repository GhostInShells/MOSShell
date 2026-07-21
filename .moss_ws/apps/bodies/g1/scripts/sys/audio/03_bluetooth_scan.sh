#!/usr/bin/env bash
# 调查: 扫描可发现的蓝牙设备（音频类优先）
# 决策: 周边是否有可用的蓝牙耳机/音箱？决定音频方案走蓝牙还是 PC1 API
set -euo pipefail

echo "=== 蓝牙扫描 (15s) ==="

if ! command -v bluetoothctl &>/dev/null; then
    echo "错误: bluetoothctl 不可用。请安装 bluez"
    exit 1
fi

# 确保蓝牙适配器已上电
bluetoothctl power on 2>/dev/null || true

echo "正在扫描..."
timeout 15 bluetoothctl scan on 2>/dev/null &
scan_pid=$!
sleep 15
kill "$scan_pid" 2>/dev/null || true

echo
echo "=== 已发现设备 ==="
bluetoothctl devices 2>/dev/null || echo "(无设备)"

echo
echo "=== 音频类设备 (Headset/Audio) ==="
bluetoothctl devices 2>/dev/null | while read -r line; do
    mac=$(echo "$line" | awk '{print $2}')
    info=$(bluetoothctl info "$mac" 2>/dev/null || true)
    if echo "$info" | grep -qiE 'Audio|Headset|Speaker'; then
        echo "$line"
        echo "$info" | grep -E 'Name:|Icon:|Class:' || true
        echo "---"
    fi
done
