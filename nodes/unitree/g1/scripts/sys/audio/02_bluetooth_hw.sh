#!/usr/bin/env bash
# 调查: 蓝牙适配器硬件存在性 + rfkill 软锁状态
# 决策: Jetson Orin NX 是否有蓝牙？是否被软锁？决定蓝牙音频方案的可行性
set -euo pipefail

echo "=== 蓝牙适配器 ==="
if command -v hciconfig &>/dev/null; then
    hciconfig -a 2>&1 || echo "(无蓝牙适配器)"
else
    echo "(hciconfig 不可用)"
fi

echo
echo "=== rfkill 状态 ==="
if command -v rfkill &>/dev/null; then
    rfkill list | grep -i bluetooth || echo "(rfkill 中无 bluetooth 条目)"
else
    echo "(rfkill 不可用)"
fi

echo
echo "=== bluetoothd 服务 ==="
systemctl status bluetooth 2>/dev/null || echo "(bluetooth 服务未安装或 systemd 不可用)"

echo
echo "=== USB 蓝牙设备 ==="
lsusb 2>/dev/null | grep -i bluetooth || echo "(lsusb 中未发现蓝牙设备)"

echo
echo "=== 内核蓝牙模块 ==="
lsmod 2>/dev/null | grep -iE 'bluetooth|btusb' || echo "(未找到蓝牙内核模块)"
