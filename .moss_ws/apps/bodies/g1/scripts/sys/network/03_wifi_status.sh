#!/usr/bin/env bash
# 调查: WiFi 射频状态、当前连接、信号强度
# 决策: PC2 能否自主联网？每次开机是否需要手动用以太网路径开启 WiFi？
set -euo pipefail

echo "=== WiFi 射频状态 ==="
if command -v nmcli &>/dev/null; then
    nmcli radio wifi
else
    echo "(nmcli 不可用)"
fi

echo
echo "=== WiFi 设备 ==="
if command -v nmcli &>/dev/null; then
    nmcli device status | grep -i wifi || echo "(无 WiFi 设备)"
else
    echo "(nmcli 不可用)"
fi

echo
echo "=== 当前 WiFi 连接 ==="
if command -v nmcli &>/dev/null; then
    nmcli -t -f ACTIVE,SSID,BSSID,MODE,CHAN,FREQ,RATE,SIGNAL,BARS device wifi list 2>/dev/null | grep '^yes' || echo "(未连接任何 WiFi)"
else
    iwconfig 2>/dev/null | grep -E 'ESSID|Signal' || echo "(无无线网卡或 iwconfig 不可用)"
fi

echo
echo "=== rfkill 状态 ==="
if command -v rfkill &>/dev/null; then
    rfkill list
else
    echo "(rfkill 不可用)"
fi
