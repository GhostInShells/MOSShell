#!/usr/bin/env bash
# 调查: v4l2 视频捕捉设备 — 是否存在、支持的分辨率/帧率/格式
# 决策: USB 摄像头能否在 PC2 上直接使用？决定 G1 视觉方案
set -euo pipefail

echo "=== v4l2 设备列表 ==="
if command -v v4l2-ctl &>/dev/null; then
    v4l2-ctl --list-devices 2>&1 || echo "(无 v4l2 设备)"
else
    echo "(v4l2-ctl 不可用 — 请 apt install v4l-utils)"
fi

echo
echo "=== /dev/video 设备 ==="
ls -la /dev/video* 2>/dev/null || echo "(无 /dev/video 设备)"

echo
echo "=== 每个设备的支持格式 ==="
for dev in /dev/video*; do
    [ -e "$dev" ] || continue
    echo "--- $dev ---"
    if command -v v4l2-ctl &>/dev/null; then
        v4l2-ctl -d "$dev" --list-formats-ext 2>&1 || echo "(无法查询 $dev)"
    fi
    echo
done

echo
echo "=== 内核视频模块 ==="
lsmod 2>/dev/null | grep -iE 'uvcvideo|videodev|v4l2' || echo "(未找到视频内核模块)"

echo
echo "=== 结论 ==="
echo "请人类判断:"
echo "  - 是否有 /dev/video 设备？插入 USB 摄像头后是否出现新设备？"
echo "  - 支持的分辨率和帧率是否满足视觉需求？"
echo "  - 如果无设备且无 USB 摄像头可插入，视觉方案需要走网络摄像头或外部方案"
