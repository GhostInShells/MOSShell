#!/bin/bash
# 杀掉所有 moss/reflex/text_to_image 旧进程，释放端口 3000
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> 查找并杀掉旧进程..."

KILL_PIDS=()

for pid in $(pgrep -f "reflex run" 2>/dev/null || true); do
    KILL_PIDS+=("$pid")
    echo "  找到 reflex run (PID $pid)"
done

for pid in $(pgrep -f "react-router dev" 2>/dev/null || true); do
    KILL_PIDS+=("$pid")
    echo "  找到 react-router dev (PID $pid)"
done

for pid in $(pgrep -f "text_to_image/main.py" 2>/dev/null || true); do
    KILL_PIDS+=("$pid")
    echo "  找到 text_to_image (PID $pid)"
done

for pid in $(pgrep -f "circus.circusd.*moss" 2>/dev/null || true); do
    KILL_PIDS+=("$pid")
    echo "  找到 moss circusd (PID $pid)"
done

for pid in $(pgrep -f "multiprocessing.spawn.*spawn_main" 2>/dev/null || true); do
    KILL_PIDS+=("$pid")
    echo "  找到 multiprocessing spawn (PID $pid)"
done

PORT_PID=$(lsof -ti :3000 2>/dev/null || true)
if [ -n "$PORT_PID" ]; then
    KILL_PIDS+=("$PORT_PID")
    echo "  找到端口 3000 占用 (PID $PORT_PID)"
fi

if [ ${#KILL_PIDS[@]} -gt 0 ]; then
    UNIQUE_PIDS=($(printf '%s\n' "${KILL_PIDS[@]}" | sort -n | uniq))
    echo "==> 杀掉 ${#UNIQUE_PIDS[@]} 个进程..."
    for pid in "${UNIQUE_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${UNIQUE_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    echo "  已全部杀掉。"
else
    echo "  没有找到需要杀掉的进程。"
fi

echo "==> 等待端口 3000 释放..."
for i in {1..10}; do
    if ! lsof -ti :3000 >/dev/null 2>&1; then
        echo "  端口 3000 已释放。"
        break
    fi
    sleep 1
done

echo "==> 等待 circusd 退出..."
for i in {1..10}; do
    if pgrep -f "circus.circusd.*moss" >/dev/null 2>&1; then
        sleep 1
    else
        echo "  circusd 已退出。"
        break
    fi
done

echo "==> 清理旧日志..."
rm -f \
    .moss_ws/apps/ui/reflex/runtime/logs/*.log \
    .moss_ws/apps/media/text_to_image/runtime/logs/*.log \
    .moss_ws/runtime/logs/moss.log \
    .moss_ws/runtime/logs/circusd.log
echo "  旧日志已清理。"

echo "==> 清理完毕。"
