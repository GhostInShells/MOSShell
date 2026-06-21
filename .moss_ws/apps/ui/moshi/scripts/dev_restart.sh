#!/bin/bash
# 杀掉所有 reflex 旧进程
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> 查找并杀掉旧进程..."

KILL_PIDS=()

# reflex 相关
for pid in $(pgrep -f "reflex run" 2>/dev/null || true); do
    KILL_PIDS+=("$pid")
    echo "  找到 reflex run (PID $pid)"
done
for pid in $(pgrep -f "react-router dev" 2>/dev/null || true); do
    KILL_PIDS+=("$pid")
    echo "  找到 react-router dev (PID $pid)"
done

# moshi main.py 相关
for pid in $(pgrep -f "moshi.*main.py" 2>/dev/null || true); do
    KILL_PIDS+=("$pid")
    echo "  找到 moshi main.py (PID $pid)"
done

# 占用 moshi 端口 9731 的进程
PORT_PID=$(lsof -ti:9731 2>/dev/null || true)
if [ -n "$PORT_PID" ]; then
    KILL_PIDS+=($PORT_PID)
    echo "  找到占用 9731 端口的进程 (PID $PORT_PID)"
fi

if [ ${#KILL_PIDS[@]} -gt 0 ]; then
    UNIQUE_PIDS=($(printf '%s\n' "${KILL_PIDS[@]}" | sort -n | uniq))
    echo "==> 杀掉 ${#UNIQUE_PIDS[@]} 个进程..."
    for pid in "${UNIQUE_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "${UNIQUE_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    echo "  已全部杀掉。"
else
    echo "  没有找到需要杀掉的进程。"
fi

echo "==> 清理完毕。"
