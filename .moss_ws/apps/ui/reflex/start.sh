#!/bin/bash
# 借鉴 minecraft_bot 的 finally 清理模式：SIGTERM/SIGINT 时显式杀掉整个 reflex
# 进程树（包括 multiprocessing 子进程），确保不留孤儿进程。

kill_descendants() {
    local ppid=$1 sig=${2:-TERM}
    for cpid in $(pgrep -P "$ppid" 2>/dev/null); do
        kill_descendants "$cpid" "$sig"
    done
    kill -"$sig" "$ppid" 2>/dev/null || true
}

cleanup() {
    echo "[reflex] Shutting down all child processes..."
    if [ -n "$REFLEX_PID" ]; then
        kill_descendants "$REFLEX_PID" "TERM"
        sleep 2
        kill_descendants "$REFLEX_PID" "KILL"
    fi
    exit 0
}
trap cleanup TERM INT

uv run reflex run &
REFLEX_PID=$!
wait $REFLEX_PID
