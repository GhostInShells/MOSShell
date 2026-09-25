#!/usr/bin/env bash
# 调查: MOSS 运行时的进程树结构
# 决策: 了解 Matrix/Cell/Channel 各层占用的进程数量和父子关系
set -euo pipefail

echo "=== 进程树 (MOSS 相关) ==="
if command -v pstree &>/dev/null; then
    pstree -p 2>/dev/null | grep -iE 'moss|python|ghost' || echo "(未找到 MOSS 相关进程)"
else
    echo "(pstree 不可用)"
fi

echo
echo "=== MOSS 相关进程 (ps) ==="
ps aux 2>/dev/null | grep -iE 'moss|ghoshell' | grep -v grep || echo "(未找到 MOSS 进程)"

echo
echo "=== 进程计数 ==="
moss_count=$(ps aux 2>/dev/null | grep -iE 'moss|ghoshell' | grep -v grep | wc -l | tr -d ' ')
echo "MOSS 相关进程数: $moss_count"

echo
echo "=== 完整进程列表 (Python) ==="
ps aux 2>/dev/null | grep python | grep -v grep || echo "(无 Python 进程)"
