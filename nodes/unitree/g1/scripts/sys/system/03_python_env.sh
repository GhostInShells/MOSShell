#!/usr/bin/env bash
# 调查: Python 工具链 — 版本、uv、虚拟环境、MOSS 关键依赖
# 决策: 开发环境就绪确认。阶段 D (MOSS 装机) 的检查点之一
set -euo pipefail

echo "=== Python ==="
python3 --version 2>&1 || echo "(Python3 不可用)"
which python3 2>/dev/null || echo "(未找到 python3)"

echo
echo "=== uv ==="
uv --version 2>&1 || echo "(uv 不可用)"
which uv 2>/dev/null || echo "(未找到 uv)"

echo
echo "=== 虚拟环境 ==="
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-(未设置)}"
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "venv Python: $("$VIRTUAL_ENV/bin/python3" --version 2>/dev/null || echo '?')"
fi

echo
echo "=== MOSS ==="
if command -v moss &>/dev/null; then
    moss --version 2>&1 || echo "(moss --version 失败)"
else
    echo "(moss 命令不可用 — MOSS 可能未安装或 venv 未激活)"
fi

echo
echo "=== 关键 Python 包 ==="
pip list 2>/dev/null | grep -iE 'ghoshell|cyclonedds|unitree|zmq|zenoh' || echo "(未找到关键包)"
