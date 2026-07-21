#!/usr/bin/env bash
# 调查: MOSS 启动验证 — cognitive entry + command tree
# 决策: 阶段 D 验收检查点。确认 MOSS 在 PC2 上正常运行
set -euo pipefail

echo "=== MOSS Cognitive Entry ==="
moss --ai start 2>&1 || echo "错误: moss --ai start 失败"

echo
echo "=== MOSS All Commands ==="
moss --ai all-commands 2>&1 || echo "错误: moss --ai all-commands 失败"

echo
echo "=== 结论 ==="
echo "如果以上两个命令正常输出，MOSS 在 PC2 上的安装验证通过。"
echo "如果失败，检查: venv 激活、uv sync 完成、.moss_ws/.env 配置。"
