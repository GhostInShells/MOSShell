#!/usr/bin/env bash
# 调查: cyclonedds 包安装状态、环境变量、共享内存配置
# 决策: DDS 通讯就绪状态。未就绪则后续所有 DDS 实验无法进行
set -euo pipefail

echo "=== cyclonedds 包 ==="
pip list 2>/dev/null | grep -i cyclonedds || echo "(未安装 cyclonedds)"
python3 -c "import cyclonedds; print('cyclonedds version:', cyclonedds.__version__)" 2>&1 || echo "(无法 import cyclonedds)"

echo
echo "=== DDS 环境变量 ==="
echo "CYCLONEDDS_URI=${CYCLONEDDS_URI:-(未设置)}"
echo "NDDSHOME=${NDDSHOME:-(未设置)}"

# 检查默认配置文件
default_cfg="/etc/cyclonedds/config.xml"
if [ -f "$default_cfg" ]; then
    echo "--- $default_cfg ---"
    cat "$default_cfg"
else
    echo "(无 $default_cfg)"
fi

echo
echo "=== 共享内存 ==="
echo "当前配置:"
# G1 外部开发时 enableSharedMemory 应为 false
if [ -n "${CYCLONEDDS_URI:-}" ]; then
    echo "$CYCLONEDDS_URI" | grep -i 'shared_memory' || echo "(URI 中无共享内存配置)"
else
    echo "(未设置 CYCLONEDDS_URI，使用默认配置)"
fi

echo
echo "=== 网卡配置 (DDS 需指定网卡) ==="
ip -brief addr show 2>/dev/null | grep -v '^lo' || echo "(无法获取网卡列表)"
