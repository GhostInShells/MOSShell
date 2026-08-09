---
name: g1-dds
description: G1 DDS 通讯环境检查。确认 cyclonedds 安装状态、环境变量配置和共享内存支持。
---

# G1 DDS 技能

## 调查命题

1. cyclonedds 包是否已安装？版本是什么？
2. DDS 相关环境变量是否正确配置（`CYCLONEDDS_URI`、网卡选择）？
3. 共享内存传输是否可用？

## 架构决策

- cyclonedds 安装状态 → DDS 通讯的前置条件。未安装则后续所有 DDS 实验无法进行
- 环境变量 → 不正确的配置会导致 DDS 发现失败或跨网卡通讯中断
- 共享内存 → G1 外部开发时需 `enableSharedMemory=false`。确认当前配置

## 脚本

| 脚本 | 输入 | 输出 | 耗时 |
|------|------|------|------|
| `01_env_check.sh` | 无 | cyclonedds 包信息、DDS 环境变量、共享内存配置 | <2s |
