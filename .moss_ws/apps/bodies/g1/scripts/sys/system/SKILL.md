---
name: g1-system
description: G1 PC2 系统环境信息。确认 Jetson 型号、系统资源、Python 工具链和 MOSS 运行状态。
---

# G1 系统技能

## 调查命题

1. Jetson 型号和 L4T 版本是什么？
2. CPU/内存/磁盘/温度概况？
3. Python 版本、uv 版本、虚拟环境是否正确？
4. MOSS 是否可正常启动？all-commands 是否正常？

## 架构决策

- Jetson 型号确认 → 验证硬件环境与文档记录一致
- 资源概况 → 为性能基线提供快速参照
- Python/moss 环境 → 确认装机完成，开发环境就绪
- moss 启动验证 → 这是阶段 D（MOSS 装机）的验收检查点

## 脚本

| 脚本 | 输入 | 输出 | 耗时 |
|------|------|------|------|
| `01_jetson_info.sh` | 无 | Jetson 型号、L4T 版本、内核版本 | <1s |
| `02_resources.sh` | 无 | CPU/内存/磁盘/温度 摘要 | <5s |
| `03_python_env.sh` | 无 | Python 版本、uv 版本、VIRTUAL_ENV、pip 关键包 | <3s |
| `04_moss_check.sh` | 无 | `moss --ai start` + `moss --ai all-commands` 输出 | ~10s |
