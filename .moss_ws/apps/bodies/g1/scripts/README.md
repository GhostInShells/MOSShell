# G1 验证脚本

安全原子化验证脚本。每个脚本验证 G1 的一个原子能力，独立运行，不依赖 MOSS channel 体系。

## 使用方式

脚本直接在 G1 PC2 上执行：

```bash
cd .moss_ws/apps/bodies/g1
uv run python scripts/01_<topic>.py
```

## 验证闭环

1. AI 编写脚本（在 macOS 上）
2. 通过 git 同步到 PC2
3. 人类在 PC2 上执行
4. 人类观察并反馈结果
5. AI 根据反馈调整理解和下一步脚本

## 脚本命名

`<序号>_<主题>.py`，序号两位数字。

## 验证点（待填充）

验证点来源于 SDK examples + 文档分析（阶段 A/B 完成后填充）。

---

*初始化: DeepSeek V4 Pro, 2026-06-07*
