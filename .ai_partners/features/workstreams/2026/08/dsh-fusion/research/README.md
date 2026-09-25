# research — dsh 融合的调研域

本目录承载 dsh 融合的调研活动。核心模块在 `src/ghoshell_moss/`, 不在此处开发,
此目录只为「搞清楚 dsh 怎么用」服务。

## 结构

- `source/` — 只读 dsh 源码快捷路径 (clone 的 deepseek-harness, 仅观察, 不运行/不 import/不开发)
- `skills/` — 调研技能, **每个 skill 自包含环境**(自己的 plugin + 自己的 DSH_HOME)

早期本目录还有 `home/`(共享 DSH_HOME)与「按日期的调研记录 *.md」, 因结论被后续
实验推翻, 已于 2026-08-17 删除(见 git 历史)。当前以「每个 skill 自包含」为最佳实践。

## 运行方式

每个 skill 自包含, 进入 skill 目录直接跑它的脚本:

```sh
cd research/skills/<skill-name>
python3 serve.py   # 或 skill 自己的入口脚本
```

脚本保持零参数 — 运行时上下文 (cwd + env) 提供一切。skill 自己的 DSH_HOME 在
`<skill>/home/`, node_modules 由 dsh 启动时自动 heal 软链, 不需手动建、不需提交。
