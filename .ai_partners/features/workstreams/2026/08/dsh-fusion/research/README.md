# research — dsh 融合的调研域

本目录承载 dsh 融合的全部调研活动, 四块各司其职:

- `source/` — 只读源码快捷路径 (clone 的 deepseek-harness, 仅观察, 不运行/不 import/不开发)
- `skills/` — 调研技能脚本 (每个 skill = 一个任务类型, 一个 skill 可配几个脚本)
- `home/` — 隔离 DSH_HOME (运行时环境, 防止污染 ~/.dsh)
- 本层 `*.md` — 调研记录 (探索轨迹与结论)

核心模块在 `src/ghoshell_moss/`, 不在此处开发。此目录只为「搞清楚 dsh 怎么用」服务。

## 运行方式

调研脚本在 `skills/` 下. 运行前进入 `research/` 目录, 让 dsh 用本域的隔离 home:

```sh
cd research
export DSH_HOME=./home
python3 skills/<skill-name>/<script>.py
```

脚本保持零参数 — 运行时上下文 (cwd + env) 提供一切, 脚本不 resolve 路径.
session 默认落在 `./.sessions` (相对 cwd), 已被 `source/.gitignore` 的 `.sessions/` 规则覆盖.
