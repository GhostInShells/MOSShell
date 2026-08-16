# source — dsh 源码快捷路径 (只读)

这里**只放 clone 下来的 dsh 源码供调研观察**, 不运行、不 import、不在此开发.
调研动作 (脚本/环境) 在同级 `skills/` 与 `home/`, 核心模块在 `src/ghoshell_moss/`.

源码目录 (`deepseek-harness/`) 被 `.gitignore` 排除, 不进 MOSS 仓库. 还原方式:

```sh
cd source
git clone https://github.com/deepseek-ai/deepseek-harness deepseek-harness
# 源码仓库无 tag, release 以 commit 记录. 对齐最新发布 commit (0.1.0-rc.5):
cd deepseek-harness && git checkout abe560f81e
```

## 版本对齐

- 源码仓库**无 tag**, release 以 commit 记录; 最新发布 commit 是
  `abe560f81e` = `release(dsh): 0.1.0-rc.5`.
- 默认对齐该 commit. 观察/调试时可在目标目录内切分支, 不影响 MOSS 侧依赖锁定.
- **注意**: optional 依赖 `deepseek-harness-sdk==0.1.0rc6` (PyPI) 与源码仓库最新
  release (rc.5) 存在一个版本差 — 源码侧没有对应 rc6 的 commit, 此差接受.
- optional 依赖版本见 MOSS pyproject `[project.optional-dependencies]` 的 `dsh` extra.
