# dsh-src — dsh 源码工作区 (调研对照用)

clone 下来的 dsh 源码默认对齐**最新发布 commit**, 供观察与调试. 此目录被
`.gitignore` 排除, 不进 MOSS 仓库 — 还原调研场景的方式如下.

## 还原 clone

```sh
cd dsh-src
git clone https://github.com/deepseek-ai/deepseek-harness deepseek-harness
# 源码仓库无 tag, release 以 commit 记录. 对齐最新发布 commit (0.1.0-rc.5):
cd deepseek-harness && git checkout abe560f81e
```

## 版本对齐

- **源码仓库无 tag**, release 以 commit 记录; 最新发布 commit 是
  `abe560f81e` = `release(dsh): 0.1.0-rc.5`.
- 默认对齐该 commit. 观察/调试时可在目标目录内切分支, 不影响 MOSS 侧依赖锁定.
- **注意**: optional 依赖 `deepseek-harness-sdk==0.1.0rc6` (PyPI) 与源码仓库最新
  release (rc.5) 存在一个版本差 — 源码侧没有对应 rc6 的 commit, 此差接受.
- optional 依赖版本见 MOSS pyproject `[project.optional-dependencies]` 的 `dsh` extra

## 调研工作区 (probes/)

`probes/` 承载 dsh 脚本调研. 每个调研是独立流程, 由模型自己跑, 不写持久化脚本
(脚本会 stale). 调研的动机/方法/结果记录在跑完后的产出中.

### 运行时不污染 ~/.dsh

- `probes/.dsh-home/` 是**独立的 DSH_HOME**, 与 `~/.dsh` 隔离.
- 启动声明 (profiles/web/*.yml + package.json) 提交进仓库, 可重建产物全 ignore
  (settings.yaml / storages/ / sessions/ / node_modules).
- 运行调研时让 dsh 用这个 home:

  ```sh
  export DSH_HOME=probes/.dsh-home
  dsh --profile web --port 3081
  ```

  首次运行会生成 settings.yaml / node_modules / sessions 等产物, 均在 ignore 内.
  如需共享插件库 (profiles/node_modules), 软链到全局安装, 不入仓库.
