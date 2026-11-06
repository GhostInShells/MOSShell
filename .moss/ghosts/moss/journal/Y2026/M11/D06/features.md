# features —— 发现路径常量化，stub 变真模板

## 1. 发现路径（`c401d5c2`）

原来的实现：读的时候看 `$CWD/features`，否则 `$CWD/.ai_partners/features`（只在一层，不上溯）；`init` 写 `$CWD/features`。**读和写在两个地方**，而且没有任何地方命名过这个布局。

他给的方案：两段发现——先向上找最近的 `.git` 定 project home（找不到就是"不是项目"），再在 `$CWD → project home` 这一段里逐级找；`.features` 与 `.ai_partners/features` 都是合法名，前者优先。

落地：常量集中在 `core/codex/features.py`

```python
FEATURES_DIRNAME = Path(".features")
LEGACY_FEATURES_DIRNAME = Path(".ai_partners") / "features"
FEATURES_DIRNAMES = (FEATURES_DIRNAME, LEGACY_FEATURES_DIRNAME)   # 顺序即优先级
PROJECT_MARKER = ".git"          # worktree 里它是文件，所以用 exists()
FEATURES_ANCHOR = "workstreams"  # 什么样的目录才算 features 根
```

`resolve_features_dir()` 改返回 `Optional[Path]`，新增 `require_features_dir()` 抛 `FeaturesDirNotFoundError`；CLI 只剩一个 `_features_root()` 把拒绝翻成一行错误 + exit 1，每个命令还会打一行 `features: <路径>`。`init` 不参与发现，默认 `$CWD/.features`。

他两次强调同一件事：**实现逻辑不许塞进 CLI**，`features.py` 是家，cli 是调用方。

## 2. #139：init 把 MOSS 自己的实例当模板

`moss features init` 把 `_features_stubs/` 拷给新项目，而其中 `SURFACE.md`、`stages/ROADMAP.md` 与 `.ai_partners/features/` 下的实例**字节相同**——MOSS 自己的 L3 清单和路线图，连同指向不存在的 stage 目录的死链，一起进了别人的项目。仓库里也根本没有 SURFACE 模板。

修法（`12303b07`）：给 SURFACE 和 ROADMAP 写真正的模板；`stages/README.md` 从"# MOSS Stage Tracking"改成项目无关的机制文档；TOPOLOGY 去掉 pillar 编号并改用 `.features/` 作目录名；regressions/README 的 `features/...` 路径改成相对树根；`find_stubs_dir()` 删掉回落到 `.ai_partners/features` 的第二优先级（那正是"把实例当脚手架"）。

他的一句判据：**可回滚的破坏性操作不需要保护，需要的是它发生在版本控制之下**——所以 `init` 的 full overwrite 不用改，重跑错了就 reset。

## 3. 方法论上的两条

- 模板只填 spec 没说的部分。第一版 SURFACE 79 行写了"这是什么/不是什么"，他一句"删掉和 specification 完全重复的内容"砍到 24 行。**spec 已经说过的话，模板一个字都不该重复。**
- 一个类目只有一个成员时不要分类。skills 一开始想按 `runtime/`、`capability/` 分两层，他判断这个池子不会大到需要分类，于是平铺。
