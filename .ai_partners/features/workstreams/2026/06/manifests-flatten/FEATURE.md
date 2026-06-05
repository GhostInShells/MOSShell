---
title: Manifests Flatten — 消除 MergedManifests，统一为显式继承模式
status: in-progress
priority: P0
created: 2026-06-05
updated: 2026-06-05
depends: []
milestone:
description: >-
  Delete MergedManifests, make every mode's manifests self-contained via explicit
  `from MOSS.manifests.xxx import *` + extend, and remove redundant --mode params
  from CLI subcommands. One manifest per mode, no hidden merge.
---

# Manifests Flatten

> Use `moss features set-status manifests-flatten <status> -m "note"` to update state.

## Motivation

### Problem 1: MergedManifests 制造隐式合并，造成理解负担

当前 `Host.__init__` 中：

```python
self._env_manifest = PackageManifests.from_environment(...)
self._manifest = MergedManifests([self._env_manifest, self._moss_mode.manifest])
```

这行背后藏着不一致的合并语义：
- **channels** — mode 的 `__main__` 完全覆盖全局
- **providers/configs/topics/resources/nuclei** — dict.update / list.extend 叠加
- **ctml_versions** — dict.update 叠加

问题：
1. 开发者看 mode 目录下的 `providers.py`（可能是空的），不知道运行时实际注册了哪些 provider
2. 合并规则是隐藏的 switch-case，不是从代码直接可见的
3. `MergedManifests` 是一个 95 行的类，唯一职责就是做这个合并
4. channels 已经采用了正确的模式——`from MOSS.manifests.channels import main` + 显式扩展——但没有推广到其他类型

### Problem 2: CLI --mode 参数去冗余

`main.py` 已在全局 callback 中定义了 `--mode` 并注入 `Environment`。但 `manifests_cli.py` 的 9 个命令、`apps_cli.py` 的 4 个命令、`scripts_cli.py` 的 3 个命令各自又定义了 `--mode` option。这是已承认的技术债（CLAUDE.md 标注 "待做"）。

## Key Decisions

### K1: 统一采用 channels 的显式继承模式

每个 mode 的 manifest 文件显式声明它继承了什么：

```python
# MOSS.modes.<name>/providers.py
from MOSS.manifests.providers import *
# mode 专属追加
my_provider = MyProvider()
```

不再有自动合并。manifest 文件的内容 = mode 的运行时能力。读文件即理解。

### K2: MergedManifests 弃用，不再参与 Host 构建

`MergedManifests` 类保留但标记为弃用，注释中写清观望原因。`Host.__init__` 直接使用 `self._moss_mode.manifest`，不再创建合并实例。

### K3: Mode manifest 构建方式不变

`_ensure_manifest_to_mode` 仍创建 `PackageManifests(mode_package)` 扫描 mode 目录。改变的只是 mode 目录下的文件内容（现在显式 import 全局 manifest + 扩展）。

### K4: `main_channel_source` 保留在 PackageManifests

`main_channel_source()` 返回 `__main__` channel 被发现时的模块路径。弃用 MergedManifests 后，返回的就是 mode 的 channel 所在模块（可能是 `MOSS.modes.<name>.channels` 或它显式 import 的 `MOSS.manifests.channels`——取决于 mode 怎么做）。CLI 通过 `getattr(host.manifests, 'main_channel_source', ...)` 访问，不需要检查 MergedManifests。

### K5: --mode 去冗余只做 manifests_cli

`manifests_cli.py` 的 9 个命令移除 `--mode` 参数，走全局。`apps_cli.py` 和 `scripts_cli.py` 的 --mode 参数涉及不同的调用链，分开处理（不在本次 scope）。

### K6: `from <module> import *` 依赖模块只导出 Provider/ConfigType 等实例

全局 manifest 模块（`MOSS.manifests.providers` 等）目前不定义 `__all__`。`from module import *` 会导入所有非 `_` 前缀的 public name。只要全局 manifest 模块只定义相关的声明实例（Provider、ConfigType 等），就不需要显式 `__all__`。

但如果全局 manifest 模块中混入了 import 的辅助类型（如 `from ghoshell_moss.host.providers import HostSessionProvider`），`from MOSS.manifests.providers import *` 也会把 `HostSessionProvider`（类，非 Provider 实例）导入 mode 模块。不过扫描器做 `isinstance(obj, Provider)` 过滤，非 Provider 实例会被自动忽略，无害。

## Design

### 改造前

```
Host.__init__
  ├── PackageManifests.from_environment()       # 扫描 MOSS.manifests.*
  ├── mode.manifest                             # PackageManifests 扫描 MOSS.modes.<name>.*
  └── MergedManifests([env_manifest, mode_manifest])  # 隐式合并 → Host.manifests
       ├── channels: mode 的 __main__ 完全覆盖
       ├── providers/configs/topics/nuclei/resources: extend/update
       └── ctml_versions: update
```

### 改造后

```
Host.__init__
  └── mode.manifest                             # PackageManifests 扫描 MOSS.modes.<name>.*
       │                                         # mode 的每个文件显式 from MOSS.manifests.xxx import *
       │                                         # + 自己的扩展
       └── Host.manifests (即 mode.manifest，单层)
```

### Mode 模板文件改造

每个 stub 文件从当前的注释-only 改为显式 import + 注释：

| 文件 | 改造前 | 改造后 |
|------|--------|--------|
| `channels.py` | `from ghoshell_moss import new_default_shell_main_channel` | 追加 `# from MOSS.manifests.channels import main` 选项注释 |
| `providers.py` | 空注释 | `from MOSS.manifests.providers import *` |
| `configs.py` | 空注释 | `from MOSS.manifests.configs import *` |
| `topics.py` | 空注释 | `from MOSS.manifests.topics import *` |
| `resources.py` | 空注释 | `from MOSS.manifests.resources import *` |
| `nuclei.py` | 空注释 | `from MOSS.manifests.nuclei import *` |

## Implementation Notes

### 改动文件清单

#### A. 核心代码（删除 MergedManifests）

| # | 文件 | 改动 |
|---|------|------|
| A1 | `src/ghoshell_moss/host/manifests/impl.py` | `MergedManifests` 类加弃用注释（不删除）。删除 `ENVIRONMENT_MODE_MANIFESTS_ROOT_PACKAGE`（L23）。删除 `PackageManifests.from_environment_moss_mode()`（L84-96）。`from_environment()` 保留。 |
| A2 | `src/ghoshell_moss/host/manifests/__init__.py` | `MergedManifests` 保留 re-export，加 `# Deprecated` 注释 |
| A3 | `src/ghoshell_moss/host/impl.py` | L13 移除 `MergedManifests` import。L50 移除 `self._env_manifest`。L65 改为 `self._manifest = self._moss_mode.manifest`。L79 不变。 |
| A4 | `src/ghoshell_moss/host/modes.py` | `_ensure_manifest_to_mode` 保持逻辑不变——仍创建 `PackageManifests(mode_package)`。但现在 mode 的 package 扫描的就是 mode 自己的文件（含显式 import），无需合并。 |

#### B. Stub 模板（引导 mode 显式继承）

| # | 文件 | 改动 |
|---|------|------|
| B1 | `src/ghoshell_moss/host/stubs/mode/providers.py` | 添加 `from MOSS.manifests.providers import *` |
| B2 | `src/ghoshell_moss/host/stubs/mode/configs.py` | 添加 `from MOSS.manifests.configs import *` |
| B3 | `src/ghoshell_moss/host/stubs/mode/topics.py` | 添加 `from MOSS.manifests.topics import *` |
| B4 | `src/ghoshell_moss/host/stubs/mode/resources.py` | 添加 `from MOSS.manifests.resources import *` |
| B5 | `src/ghoshell_moss/host/stubs/mode/nuclei.py` | 添加 `from MOSS.manifests.nuclei import *` |
| B6 | `src/ghoshell_moss/host/stubs/mode/channels.py` | 移除 "MergedManifests 合并时" 注释语言。添加 `# from MOSS.manifests.channels import main` 复用模式注释。 |
| B7 | `src/ghoshell_moss/host/stubs/workspace/src/MOSS/manifests/channels.py` | 移除 L12 "MergedManifests 合并时" 注释 |
| B8 | `src/ghoshell_moss/host/stubs/workspace/src/MOSS/manifests/configs.py` | 移除 L13 "MergedManifests 合并时" 注释 |

#### C. CLI — 去冗余 --mode

| # | 文件 | 改动 |
|---|------|------|
| C1 | `src/ghoshell_moss/cli/manifests_cli.py` | 9 个命令：移除 `mode: str \| None = typer.Option(...)` 参数及 `Host(mode=mode)` 中的 mode 传递，改为 `Host()` 走全局 Environment |

#### D. 工作空间实例文件

| # | 文件 | 改动 |
|---|------|------|
| D1 | `.moss_ws/src/MOSS/modes/default/providers.py` | 添加 `from MOSS.manifests.providers import *` |
| D2 | `.moss_ws/src/MOSS/modes/default/configs.py` | 添加 `from MOSS.manifests.configs import *` |
| D3 | `.moss_ws/src/MOSS/modes/default/topics.py` | 添加 `from MOSS.manifests.topics import *` |
| D4 | `.moss_ws/src/MOSS/modes/default/resources.py` | 添加 `from MOSS.manifests.resources import *` |
| D5 | `.moss_ws/src/MOSS/modes/default/nuclei.py` | 添加 `from MOSS.manifests.nuclei import *` |
| D6 | `.moss_ws/src/MOSS/modes/default/channels.py` | 可选改造：从 `new_shell_main_channel()` 改为 `from MOSS.manifests.channels import main`（验证复用模式） |
| D7 | `.moss_ws/src/MOSS/manifests/channels.py` | 移除 "MergedManifests 合并时" 注释 |
| D8 | `.moss_ws/src/MOSS/manifests/configs.py` | 移除 "MergedManifests 合并时" 注释（如存在） |

#### E. 文档

| # | 文件 | 改动 |
|---|------|------|
| E1 | `src/ghoshell_moss/core/blueprint/manifests.py` | `Manifests.explain()` — 改 modes 部分：从 "叠加/合并" 语言改为 "显式继承" 语言 |
| E2 | `src/ghoshell_moss/host/manifests/impl.py` | `PackageManifests.explain()` — 更新 mode package 的自描述 |
| E3 | `src/ghoshell_moss/cli/how_tos/host-dev/register-manifests.md` | 移除 "Mode 合并" 列，改为显式继承说明 |
| E4 | `src/ghoshell_moss/cli/how_tos/host-dev/create-a-mode.md` | 更新 manifest 叠加→继承语义，更新 channels.py 示例 |
| E5 | `src/ghoshell_moss/cli/docs/workspace-and-mode.md` | 更新 manifests 合并部分（多处） |
| E6 | `src/ghoshell_moss/cli/docs/matrix-system.md` | 更新 L127-135 manifests 部分 |
| E7 | `src/ghoshell_moss/cli/docs/glossary.md` | 更新 manifest/Mode 条目 |
| E8 | `.moss_ws/src/MOSS/modes/CLAUDE.md` | 更新 Mode 开发指南中的合并语义 |

#### F. 无需改动的文件

- `src/ghoshell_moss/host/matrix.py` — `MatrixImpl` 接收 `Manifests` 接口，不需要知道内部是 PackageManifests 还是 MergedManifests
- `src/ghoshell_moss/host/repl/inspector_manifests.py` — `ManifestsInspector` 消费 `Manifests` 接口，不受影响
- `tests/ghoshell_moss/host/test_matrix_init.py` — mock 的是 `Manifests` 接口，不受影响
- `src/ghoshell_moss/host/tui_entries/moss_runtime_ui.py` — L32 只传 `host.manifests`，不受影响
- `.ai_partners/features/` 下的 FEATURE.md — 历史记录，不修改
- `MergedManifests` 类 — 保留但弃用，注释标记

### 注意事项

1. **`from module import *` 的去重**：扫描器 `search_provider_infos_from_package` 已经用 `set(providers)` 按对象 identity 去重，所以即使 mode 和某个子模块重复导入了同一个 Provider 实例，不会重复注册。

2. **`main_channel_source` 的兼容**：`manifests_cli.py:325` 用 `getattr(host.manifests, 'main_channel_source', lambda: None)()` 调用。删除 MergedManifests 后，`host.manifests` 是 `PackageManifests` 实例（有 `main_channel_source` 方法），调用正常。

3. **default mode 是特例**：default mode 的 manifest 文件做 `from MOSS.manifests.xxx import *`，即与全局完全一致。非 default mode 可以只 import 需要的，也可以完全覆盖。

4. **ctml_versions 需要额外处理**：`ctml_versions` 不是 Python 模块声明，而是 workspace 目录下的 `.md` 文件扫描。当前流程：
   - `PackageManifests.from_environment()` → `find_ctml_versions_from_env()` → 扫描全局 ctml dirs
   - `MergedManifests.__init__` → `self._ctml_versions.update(manifest.ctml_versions())`
   删除 MergedManifests 后，`_ensure_manifest_to_mode` 直接调用 `PackageManifests(package_path)`，ctml_versions 为空 dict。
   **解决**：在 `_ensure_manifest_to_mode` 中也调用 `find_ctml_versions_from_env()`：
   ```python
   def _ensure_manifest_to_mode(package_path: str, mode: Mode) -> Mode:
       if mode.__manifest__ is None:
           env = Environment.discover()
           env.bootstrap()
           ctml_versions = PackageManifests.find_ctml_versions_from_env(env=env)
           mode.with_manifest(PackageManifests(package_path, ctml_versions=ctml_versions))
       return mode
   ```
   这样 mode 始终有环境级别的 ctml_versions，除非 mode 自己有 `ctml_versions/` 目录。

## Acceptance Path

### 验证 1: 核心功能回归

```bash
# manifests explain 不再显示合并规则，改为显式继承描述
moss manifests explain
moss --mode default manifests explain

# 所有 manifest 子命令返回相同结果（全局 = default mode，因为 default mode 显式 import *）
moss manifests providers
moss --mode default manifests providers
diff <(moss --ai manifests providers) <(moss --ai --mode default manifests providers)

# channels 同样
moss manifests channels
moss --mode default manifests channels
```

### 验证 2: 显式继承 = 原有合并

```bash
# 改造后 default mode 的 providers/configs/topics/nuclei/resources
# 结果应与改造前完全一致（因为 default mode 显式 import * 了全局的）
moss --ai manifests providers
moss --ai manifests configs
moss --ai manifests topics
moss --ai manifests nuclei
moss --ai manifests resources
```

### 验证 3: Mode 创建模板生成正确

```bash
# 创建测试 mode
moss modes create test_manifest_verify -d "manifest flatten verify"
# 检查生成的文件是否正确包含 from MOSS.manifests.xxx import *
grep "from MOSS.manifests" $(moss --ai modes show test_manifest_verify | grep "Mode Directory" | awk '{print $NF}')/*.py
# 清理
rm -rf <mode_dir>
```

### 验证 4: CLI --mode 去冗余

```bash
# 验证 manifests commands 不再有独立的 --mode help text
moss --ai help manifests providers | grep -c "mode"
# 预期: 0 (mode 不应该出现在命令级别 help 中)

# 但全局 --mode 仍然生效
moss --mode default manifests providers  # 正常工作
```

### 验证 5: Host/Matrix 启动通过

```bash
# moss-repl 正常启动
moss-repl  # 快速验证启动无异常
# moss-as-mcp 正常启动
moss-as-mcp --help  # 至少参数解析正常
```

### 验证 6: IoC 容器注册正确

```bash
# 确保 MatrixImpl._prepare_container 注册的 provider 数量不变
moss --ai manifests providers | wc -l  # 改造前后对比
moss --ai manifests contracts | wc -l  # 改造前后对比
```

### 验证 7: 现有测试

```bash
pytest tests/ghoshell_moss/host/test_matrix_init.py -v
pytest tests/ghoshell_moss/host/test_environment_set_mode.py -v
```


## 2026-06-05 实施记录

### 已完成

- [x] Stub 模板 (B1-B8): mode stub 文件全部加 `from MOSS.manifests.xxx import *`
- [x] 工作空间实例 (D1-D8): `.moss_ws` default mode 和全局 manifest 同步更新
- [x] 核心代码 (A1-A4): Host 直接使用 mode.manifest，MergedManifests 弃用保留
- [x] CLI (C1): manifests_cli.py 9 个命令移除 --mode 参数
- [x] 关键文档 (E1-E4): explain()、create-a-mode.md、register-manifests.md、modes CLAUDE.md
- [x] 验收: 11 tests pass, providers 数量不变, mode 创建模板正确

### 待办 (下一个会话)

- [ ] E5: workspace-and-mode.md — manifests 合并语言改为显式继承
- [ ] E6: matrix-system.md — L127-135
- [ ] E7: glossary.md — manifest/Mode 条目
- [ ] 其余 docs/*.md 中 MergedManifests 引用清理
