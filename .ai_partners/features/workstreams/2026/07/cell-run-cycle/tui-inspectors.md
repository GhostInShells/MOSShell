# TUI Inspector 体系重整 — M4a 规划

cell-run-cycle 的 TUI inspector 体系重整。背景：matrix-cell-governance 闭合后，
Matrix / Manifests / Cell / Project / Host ABC 全部重绘，现有 4 个 inspector 中
2 个需要修或重写。

## 现状诊断

| Inspector | 文件 | 状态 | 问题 |
|---|---|---|---|
| MatrixInspector | `inspector_matrix.py` | 部分可用 | `info()` 中 `matrix.mode.name` 不存在（Matrix ABC 无 mode 属性） |
| ManifestsInspector | `inspector_manifests.py` | **已死** | 依赖旧 `Manifests` god-basket；已从 REPL 拆线 |
| MOSSRuntimeInspector | `inspector_moss_runtime.py` | 正常 | 依赖 `MossRuntime` + `TuiRender`，新 API 兼容 |
| GhostInspector | `inspector_ghost.py` | 正常 | 依赖 `GhostRuntime`/`Ghost`/`Mindflow`/`MOSShell`，无变化 |

关键 API 变化：

- **Matrix**: 无 `mode` 属性（身份坐标在 `env`），表面积按 §YY 收敛（`this`/`network`/`mesh()`/
  `processes`/`jobs`/`home`）
- **Manifests**: `Manifests` ABC → `MatrixManifest` + `ModeManifests`，各方法返回
  `Iterable[Manifest[T]]`，`Manifest.is_error()` 显性化
- **Cell**: God-model 解体 → `CellManifest`/`CellRecord`/`CellPresence` 三域
- **Project**: 重写为治理域句柄，有 `matrix_manifests()` 和 `cells`
- **HostMode**: 有 `manifests()` 返回 `ModeManifests`

## I1. MatrixInspector — 修

**文件**: `src/ghoshell_moss/host/repl/inspector_matrix.py`

**构造**: `MatrixInspector(matrix: Matrix)` — 不变

**方法变更**:

- `this()` — 保留（原 `this_cell`，改名对齐 `matrix.this` 属性名）
- `identity()` — 新增：env 坐标快照（mode_name/ghost_name/network/scope/cell_address/pid）
- `info()` — 修：删 `matrix.mode.name`，改用 `is_running`/`is_host`/`is_host_running`
- `network()` — 新增：`matrix.network` 元信息
- `contracts()` — 保留
- `processes()` — 新增：`matrix.processes` 状态

原则：只观察 Matrix ABC 承诺的表面，不穿透实现细节。

## I2. ManifestsInspector — 重写

**文件**: `src/ghoshell_moss/host/repl/inspector_manifests.py`（原地重写）

**构造**: `ManifestsInspector(matrix_mf: MatrixManifest, mode_mf: ModeManifests | None)`

**方法**（每个 walk `Manifest[T]`，`is_error()` 条目标记不隐藏）:

- `explain()` — matrix_mf.explain() + mode_mf.explain() 拼接
- `providers()` — 两层（matrix + mode effective），name/singleton/found_at/error
- `configs()` — 两层，name/import_path/description/found_at/error
- `topics()` — 两层，name/topic_type/description/found_at/error
- `signals()` — 两层，name/description/found_at/error
- `parameters()` — 两层（单值 Manifest），name/description/found_at/error
- `resources()` — 两层，scheme/host/description/found_at/error
- `channel()` — mode 层专属，name/type/description/found_at
- `nuclei()` — mode 层专属，name/description/signal_names/found_at/error

返回值统一用 list[dict]（REPL 中 JSON 序列化友好）。`Manifest.is_error()` 为 True
的条目在 dict 中加 `"error": str(m.error())` 字段而非丢弃。

## I3. MOSSRuntimeInspector — 保留

**文件**: `src/ghoshell_moss/host/repl/inspector_moss_runtime.py`

不变。8 个方法全部兼容新 MossRuntime API（`moss_instruction`/`moss_dynamic_messages`/
`moss_exec`/`shell.commands` 等路径未变）。

## I4. GhostInspector — 保留

**文件**: `src/ghoshell_moss/host/repl/inspector_ghost.py`

不变。`GhostRuntime`/`Ghost`/`Mindflow`/`MOSShell` 均未在本轮重构中改变。

## I5. 接线重整

### MOSSRuntimeREPLState (`moss_runtime_ui.py`)

```python
def _create_repl_inspectors(self) -> dict[str, object]:
    moss = self._moss_runtime
    mode = moss.mode if moss.is_running() else None
    return {
        "matrix": MatrixInspector(moss.matrix),
        "manifests": ManifestsInspector(
            moss.project.matrix_manifests(),
            mode.manifests() if mode else None,
        ),
        "moss": MOSSRuntimeInspector(moss, self.console),
    }
```

### GhostREPLState (`ghost_ui.py`)

ManifestsInspector 追加到现有 `_create_repl_inspectors`（当前只有 ghost + matrix）。

### MossHostTUI.welcome() (`tui.py`)

`welcome()` 中 `self.host.matrix()` 调用属于 M1 纪律违反。顺修：改用
`self.runtime` 拿 matrix。但 `welcome()` 在 `__init__` 中调用，需确认
此时 `_get_runtime()` 已完成 — 检查调用时序后决定处置方式。

## 实施顺序

1. MatrixInspector — 修 `info()` + 加新方法
2. ManifestsInspector — 原地重写
3. 接线 — 更新两个 TUI state 的 `_create_repl_inspectors`
4. tui.py welcome() — 顺修 M1 纪律
5. 冒烟验证

## 验证

```bash
# inspector 模块可 import
.venv/bin/python -c "from ghoshell_moss.host.repl.inspector_matrix import MatrixInspector"
.venv/bin/python -c "from ghoshell_moss.host.repl.inspector_manifests import ManifestsInspector"

# 完整 TUI 冒烟（需 zenoh router）
python -m ghoshell_moss.host.tui_entries.ghost_ui
# REPL 中验证: /matrix.this()  /matrix.identity()  /manifests.explain()
#   /manifests.providers()  /moss.instructions()  /ghost.health()
```
