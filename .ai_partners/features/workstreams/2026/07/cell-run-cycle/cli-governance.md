# CLI Governance — M1/M3/M4b 纪律修正

cell-run-cycle 的 CLI 体系治理任务。目标：清除所有致命崩溃点 + 纪律违反。

## T1. M4b — 删除 fractal 遗骸

**判据**: `moss_as_fractal.py` 及其依赖的 fractal 模块全部删除。
- `cli/moss_as_fractal.py`
- `host/repl/inspector_fractal.py`
- `core/blueprint/fractal.py`
- `host/fractal/` 整个目录
- `pyproject.toml` 中的 `moss-as-fractal` 入口点

## T2. utils.py — 删除死代码 print_host_mode_info

**判据**: `print_host_mode_info` 函数删除，`from ghoshell_moss.host import Host` 删除。
零调用者，且从 concrete Host 上拿属性是 M1 同类病灶。

## T3. M3 — ghost_run.py seal 对齐

**判据**: `Environment.discover()` + `set_*` → `Environment(mode=, scope=).seal()`。
Ghost name 不再 set 到 env，直接传给 GhostTUI。

## T4. M3 — cli_controller.py seal 对齐

**判据**: 所有 `set_*` 调用消除，不再运行时崩溃。
- `main_entry()`: CLI overrides 通过 `os.environ` 或重建 Environment
- `interactive_config()`: 交互式选择的 mode/ghost/scope 通过环境变量传递
  （子进程执行模型天然适合 env var 传递，不依赖 in-place mutate）

## T5. M1 — moss_as_mcp.py 继承纪律

**判据**: `moss_host.matrix().logger` → `toolset.matrix.logger`
