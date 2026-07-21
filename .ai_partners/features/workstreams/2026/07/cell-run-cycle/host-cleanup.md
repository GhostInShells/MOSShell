# Host 废弃代码清理

cell-run-cycle 的子任务 — 清理 `ghoshell_moss/host/` 下的死代码。

## D1. 直接删除 (无依赖)

| 文件 | 原因 |
|------|------|
| `host/matrix.py` | DEPRECATED, raise ImportError |
| `host/app_store.py` | 465 行全部注释掉 |
| `host/ghosts.py` | 零导入者, project/manifests/ghosts.py 替代 |
| `host/repl/echo_case.py` | 测试脚手架, 零导入者 |
| `host/session/` (空目录) | 只有 __pycache__ |
| `host/topics/` (空目录) | 只有 __pycache__ |
| `host/channels/` (空目录) | 空 __init__.py, 零导入者 |

## D2. host/manifests/ 清理

整个目录被 `project/manifests/` 替代，无活跃调用者。

- 删 `MergedManifests` 类 (自声明 DEPRECATED)
- 删 `search_channels_from_package` (自声明 deprecated)
- 评估是否整个目录删除（架构只被 architecture.py 反射引用）

## D3. host/modes.py 清理

- 删注释掉的 `list_modes_from_root_package`
- 删注释掉的 `find_mode_from_package`  
- 删注释掉的 `_ensure_manifest_to_mode`
- 确认 `new_mode()` 是否有调用者，无则也删

## D4. architecture.py 更新

- 删 `import ghoshell_moss.host.manifests as host_manifests`（如果 host/manifests/ 整个删除）
