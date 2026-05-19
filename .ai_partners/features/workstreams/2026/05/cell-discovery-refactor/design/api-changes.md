# Cell Discovery — 具体 API 变更

## 1. Cell (core/blueprint/matrix.py)

### Before

```python
class Cell(ABC):
    name: str
    description: str
    type: CellType | str
    where: str

    @abstractmethod
    def is_alive(self) -> bool: ...

    def to_dict(self) -> dict[str, Any]:
        return {
            ...
            "is_alive": self.is_alive(),
        }
```

### After

```python
class Cell(ABC):
    name: str
    description: str
    type: CellType | str
    where: str
    reported_at: float = 0.0   # Unix timestamp, 0 = never seen

    @property
    def is_alive(self) -> bool:
        if self.reported_at == 0:
            return False
        return (time.time() - self.reported_at) < 30

    def to_dict(self) -> dict[str, Any]:
        return {
            ...
            "reported_at": self.reported_at,
            "is_alive": self.is_alive(),  # 保留兼容
        }
```

### Subclass changes

| Subclass | Before | After |
|---|---|---|
| `AppCell` | `is_alive()` returns `_alive_event.is_set()` | 移除 override，reported_at 由构造传入或网络更新 |
| `HostMainCell` | `is_alive()` returns `_alive_event.is_set()` | 同上 |
| `UnknownCell` | `is_alive()` returns `False` | reported_at=0，is_alive 自动返回 False |
| `FractalCell` | `is_alive()` returns `True` | reported_at=time.time()，is_alive 自动返回 True |

---

## 2. Matrix ABC (core/blueprint/matrix.py)

### 新增

```python
@abstractmethod
async def alist_cells(self) -> dict[CellAddress, Cell]:
    """异步查询全量 cell 状态。承诺不阻塞 event loop。"""
```

### 不变

`list_cells()` 签名和语义不变（返回本地缓存）。

---

## 3. MatrixImpl (host/matrix.py)

### 删除 (约 120 行)

- `_cell_alive_events: dict[str, threading.Event]` 属性
- `_register_cell_liveness_listener()` 方法 (line 507-522)
- `_check_initial_liveness()` 方法 (line 464-475)
- `_all_cell_liveness_check_ctx_manager()` 方法 (line 486-505)
- `_this_liveness_ctx_managers()` 方法 (line 454-462)
- `_matrix_cell_liveness_key_expr()` 方法 (line 481-483)
- `_matrix_cell_liveness_key_prefix()` 方法 (line 477-479)
- `__init__` 中 `cell_alive_events` 的构建逻辑 (line 117-144 中的 event 相关)
- `AppCell.__init__` 的 `event: threading.Event` 参数
- `HostMainCell.__init__` 的 `event: threading.Event` 参数

### 新增 (约 80 行)

6 个钩子方法 + 2 个发现方法:

```python
# --- Cell 注册表 ---

def _build_cells(self) -> dict[str, Cell]:
    """从 AppStore + main cell 构建初始 cell 注册表。
    不涉及网络，纯本地构建。"""

# --- 自我声明 ---

def _cell_info_json(self) -> bytes:
    """本 cell 的 JSON 序列化信息，用于 put 到网络。"""

def _announce_this_cell(self) -> None:
    """session.put("MOSS/{scope}/cells/{address}", info)"""

def _unannounce_this_cell(self) -> None:
    """session.delete("MOSS/{scope}/cells/{address}")"""

# --- 发现机制 (仅 main cell) ---

def _start_cell_discovery(self) -> None:
    """declare_queryable("MOSS/{scope}/cells/query")"""

def _stop_cell_discovery(self) -> None:
    """undeclare queryable"""

async def _query_cells_from_network(self) -> dict[str, Any]:
    """wildcard get("MOSS/{scope}/cells/**") → 解析 JSON → 返回"""

# --- 公共 API ---

def list_cells(self) -> dict[str, Cell]:
    """同步返回本地缓存。"""

async def alist_cells(self) -> dict[str, Cell]:
    """异步从网络查询，更新本地缓存后返回。"""
```

### __aenter__ 变更

```diff
- self._exit_stack.enter_context(self._all_cell_liveness_check_ctx_manager(zenoh_session))
- self._exit_stack.enter_context(self._this_liveness_ctx_managers(zenoh_session))
+ self._announce_this_cell()
+ self._start_cell_discovery()

  # liveness event 相关行删除:
- if event := self._cell_alive_events.get(self._this_cell_address):
-     event.set()
```

### __aexit__ 变更

```diff
+ self._stop_cell_discovery()
+ self._unannounce_this_cell()

  # liveness event 相关行删除:
- if event := self._cell_alive_events.get(self._this_cell_address):
-     event.clear()
```

---

## 4. 受影响的调用方

| 文件 | 改动 |
|---|---|
| `host/matrix.py` | 主改动，如上 |
| `host/fractal/_base.py` | FractalCell 移除 `is_alive()` override，设置 `reported_at = time.time()` |
| `host/repl/inspector_matrix.py` | `list_cells()` 调用不变；`to_dict()` 返回 key 名变化 |
| `host/repl/inspector_fractal.py` | `c.is_alive()` → 保持兼容（property 仍存在） |
| `host/tui.py` | `this.to_dict()` 的 key 名变化 `is_alive` → 同时有 `reported_at` |
| `host/stubs/.../matrix_exam/main.py` | `this.is_alive()` 调用仍兼容 |
| `core/blueprint/matrix.py` | Cell + Matrix ABC 变更 |
| `core/blueprint/.design/2026-05-09-fractal_design.md` | 设计文档引用更新 |

不需要改动的文件 (>20 个): 所有使用 `matrix.container`, `matrix.session`, `matrix.logger`, `matrix.create_task()`, `matrix.this`, `matrix.mode`, `matrix.manifests`, `matrix.channel_proxy()`, `matrix.provide_channel()` 的调用方。

---

## 5. Zenoh Key 约定

```
MOSS/{session_scope}/cells/{cell_address}   ← 每个 cell put 自己的 info (JSON)
MOSS/{session_scope}/cells/query             ← main cell 的 queryable
```

cell_address 格式保持: `{type}/{name}` (如 `host/main`, `app/myapp`)
