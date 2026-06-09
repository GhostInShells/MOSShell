"""Layout state snapshot — 每个 layout 独立管理，支持持久化和版本控制。

通过 Reflex get_state API 一把读取 ComponentState 所有字段值，
写入内存快照并可选持久化到 JSON 文件。
"""

import json
import logging
import typing
from pathlib import Path

import reflex as rx
from PIL import Image

logger = logging.getLogger(__name__)

SNAPSHOT_VERSION = 1


class LayoutSnapshot:
    """单个 layout 的状态快照管理器。"""

    def __init__(self, name: str, state_class: type[rx.State], storage_dir: Path) -> None:
        self._name = name
        self._state_class = state_class
        self._storage_path = storage_dir / name / "snapshot.json"
        self._data: dict[str, object] = {}
        self._init_defaults()

    # ---- public ----

    async def refresh(self, root_state: rx.State) -> None:
        """通过 root_state.get_state() 一把读取所有字段运行时值。"""
        try:
            layout_state = await root_state.get_state(self._state_class)
        except Exception:
            logger.warning("get_state failed for layout %r", self._name, exc_info=True)
            return

        for name, type_hint in self._state_class.__annotations__.items():
            val = getattr(layout_state, name, None)
            self._data[name] = self._summarize(val, type_hint)

    def get(self) -> dict[str, object]:
        return self._data

    def save(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": SNAPSHOT_VERSION, "data": self._data}
        self._storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, default=str)
        )

    # ---- internal ----

    def _init_defaults(self) -> None:
        for name, type_hint in self._state_class.__annotations__.items():
            self._data[name] = _default_for(type_hint)

    @staticmethod
    def _summarize(val: object, type_hint: type) -> object:
        """将运行时值压缩为 token 友好的摘要。

        阶段规则：
        - None           → 按类型给默认哨兵值
        - str            → 截断 200 字符
        - list[str]      → 前 5 项，每项 ≤60 字符，总量 ≤300 字符
        - list[Image]    → 前 5 项，每项报尺寸
        - 其他 list      → 仅报长度
        - Image.Image    → 报尺寸
        - 其他           → str() 后截断 200 字符
        """
        if val is None:
            return _default_for(type_hint)
        if isinstance(val, str):
            return val[:200] + "..." if len(val) > 200 else val
        if isinstance(val, list):
            return _summarize_list(val, type_hint)
        if isinstance(val, Image.Image):
            return f"<Image {val.size}>"
        return str(val)[:200]


def _summarize_list(val: list, type_hint: type) -> object:
    """根据元素类型选择列表摘要策略。"""
    args = typing.get_args(type_hint)
    elem_type = args[0] if args else None
    if elem_type is str:
        return _summarize_str_list(val)
    if elem_type is Image.Image:
        return f"<Images len={len(val)}>"
    return len(val)


def _summarize_str_list(val: list[str]) -> str:
    """list[str]：展示前 5 项内容，每项 ≤60 字符，总量 ≤300 字符。"""
    parts: list[str] = []
    for item in val[:5]:
        s = str(item)
        parts.append(s[:60] + "..." if len(s) > 60 else s)
    result = ", ".join(parts)
    if len(val) > 5:
        result += f", ... (+{len(val) - 5} more)"
    if len(result) > 300:
        result = result[:300] + "..."
    return result


def _default_for(type_hint: type) -> object:
    """类型对应的默认哨兵值（refresh 前使用）。"""
    origin = typing.get_origin(type_hint)
    if origin is list:
        return 0
    if type_hint is str:
        return ""
    if type_hint is Image.Image:
        return "<no image>"
    return "(unknown)"
