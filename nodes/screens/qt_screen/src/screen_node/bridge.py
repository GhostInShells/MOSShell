"""ScreenBridge — thread-safe QObject bridge between Matrix (asyncio) and Qt (GUI).

Decision 9 three-path single-writer architecture:

  Control in  (channel → GUI):  Signal(str) queued connection + Future
  Rendering   (GUI internal):   QML property bindings, channel never touches
  State out   (GUI → channel):  EventBucket (peek for context, drain for signal)

All scene mutation happens on the GUI thread. Channel commands post operations
via Signal with Qt.QueuedConnection; the handler executes on the GUI thread
and resolves the Future. The channel side awaits via asyncio.wrap_future.
"""

from __future__ import annotations

import json
import logging
import threading
from uuid import uuid4
from concurrent.futures import Future
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot, Qt
from PySide6.QtQml import QJSValue

from .bucket import EventBucket

logger = logging.getLogger("moss.screen.bridge")

# Animation durations in milliseconds — must match QML Behavior/Animation values.
ANIM_FOCUS_MS = 1100
ANIM_TRANSITION_MS = 800


class ScreenBridge(QObject):
    """Cross-thread bridge: Matrix thread submits ops, GUI thread executes.

    Exposed to QML via rootContext().setContextProperty("bridge", self).
    QML calls human_clicked / web_badge_changed (Slots) directly.
    """

    # --- Signal for cross-thread dispatch (Python thread → Qt thread) ---
    # Connected with Qt.QueuedConnection for thread safety.
    _dispatch = Signal(str)

    # --- Signals for QML binding (Python → QML) ---
    window_opened = Signal(str, str, str)       # id, url, label
    window_closed = Signal(str)                  # id
    window_badge_changed = Signal(str, int)      # id, badge
    layout_switched = Signal(str)                # layout_name
    slot_changed = Signal(str, str, str)         # slot_type, slot_name, window_id
    slot_cleared = Signal(str, str)              # slot_type, slot_name
    background_changed = Signal(str)             # window_id

    def __init__(self, bucket: EventBucket, parent: QObject | None = None):
        super().__init__(parent)
        self._bucket = bucket
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._root: QObject | None = None
        self._snapshot_cache: dict = {}  # written on GUI thread, read on Matrix thread

        # Queued connection: emit from any thread, handled on GUI thread.
        self._dispatch.connect(self._on_dispatch, Qt.QueuedConnection)

    # ---- QML root -----------------------------------------------------------

    def set_root(self, root: QObject) -> None:
        self._root = root

    # ---- submit (Matrix thread) ---------------------------------------------

    def submit(self, op: str, args: dict[str, Any] | None = None) -> Future:
        """Matrix thread: submit an operation, return Future resolved by GUI thread."""
        rid = str(uuid4())
        f: Future = Future()
        with self._lock:
            self._futures[rid] = f
        payload = json.dumps({"rid": rid, "op": op, "args": args or {}})
        self._dispatch.emit(payload)
        return f

    # ---- dispatch handler (GUI thread) --------------------------------------

    @Slot(str)
    def _on_dispatch(self, payload: str) -> None:
        """GUI thread: execute operation and resolve Future."""
        data = json.loads(payload)
        rid = data["rid"]
        op = data["op"]
        args = data["args"]

        try:
            result = self._execute(op, args)
        except Exception as exc:
            logger.exception("bridge dispatch failed: op=%s args=%s", op, args)
            result = exc

        self._refresh_snapshot()

        with self._lock:
            f = self._futures.pop(rid, None)
        if f is not None:
            if isinstance(result, Exception):
                f.set_exception(result)
            else:
                f.set_result(result)

    _DISPATCH: dict[str, tuple[str, ...]] = {
        # op -> ordered positional arg names matching QML function signatures
        "open_window":     ("id", "url", "label"),
        "close_window":    ("id",),
        "set_background":  ("id",),
        "switch_layout":   ("name",),
        "focus_window":    ("id", "slot"),
        "front_window":    ("id", "index"),
        "float_window":    ("id",),
        "clear_slot":      ("slot",),
    }

    def _execute(self, op: str, args: dict) -> Any:
        """Execute operation on QML scene. Runs on GUI thread only."""
        root = self._root
        if root is None:
            raise RuntimeError("ScreenBridge: QML root not set")

        method = getattr(root, op, None)
        if method is None:
            raise ValueError(f"Unknown bridge op: {op}")

        # QML functions require positional args, not keyword args.
        # Build positional args list from the dispatch signature.
        param_names = self._DISPATCH.get(op, ())
        if param_names:
            pos_args = [args.get(name) for name in param_names]
            return method(*pos_args)
        # Fallback for ops without explicit dispatch (e.g. future additions)
        return method(**args)

    # ---- snapshot (Matrix thread) -----------------------------------------

    def snapshot(self) -> dict:
        """Matrix thread: return thread-safe copy of current screen state."""
        with self._lock:
            return self._snapshot_cache.copy()

    @staticmethod
    def _to_native(v: Any) -> Any:
        """Convert QJSValue to Python dict/list/str recursively.

        PySide6 does NOT auto-convert QML property var returns to Python
        types (unlike PyQt6). QML property var stores JS values wrapped
        as QJSValue; toVariant() unwraps them for shiboken auto-conversion.
        """
        if isinstance(v, QJSValue):
            v = v.toVariant()
        if isinstance(v, dict):
            return {k: ScreenBridge._to_native(val) for k, val in v.items()}
        if isinstance(v, list):
            return [ScreenBridge._to_native(item) for item in v]
        return v

    def _refresh_snapshot(self) -> None:
        """GUI thread: read QML state into cache."""
        root = self._root
        if root is None:
            return

        try:
            windows_raw = root.property("windows")
            layout_name_raw = root.property("layoutName")
            background_id_raw = root.property("backgroundId")
            focus_id_raw = root.property("focusId")
            front_ids_raw = root.property("frontIds")
            float_ids_raw = root.property("floatIds")
        except Exception:
            logger.exception("_refresh_snapshot: QML property read failed")
            return

        windows = self._to_native(windows_raw) or {}
        front_ids = self._to_native(front_ids_raw) or []
        float_ids = self._to_native(float_ids_raw) or []

        snap = {
            "windows": windows,
            "layout": {
                "name": str(self._to_native(layout_name_raw) or "solo"),
                "background": str(self._to_native(background_id_raw) or ""),
                "slots": {
                    "focus": str(self._to_native(focus_id_raw) or ""),
                    "front": front_ids,
                    "float": float_ids,
                },
            },
        }

        with self._lock:
            self._snapshot_cache = snap

    # ---- Slots: QML → Python (called from QML/JS on GUI thread) ------------

    @Slot(str, str)
    def human_clicked(self, window_id: str, action: str) -> None:
        """QML calls this when a human clicks a meta item or window."""
        self._bucket.push("human_clicked", window_id=window_id, action=action)

    @Slot(str, int)
    def web_badge_changed(self, window_id: str, badge: int) -> None:
        """QML calls this when a WebEngineView page calls navigator.setAppBadge."""
        self._bucket.push("badge_changed", window_id=window_id, badge=badge)
        # Also update the QML meta item badge via signal
        self.window_badge_changed.emit(window_id, badge)

    @Slot(str)
    def animation_finished(self, operation_id: str) -> None:
        """QML calls this when a layout animation completes."""
        with self._lock:
            f = self._futures.pop(operation_id, None)
        if f is not None:
            f.set_result(True)
