"""安全 Python 代码执行沙盒 | 基础设施 | alpha

基于 ModuleType 容器，builtins 可控，父子命名空间共享，有状态生命周期。
"""

from types import ModuleType
from typing import Any, Callable
import builtins
import io
import traceback
from contextlib import redirect_stdout

__all__ = ["Sandbox", "SANDBOX_BUILTINS"]

_SANDBOX_BLOCKED = frozenset({
    "__import__", "open", "eval", "exec", "compile",
    "input", "breakpoint",
})

SANDBOX_BUILTINS: dict[str, Any] = {
    k: v for k, v in vars(builtins).items()
    if k not in _SANDBOX_BLOCKED
}


class Sandbox:
    """安全 Python 代码执行沙盒。

    builtins 可控，init/destroy 生命周期，父子命名空间共享。
    exec() 捕获 stdout 和异常 traceback。

    Example:
        sandbox = Sandbox(name="demo")
        sandbox.exec("x = 1 + 2\\nprint(x)")
        assert sandbox.get("x") == 3
        sandbox.close()
    """

    def __init__(
        self,
        name: str = "__sandbox__",
        *,
        parent: "Sandbox | None" = None,
        builtins: dict[str, Any] | None = SANDBOX_BUILTINS,
        on_init: Callable[["Sandbox"], None] | None = None,
        on_destroy: Callable[["Sandbox"], None] | None = None,
    ):
        self._name = name
        self._parent = parent
        self._builtins = builtins
        self._on_destroy = on_destroy
        self._closed = False

        if parent is not None:
            if parent._closed:
                raise RuntimeError("Parent sandbox is closed")
            self._module = parent._module
        else:
            self._module = ModuleType(name)

        if on_init is not None:
            on_init(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def parent(self) -> "Sandbox | None":
        return self._parent

    def _swap_builtins(self):
        """Temporarily install this sandbox's builtins into the shared namespace."""
        ns = self._module.__dict__
        self._prev_builtins = ns.get("__builtins__", _MISSING)
        if self._builtins is not None:
            ns["__builtins__"] = self._builtins

    def _restore_builtins(self):
        ns = self._module.__dict__
        if self._prev_builtins is _MISSING:
            ns.pop("__builtins__", None)
        else:
            ns["__builtins__"] = self._prev_builtins

    def exec(self, code: str) -> str:
        """执行 Python 代码，返回 stdout + 异常 traceback（如有）。

        代码在持久化命名空间中执行，变量跨调用累积。
        每次执行时应用本 sandbox 的 builtins 控制。
        """
        if self._closed:
            raise RuntimeError(f"Sandbox '{self._name}' is closed")

        self._swap_builtins()
        try:
            buffer = io.StringIO()
            try:
                with redirect_stdout(buffer):
                    exec(code, self._module.__dict__)
            except Exception:
                output = buffer.getvalue()
                if output and not output.endswith("\n"):
                    output += "\n"
                output += traceback.format_exc()
                return output
            return buffer.getvalue()
        finally:
            self._restore_builtins()

    def get(self, name: str) -> Any:
        """获取命名空间中的变量。"""
        if name not in self._module.__dict__:
            raise AttributeError(f"'{self._name}' has no attribute '{name}'")
        return self._module.__dict__[name]

    def set(self, name: str, value: Any) -> None:
        """设置命名空间中的变量。"""
        self._module.__dict__[name] = value

    def close(self) -> None:
        """关闭沙盒，触发 on_destroy。"""
        if self._closed:
            return
        self._closed = True
        if self._on_destroy is not None:
            self._on_destroy(self)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        return None


_MISSING = object()
