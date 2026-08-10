"""
管理 ghoshell_moss 第三方依赖的检查.

契约: ``depend_*()`` 是纯 gate —— 要么正常返回, 要么抛 ``ImportError``.
内部用 ``importlib.util.find_spec`` 做轻量检查, **不执行包的 import**:
import 做存在性检查会把重型包装进 ``sys.modules`` (如 anthropic 拖入
上千个模块). 真实使用点的 import 由调用方自己负责 (惰性 import).

``available()`` 是布尔查询, 供条件注册 (如 CLI 命令按 extra 显隐) 使用.
"""

import importlib.util

_available_cache: dict[str, bool] = {}


def available(*module_names: str) -> bool:
    """模块是否可寻址 (已安装). find_spec 不执行包代码, 进程内缓存.

    只传顶层模块名 (anthropic / pydantic_ai / zenoh ...) —— 点分名会触发
    父包导入, 又变重.
    """
    for name in module_names:
        if name not in _available_cache:
            _available_cache[name] = importlib.util.find_spec(name) is not None
        if not _available_cache[name]:
            return False
    return True


def _require(*module_names: str, hint: str) -> None:
    if not available(*module_names):
        raise ImportError(hint)


def depend_cli():
    _require("typer", "rich", "dotenv", hint="install ghoshell_moss[cli]")


def depend_matrix():
    depend_cli()
    _require("zenoh", hint="install ghoshell_moss[matrix]")


def depend_host():
    depend_matrix()
    _require("prompt_toolkit", "pexpect", hint="install ghoshell_moss[host]")


def depend_mcp():
    _require("mcp", hint="mcp not installed. run: uv sync --all-extras")


def depend_ghost():
    _require("pydantic_ai", "anthropic", hint="install ghoshell_moss[ghost]")
