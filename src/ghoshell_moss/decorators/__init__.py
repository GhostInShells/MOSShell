"""decorators — 把外部能力包成普通可调用函数的边界原语.

目前只有一个 ``@cli``: 把真实 CLI 命令包装成签名即契约的异步函数.
"""

from ghoshell_moss.decorators.cli import CliResult, cli

__all__ = ["cli", "CliResult"]
