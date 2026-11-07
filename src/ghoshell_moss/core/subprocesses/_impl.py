"""Subprocesses 默认实现 — re-exported from :mod:`ghoshell_processes`.

实现已迁到独立包 ``ghoshell_processes``（与契约同源），这里只保留原路径。唯一的行为差异是
默认 logger：实现里不传 logger 时用 ``logging.getLogger(__name__)``；moss 的 host 在构造时
显式注入 ``LoggerItf``，因此运行时口径不变。
"""

from ghoshell_processes import Subprocesses, SubprocessesImpl

__all__ = ["Subprocesses", "SubprocessesImpl"]
