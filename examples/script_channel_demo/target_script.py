"""Demo target script to be loaded as a Channel.

`script_channel` 使用 stdout 作为协议通道，所以这个脚本不要 print 到 stdout。
如果需要日志，请使用 logging 并输出到 stderr。
"""

import logging

logger = logging.getLogger(__name__)


def add(a: int, b: int) -> int:
    return a + b


async def hello(name: str = "world") -> str:
    logger.warning("hello() called with name=%s", name)
    return f"Hello, {name}!"
