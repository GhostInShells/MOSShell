from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_channel
from ghoshell_container import Container
import pytest


@pytest.mark.asyncio
async def test_channel_builder_with_logger():
    from ghoshell_moss.contracts.logger import LoggerItf
    import logging

    chan = new_channel(name="test")
    _logger: LoggerItf | None = None

    _created = logging.getLogger()

    @chan.build.startup
    async def startup():
        nonlocal _logger
        _logger = CommandUtil.logger()

    async with chan.bootstrap():
        # 初始化时一定能返回一个 logger.
        assert _logger is not None
        # 并不是我们手动创建的实例.
        assert _logger is not _created

    container = Container(name="test")
    container.set(LoggerItf, _created)

    async with chan.bootstrap(container=container):
        # 初始化时一定能返回一个 logger.
        assert _logger is not None
        # 拿到的应该是 ioc 绑定的对象.
        assert _logger is _created

