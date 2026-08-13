from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_channel
from ghoshell_moss.core.blueprint.states_channel import (
    new_prime_channel, new_channel_from_state, new_channel_state,
)
from ghoshell_container import Container
from ghoshell_moss.contracts.logger import LoggerItf
import pytest
import logging


@pytest.mark.asyncio
async def test_states_channel_with_logger():
    chan = new_prime_channel(name="test")
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


@pytest.mark.asyncio
async def test_main_state_channel_startup():
    main_state = new_channel_state(name="main")
    _logger: LoggerItf | None = None
    _created = logging.getLogger()

    @main_state.startup
    async def startup():
        nonlocal _logger
        _logger = CommandUtil.logger()

    container = Container(name="test")
    container.set(LoggerItf, _created)

    channel = new_channel_from_state(main_state)
    async with channel.bootstrap(container=container):
        assert _logger is not None
        assert _logger is _created
