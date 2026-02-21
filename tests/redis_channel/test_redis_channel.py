import pytest
from fakeredis.aioredis import FakeRedis, FakeServer

from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.transports.redis_channel.redis_channel import (
    RedisChannelProvider,
    RedisChannelProxy,
    RedisConnectionConfig,
)


@pytest.mark.asyncio
async def test_redis_channel_baseline():
    """测试 Redis channel 的基本功能"""
    server = FakeServer()
    async with FakeRedis(server=server) as fake_redis:
        to_provider_stream = "to_provider"
        to_proxy_stream = "to_proxy"

        provider = RedisChannelProvider(
            config=RedisConnectionConfig(
                redis=fake_redis,
                write_stream=to_proxy_stream,
                read_stream=to_provider_stream,
            )
        )

        proxy = RedisChannelProxy(
            config=RedisConnectionConfig(
                redis=fake_redis,
                write_stream=to_provider_stream,
                read_stream=to_proxy_stream,
            ),
            name="test_redis_channel",
        )

        # 创建一个简单的测试 channel
        test_channel = PyChannel(name="test_server")

        # 添加一个简单的测试命令
        @test_channel.build.command()
        async def foo(value: int = 42) -> str:
            return f"Received: {value}"

        provider.run_in_thread(test_channel)

        async with provider.arun(test_channel):
            async with proxy.bootstrap() as runtime:
                # 验证 proxy 已连接
                await runtime.wait_connected()
                assert runtime.is_running()

                # 获取 channel meta
                meta = runtime.own_meta()
                assert meta is not None
                assert meta.name == "test_redis_channel"
                assert len(meta.commands) == 1
                assert meta.commands[0].name == "foo"

                # 获取命令并执行
                cmd = runtime.get_command("foo")
                assert cmd is not None

                # 测试命令执行
                result = await cmd(123)
                assert result == "Received: 123"

                # 测试带默认值的调用
                result = await cmd()
                assert result == "Received: 42"
