import asyncio
import random

import pytest

from ghoshell_moss import CommandError
from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.transports.zmq_channel.zmq_channel import ZMQSocketType, create_zmq_channel


def get_random_port():
    """获取一个随机可用端口"""
    return random.randint(10000, 20000)


@pytest.mark.asyncio
async def test_zmq_channel_baseline():
    """测试 ZMQ channel 的基本功能"""
    # 使用随机端口避免冲突
    port = get_random_port()
    address = f"tcp://127.0.0.1:{port}"

    # 创建 provider 和 proxy
    provider, proxy = create_zmq_channel(
        name="proxy",
        address=address,
        socket_type=ZMQSocketType.PAIR,
    )

    # 创建一个简单的测试 channel
    test_channel = PyChannel(name="provider")

    # 添加一个简单的测试命令
    @test_channel.build.command()
    async def foo(value: int = 42) -> str:
        return f"Received: {value}"

    # 在后台线程中运行 provider
    provider.run_in_thread(test_channel)

    try:
        # 启动 proxy
        async with proxy.bootstrap() as proxy_runtime:
            await proxy_runtime.wait_connected()
            # 验证 proxy 已连接
            assert proxy_runtime.is_running()

            # 获取 channel meta
            meta = proxy_runtime.own_meta()
            assert meta is not None
            assert meta.name == "proxy"
            assert len(meta.commands) == 1
            assert meta.commands[0].name == "foo"

            # 获取命令并执行
            cmd = proxy_runtime.get_command("foo")
            assert cmd is not None

            # 测试命令执行
            result = await cmd(123)
            assert result == "Received: 123"

            # 测试带默认值的调用
            result = await cmd()
            assert result == "Received: 42"

    finally:
        # 确保清理资源
        provider.close()


@pytest.mark.asyncio
async def test_zmq_channel_with_timeout():
    """测试带超时设置的 ZMQ channel"""
    port = get_random_port()
    address = f"tcp://127.0.0.1:{port}"

    # 创建带超时设置的 provider 和 proxy
    provider, proxy = create_zmq_channel(
        name="timeout_channel",
        address=address,
        socket_type=ZMQSocketType.PAIR,
        recv_timeout=2.0,  # 2秒接收超时
        send_timeout=1.0,  # 1秒发送超时
    )

    test_channel = PyChannel(name="timeout_server")

    # 添加一个会延迟响应的命令
    @test_channel.build.command()
    async def delayed_command(delay: float = 0.1) -> str:
        await asyncio.sleep(delay)
        return f"Delayed by {delay}s"

    provider.run_in_thread(test_channel)

    try:
        async with proxy.bootstrap() as runtime:
            await runtime.wait_connected()
            # 测试正常延迟命令
            cmd = runtime.get_command("delayed_command")
            assert cmd is not None
            result = await cmd(0.5)
            assert result == "Delayed by 0.5s"

            # 测试超时命令（应该会超时）
            # 注意：这里我们期望超时，所以应该捕获 TimeoutError
            with pytest.raises(CommandError):
                result = await asyncio.wait_for(cmd(3.0), timeout=0.5)

    finally:
        provider.close()


@pytest.mark.asyncio
async def test_zmq_channel_lost_connection():
    """测试 ZMQ channel 的重连能力"""
    port = get_random_port()
    address = f"tcp://127.0.0.1:{port}"

    # 使用较短的心跳间隔和超时时间，以便测试能快速进行
    provider, proxy = create_zmq_channel(
        name="reconnect_channel",
        address=address,
        socket_type=ZMQSocketType.PAIR,
        heartbeat_interval=0.1,  # 100ms 心跳间隔
        heartbeat_timeout=0.3,  # 300ms 心跳超时
    )

    test_channel = PyChannel(name="reconnect_server")

    @test_channel.build.command()
    async def simple_command() -> str:
        return "Hello from provider"

    # 先启动 provider
    provider.run_in_thread(test_channel)

    # 等待 provider 启动完成
    await asyncio.sleep(0.1)

    # 启动 proxy
    runtime = proxy.bootstrap()
    assert runtime is not None
    assert runtime.container is not None
    async with runtime:
        await runtime.wait_connected()
        # 验证连接正常
        assert runtime.is_running()

        # 执行命令
        cmd = runtime.get_command("simple_command")
        result = await cmd()
        assert result == "Hello from provider"
        result = await cmd()
        assert result == "Hello from provider"

        # 模拟连接中断（通过关闭 server）
        provider.close()
        await asyncio.sleep(0.1)
        assert not provider.is_running()
        with pytest.raises(CommandError):
            await cmd()

        assert not runtime.is_available()


@pytest.mark.asyncio
async def test_zmq_channel_lasy_bind():
    port = get_random_port()
    address = f"tcp://127.0.0.1:{port}"

    # 使用较短的心跳间隔和超时时间，以便测试能快速进行
    provider, proxy = create_zmq_channel(
        name="test",
        address=address,
    )

    provider_channel = PyChannel(name="provider")

    @provider_channel.build.command()
    async def hello() -> str:
        return "Hello"

    async with proxy.bootstrap() as runtime:
        assert not runtime.is_connected()
        assert not runtime.is_available()

        # 启动连接.
        async with provider.arun(provider_channel):
            await runtime.wait_connected()
            assert runtime.is_connected()
            cmd = runtime.get_command("hello")
            assert await cmd() == "Hello"


@pytest.mark.asyncio
async def test_zmq_channel_multiple_commands():
    """测试 ZMQ channel 处理多个命令的能力"""
    port = get_random_port()
    address = f"tcp://127.0.0.1:{port}"

    provider, proxy = create_zmq_channel(
        name="multi_cmd_channel",
        address=address,
        socket_type=ZMQSocketType.PAIR,
    )

    test_channel = PyChannel(name="multi_cmd_server")

    # 添加多个命令
    @test_channel.build.command()
    async def add(a: int, b: int) -> int:
        return a + b

    @test_channel.build.command()
    async def multiply(a: int, b: int) -> int:
        return a * b

    @test_channel.build.command()
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    provider.run_in_thread(test_channel)

    try:
        async with proxy.bootstrap() as runtime:
            await runtime.wait_connected()
            # 验证所有命令都存在
            meta = runtime.own_meta()
            assert len(meta.commands) == 3
            command_names = {cmd.name for cmd in meta.commands}
            assert command_names == {"add", "multiply", "greet"}

            # 测试所有命令
            add_cmd = runtime.get_command("add")
            multiply_cmd = runtime.get_command("multiply")
            greet_cmd = runtime.get_command("greet")

            # 执行加法
            result = await add_cmd(2, 3)
            assert result == 5

            # 执行乘法
            result = await multiply_cmd(4, 5)
            assert result == 20

            # 执行问候
            result = await greet_cmd("World")
            assert result == "Hello, World!"

            # 测试并发命令执行
            tasks = [add_cmd(1, 2), multiply_cmd(3, 4), greet_cmd("Test")]

            results = await asyncio.gather(*tasks)
            assert results == [3, 12, "Hello, Test!"]

    finally:
        provider.close()
