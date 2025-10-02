from ghoshell_moss.shell.runtime import ChannelRuntimeImpl
from ghoshell_container import Container
from ghoshell_moss.channels import PyChannel
import pytest


@pytest.mark.asyncio
async def test_channel_runtime_impl_baseline():

    async def foo() -> int:
        return 123

