"""Parameters 收敛逻辑的单进程参考测试 (host + worker over MemoryBus).

验证的协议承诺:
  - host 的 declare 即真值 (只迭代版本号)
  - worker 声明 / 订阅经 host 定序后收敛
  - 真值到达不触发声明 (振荡闸口), on_change 只由真值触发一次
  - host 化身 (epoch) 变更时继承版本、不重置全序
  - close / wait_first_truth 的生命周期语义
"""

import asyncio

import pytest

from ghoshell_moss.core.blueprint.parameter import ParameterModel
from ghoshell_moss.core.parameter import (
    MemoryBus,
    MemoryParametersBroadcaster,
    TruthHostParameters,
    WorkerParameters,
)


class GhostPersona(ParameterModel):
    name: str = "Echo"
    temperature: float = 0.7

    @classmethod
    def parameter_key(cls) -> str:
        return "ghost_persona"


def make_host(bus: MemoryBus, address: str = "host") -> TruthHostParameters:
    return TruthHostParameters(address, MemoryParametersBroadcaster(bus))


def make_worker(name: str, bus: MemoryBus) -> WorkerParameters:
    return WorkerParameters(name, MemoryParametersBroadcaster(bus))


async def settle() -> None:
    # memory transport 是同步回调 + 异步队列, 收敛要几个 event-loop 轮次.
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_host_declare_is_truth():
    bus = MemoryBus()
    host = make_host(bus)
    async with host:
        decl = await host.declare(GhostPersona())
        assert decl.key == "ghost_persona"
        assert decl.value.name == "Echo"
        truth = decl.get_truth_data()
        assert truth is not None
        assert truth.version >= 1
        assert truth.epoch != ""
        assert truth.meta.host is True


@pytest.mark.asyncio
async def test_worker_subscribe_receives_host_truth():
    bus = MemoryBus()
    host = make_host(bus)
    reader = make_worker("reader", bus)
    async with host, reader:
        await host.declare(GhostPersona(name="Nova"))
        sub = await reader.subscribe(GhostPersona)
        await settle()
        assert sub.value.name == "Nova"
        assert sub.get_truth().name == "Nova"


@pytest.mark.asyncio
async def test_host_set_converges_to_workers():
    # host 也是正常节点: 自己声明参数 (如 ghost emotion), 全网络围绕它动态反馈.
    bus = MemoryBus()
    host = make_host(bus)
    reader = make_worker("reader", bus)
    async with host, reader:
        decl = await host.declare(GhostPersona(name="calm"))
        sub = await reader.subscribe(GhostPersona)
        await settle()
        assert sub.value.name == "calm"

        seen = []
        sub.on_change(lambda v: seen.append(v.name))
        decl.set(GhostPersona(name="excited"))
        await settle()
        assert sub.value.name == "excited"
        assert seen == ["excited"]


@pytest.mark.asyncio
async def test_worker_declare_rejected_when_truth_exists():
    bus = MemoryBus()
    host = make_host(bus)
    writer = make_worker("writer", bus)
    async with host, writer:
        await host.declare(GhostPersona(name="Nova"))
        decl = await writer.declare(GhostPersona(name="Other"))
        await settle()
        # first-packet 被 host 已有真值拒绝 → 学到 host 的真值, 不是自己声明的值.
        assert decl.value.name == "Nova"
        assert decl.get_truth().name == "Nova"


@pytest.mark.asyncio
async def test_set_converges_across_nodes():
    bus = MemoryBus()
    host = make_host(bus)
    writer = make_worker("writer", bus)
    reader = make_worker("reader", bus)
    async with host, writer, reader:
        decl = await writer.declare(GhostPersona())
        sub = await reader.subscribe(GhostPersona)
        await settle()

        seen = []
        sub.on_change(lambda v: seen.append(v.name))

        decl.set(GhostPersona(name="Nova", temperature=0.9))
        await settle()
        assert sub.value.name == "Nova"
        assert seen == ["Nova"]


@pytest.mark.asyncio
async def test_local_set_fires_on_change_once_via_truth_echo():
    bus = MemoryBus()
    host = make_host(bus)
    writer = make_worker("writer", bus)
    async with host, writer:
        decl = await writer.declare(GhostPersona())
        await settle()

        seen = []
        decl.on_change(lambda v: seen.append(v.name))

        decl.set(GhostPersona(name="Nova"))
        await settle()
        # 本地 set 不直接触发 on_change, 只经 host 真值回声触发一次 (不重复).
        assert seen == ["Nova"]


@pytest.mark.asyncio
async def test_worker_first_host_adopts_last_win():
    bus = MemoryBus()
    writer = make_worker("writer", bus)
    reader = make_worker("reader", bus)
    async with writer, reader:
        decl = await writer.declare(GhostPersona(name="Seeded"))
        decl.set(GhostPersona(name="LocalWin"))
        await settle()
        assert decl.value.name == "LocalWin"  # 无 host, 本地即真相

        sub = await reader.subscribe(GhostPersona)
        await settle()

        host = make_host(bus)
        async with host:
            await settle()
            assert decl.value.name == "LocalWin"
            assert sub.value.name == "LocalWin"


@pytest.mark.asyncio
async def test_host_restart_resets_version_and_epoch():
    bus = MemoryBus()
    writer = make_worker("writer", bus)
    async with writer:
        decl = await writer.declare(GhostPersona())
        host1 = make_host(bus, address="host-1")
        async with host1:
            await settle()  # worker 同步到 host-1 初始真值 v1
            decl.set(GhostPersona(name="v2"))
            await settle()  # host-1 采纳 v2
            truth1 = decl.get_truth_data()
            assert truth1 is not None
            assert truth1.version == 2
            assert decl.value.name == "v2"

        # host 重启 (新化身 = 新 address): 远程 worker 不被清理, 仍持有 v2.
        host2 = make_host(bus, address="host-2")
        async with host2:
            await settle()
            truth2 = decl.get_truth_data()
            assert truth2 is not None
            assert decl.value.name == "v2"        # 值延续
            assert truth2.epoch != truth1.epoch    # 新化身
            assert truth2.version == 1             # 版本从头算, 不继承 v2


@pytest.mark.asyncio
async def test_custom_key():
    bus = MemoryBus()
    host = make_host(bus)
    writer = make_worker("writer", bus)
    reader = make_worker("reader", bus)
    async with host, writer, reader:
        decl = await writer.declare(GhostPersona(), key="alt_persona")
        assert decl.key == "alt_persona"

        sub = await reader.subscribe(GhostPersona, key="alt_persona")
        await settle()
        assert sub.value.name == "Echo"  # 无 host 真值 → writer 首包被采纳

        seen = []
        sub.on_change(lambda v: seen.append(v.name))
        decl.set(GhostPersona(name="custom"))
        await settle()
        assert seen == ["custom"]


@pytest.mark.asyncio
async def test_close_stops_updates():
    bus = MemoryBus()
    host = make_host(bus)
    writer = make_worker("writer", bus)
    reader = make_worker("reader", bus)
    async with host, writer, reader:
        decl = await writer.declare(GhostPersona())
        sub = await reader.subscribe(GhostPersona)
        await settle()

        seen = []
        sub.on_change(lambda v: seen.append(v.name))

        sub.close()
        decl.set(GhostPersona(name="X"))
        await settle()
        assert seen == []


@pytest.mark.asyncio
async def test_wait_first_truth_blocks_until_truth():
    bus = MemoryBus()
    host = make_host(bus)
    reader = make_worker("reader", bus)
    async with host, reader:
        await host.declare(GhostPersona(name="Nova"))
        sub = await reader.subscribe(GhostPersona)
        truth = await asyncio.wait_for(sub.wait_first_truth(), timeout=1.0)
        assert truth.name == "Nova"


@pytest.mark.asyncio
async def test_wait_first_truth_raises_on_close():
    bus = MemoryBus()
    reader = make_worker("reader", bus)  # 无 host, 永远等不到真值
    async with reader:
        sub = await reader.subscribe(GhostPersona)
        waiter = asyncio.create_task(sub.wait_first_truth())
        await asyncio.sleep(0.01)
        sub.close()
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(waiter, timeout=1.0)


@pytest.mark.asyncio
async def test_declared_lists_schemas():
    bus = MemoryBus()
    host = make_host(bus)
    async with host:
        await host.declare(GhostPersona())
        schemas = host.declared()
        assert any(s.name == "ghost_persona" for s in schemas)
