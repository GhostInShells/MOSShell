"""Parameters 的 zenoh transport 测试 — 单测 (key 映射) + 集成 (真 session)."""

import asyncio

import pytest
import zenoh

from ghoshell_moss.depends import depend_matrix

depend_matrix()

from ghoshell_moss.core.blueprint.parameter import ExampleParameter
from ghoshell_moss.core.parameter import TruthHostParameters, WorkerParameters
from ghoshell_moss.matrix.zenoh_helper import MatrixNamespace
from ghoshell_moss.matrix.parameters import ParameterNamespace, ZenohParametersBroadcaster


def test_parameter_namespace_key_groups():
    # key 组全部由 namespace 派生, 无魔法值.
    ns = ParameterNamespace(MatrixNamespace("test_scope"))
    base = "MOSS/matrix/scopes/test_scope/parameters"
    assert ns.truth_key("ghost") == f"{base}/host/truth/ghost"
    assert ns.declaration_key("ghost") == f"{base}/worker/declaration/ghost"
    assert ns.declaration_wildcard() == f"{base}/worker/declaration/**"
    assert ns.liveness_key("host/name/uid") == f"{base}/host/liveness/host/name/uid"


@pytest.mark.asyncio
async def test_host_declare_worker_subscribe_converges():
    session = zenoh.open(zenoh.Config())
    ns = MatrixNamespace("test_scope_converge")
    try:
        host = TruthHostParameters("host/name/uid1", ZenohParametersBroadcaster(session, ns))
        worker = WorkerParameters("worker/name/uid2", ZenohParametersBroadcaster(session, ns))
        async with host, worker:
            decl = await host.declare(ExampleParameter(example="calm"))
            sub = await worker.subscribe(ExampleParameter)
            await asyncio.sleep(0.3)

            assert sub.value is not None
            assert sub.value.example == "calm"

            seen = []
            sub.on_change(lambda v: seen.append(v.example))
            decl.set(ExampleParameter(example="excited"))
            await asyncio.sleep(0.3)
            assert seen == ["excited"]

            sub.close()
            decl.set(ExampleParameter(example="third"))
            await asyncio.sleep(0.3)
            assert seen == ["excited"]  # 退订后不再收
    finally:
        session.close()


@pytest.mark.asyncio
async def test_host_restart_over_zenoh_extends_value_with_new_epoch():
    """跨 session 的 host 重启: 远程 worker 不被清理, 新化身续值、版本从头算.

    与 memory 参考测试同语义, 这里走 zenoh liveness:
      host1 online → worker 重播 → host1 采纳 (v1)
      host1 offline (token 撤销) → host2 online (新 address) → worker 重播 → host2 采纳
    """
    session = zenoh.open(zenoh.Config())
    ns = MatrixNamespace("test_scope_restart")
    try:
        worker = WorkerParameters("worker/name/uid2", ZenohParametersBroadcaster(session, ns))
        async with worker:
            decl = await worker.declare(ExampleParameter(example="v1"))
            await asyncio.sleep(0.3)
            assert decl.value.example == "v1"  # 无 host, 本地即真相

            host1 = TruthHostParameters("host/name/h1", ZenohParametersBroadcaster(session, ns))
            async with host1:
                await asyncio.sleep(0.3)  # host1 online → worker 重播 → host1 采纳
                truth1 = decl.get_truth_data()
                assert truth1 is not None
                assert truth1.epoch == "host/name/h1"
                assert truth1.version == 1
                assert decl.value.example == "v1"

                decl.set(ExampleParameter(example="v2"))
                await asyncio.sleep(0.3)
                assert decl.get_truth_data().version == 2
                assert decl.value.example == "v2"

            # host 重启 (新化身 = 新 address): worker 仍在, 持有 v2.
            host2 = TruthHostParameters("host/name/h2", ZenohParametersBroadcaster(session, ns))
            async with host2:
                await asyncio.sleep(0.3)
                truth2 = decl.get_truth_data()
                assert truth2 is not None
                assert truth2.epoch == "host/name/h2"  # 新化身
                assert decl.value.example == "v2"       # 值延续
                assert truth2.version == 1              # 版本从头算
    finally:
        session.close()
