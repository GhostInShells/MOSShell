"""CellDiscovery 单测 — queryable-based cell discovery。"""
import json
import time

from ghoshell_moss.depends import depend_zenoh

depend_zenoh()

import zenoh
from ghoshell_moss.host.cell_discovery import CellDiscovery

SCOPE = "test_scope"


# --- key expressions (pure unit, no zenoh) ---


def test_cell_prefix():
    cd = CellDiscovery(session_scope=SCOPE)
    assert cd.cell_prefix() == f"MOSS/{SCOPE}/cells"


def test_cell_key():
    cd = CellDiscovery(session_scope=SCOPE)
    addr = "app/group/name"
    assert cd.cell_key(addr) == f"MOSS/{SCOPE}/cells/{addr}"


def test_query_portal_key():
    cd = CellDiscovery(session_scope=SCOPE)
    assert cd.query_portal_key() == f"MOSS/{SCOPE}/cells/query"


# --- announce_cell (needs zenoh) ---


def test_announce_cell():
    """announce_cell 声明 queryable，wildcard get 可查询到。"""
    cd = CellDiscovery(session_scope=SCOPE)
    address = "host/default"
    info = {"address": address, "type": "host", "name": "default"}

    with zenoh.open(zenoh.Config()) as session:
        with cd.announce_cell(session, address, info):
            time.sleep(0.03)
            replies = list(session.get(
                f"{cd.cell_prefix()}/**",
                target=zenoh.QueryTarget.ALL,
                consolidation=zenoh.QueryConsolidation(zenoh.ConsolidationMode.NONE),
            ))
            payloads = [json.loads(r.ok.payload.to_string()) for r in replies if r.ok is not None]
            assert any(p["address"] == address for p in payloads)

        # 退出 context manager 后 queryable 被 undeclare
        time.sleep(0.03)
        replies = list(session.get(
            f"{cd.cell_prefix()}/**",
            target=zenoh.QueryTarget.ALL,
            consolidation=zenoh.QueryConsolidation(zenoh.ConsolidationMode.NONE),
        ))
        payloads = [json.loads(r.ok.payload.to_string()) for r in replies if r.ok is not None]
        assert not any(p["address"] == address for p in payloads)


# --- serve_query_portal + query_cells (needs zenoh) ---


def test_query_cells_discovers_announced_cells():
    """query_cells 返回所有已宣告 cell 的 info。"""
    cd = CellDiscovery(session_scope=SCOPE)

    with zenoh.open(zenoh.Config()) as session:
        with cd.announce_cell(session, "host/default", {"address": "host/default", "type": "host"}):
            with cd.announce_cell(session, "app/echo", {"address": "app/echo", "type": "app"}):
                time.sleep(0.05)
                result = cd.query_cells(session)
                assert "host/default" in result
                assert "app/echo" in result
                assert result["host/default"]["type"] == "host"


def test_query_cells_empty_when_no_cells():
    """没有任何 cell 宣告时 query_cells 返回空 dict。"""
    cd = CellDiscovery(session_scope=SCOPE)

    with zenoh.open(zenoh.Config()) as session:
        result = cd.query_cells(session)
        assert result == {}


def test_query_cells_dynamic_leave():
    """cell undeclare 后 query_cells 不再返回该 cell。"""
    cd = CellDiscovery(session_scope=SCOPE)
    address = "app/to_leave"

    with zenoh.open(zenoh.Config()) as session:
        with cd.announce_cell(session, address, {"address": address}):
            time.sleep(0.03)
            result = cd.query_cells(session)
            assert address in result

        # undeclare 后消失
        time.sleep(0.05)
        result = cd.query_cells(session)
        assert address not in result


def test_multiple_cells_discovered():
    """多个 cell 同时宣告，全部被发现。"""
    cd = CellDiscovery(session_scope=SCOPE)
    addresses = [f"app/test/cell_{i}" for i in range(5)]

    with zenoh.open(zenoh.Config()) as session:
        # 用 exit_stack 管理多个 context manager
        import contextlib
        with contextlib.ExitStack() as stack:
            for addr in addresses:
                stack.enter_context(
                    cd.announce_cell(session, addr, {"address": addr, "type": "app"})
                )
            time.sleep(0.06)
            result = cd.query_cells(session)
            for addr in addresses:
                assert addr in result, f"{addr} not discovered"


def test_serve_query_portal():
    """serve_query_portal 返回 cells_provider 提供的缓存数据。"""
    cd = CellDiscovery(session_scope=SCOPE)

    cache = {
        "host/default": {"address": "host/default", "type": "host"},
        "app/echo": {"address": "app/echo", "type": "app"},
    }

    with zenoh.open(zenoh.Config()) as session:
        with cd.serve_query_portal(session, lambda: cache):
            time.sleep(0.03)
            replies = list(session.get(cd.query_portal_key()))
            assert len(replies) == 1
            result = json.loads(replies[0].ok.payload.to_string())
            assert "host/default" in result
            assert "app/echo" in result
            assert result["host/default"]["type"] == "host"
