"""CellDiscovery 单测 — key 表达式 + Zenoh liveness 集成。"""
import threading

from ghoshell_moss.depends import depend_zenoh

depend_zenoh()

import zenoh
from ghoshell_moss.host.cell_discovery import CellDiscovery

SCOPE = "test_scope"


# --- key expressions (pure unit, no zenoh) ---


def test_key_prefix():
    cd = CellDiscovery(session_scope=SCOPE)
    assert cd.liveness_prefix() == f"MOSS/{SCOPE}/cell/liveness"


def test_key_single_cell():
    cd = CellDiscovery(session_scope=SCOPE)
    addr = "app/group/name"
    assert cd.liveness_key(addr) == f"MOSS/{SCOPE}/cell/liveness/{addr}"


def test_key_wildcard():
    cd = CellDiscovery(session_scope=SCOPE)
    assert cd.liveness_wildcard() == f"MOSS/{SCOPE}/cell/liveness/**"


# --- declare_this_cell (needs zenoh) ---


def test_declare_this_cell():
    """declare_this_cell 声明 liveness token，wildcard 可查询到。"""
    cd = CellDiscovery(session_scope=SCOPE)
    address = "host/default"

    with zenoh.open(zenoh.Config()) as session:
        with cd.declare_this_cell(session, address):
            # wildcard 查询应看到此 token
            samples = list(session.liveliness().get(cd.liveness_wildcard()))
            ok_keys = {str(s.ok.key_expr) for s in samples if s.ok is not None}
            assert cd.liveness_key(address) in ok_keys

        # 退出 context manager 后 token 被 undeclare
        samples = list(session.liveliness().get(cd.liveness_wildcard()))
        ok_keys = {str(s.ok.key_expr) for s in samples if s.ok is not None}
        assert cd.liveness_key(address) not in ok_keys


# --- discover_cells (needs zenoh) ---


def test_discover_cells_detects_live_cells():
    """discover_cells 监听已知 cell，另一 session 声明 token 后 event 被 set。"""
    cd = CellDiscovery(session_scope=SCOPE)
    address = "app/test/cell"
    this = "host/default"

    cells = {
        this: None,  # value unused by discover_cells
        address: None,
    }
    events = {
        this: threading.Event(),
        address: threading.Event(),
    }

    with zenoh.open(zenoh.Config()) as session:
        # 初始: 只有 this 被 set
        with cd.discover_cells(session, cells, events, this_address=this):
            assert events[this].is_set()
            # remote cell 还没上线
            assert not events[address].is_set()

            # 另一个 session 声明 liveness token
            with zenoh.open(zenoh.Config()) as remote:
                token = remote.liveliness().declare_token(cd.liveness_key(address))
                # 给 zenoh 一点时间传播 liveness
                import time
                for _ in range(50):
                    if events[address].is_set():
                        break
                    time.sleep(0.01)
                assert events[address].is_set(), "remote cell liveness not detected"
                token.undeclare()


def test_discover_cells_detects_disconnect():
    """cell 下线 (DELETE) 后 event 被 clear。"""
    cd = CellDiscovery(session_scope=SCOPE)
    address = "app/test/disconnect"
    this = "host/default"

    cells = {this: None, address: None}
    events = {this: threading.Event(), address: threading.Event()}

    with zenoh.open(zenoh.Config()) as session:
        with cd.discover_cells(session, cells, events, this_address=this):
            # 让 cell 上线
            with zenoh.open(zenoh.Config()) as remote:
                token = remote.liveliness().declare_token(cd.liveness_key(address))
                import time
                for _ in range(50):
                    if events[address].is_set():
                        break
                    time.sleep(0.01)
                assert events[address].is_set()

                token.undeclare()
                # 等 DELETE 传播
                for _ in range(50):
                    if not events[address].is_set():
                        break
                    time.sleep(0.01)
                assert not events[address].is_set(), "disconnected cell still marked alive"


def test_discover_cells_initial_query():
    """discover_cells 启动时做 wildcard 查询，发现已存在的 cell。"""
    cd = CellDiscovery(session_scope=SCOPE)
    address = "app/already/running"
    this = "host/default"

    cells = {this: None, address: None}
    events = {this: threading.Event(), address: threading.Event()}

    # 先在一个 session 里声明 token（模拟已运行的 cell）
    with zenoh.open(zenoh.Config()) as remote:
        token = remote.liveliness().declare_token(cd.liveness_key(address))

        # 另一个 session 做 discover — initial query 应发现已存在的 token
        with zenoh.open(zenoh.Config()) as session:
            with cd.discover_cells(session, cells, events, this_address=this):
                import time
                for _ in range(50):
                    if events[address].is_set():
                        break
                    time.sleep(0.01)
                assert events[address].is_set(), "initial query missed already-running cell"

        token.undeclare()


def test_discover_cells_skips_unknown_address():
    """initial query 发现的 key 如果不在 alive_events 里，不报错。"""
    cd = CellDiscovery(session_scope=SCOPE)
    this = "host/default"
    cells = {this: None}
    events = {this: threading.Event()}

    with zenoh.open(zenoh.Config()) as remote:
        # 声明一个不在 cells 里的 token
        unknown = "app/unknown/cell"
        token = remote.liveliness().declare_token(cd.liveness_key(unknown))

        with zenoh.open(zenoh.Config()) as session:
            # 不应报错 — unknown address 被静默跳过
            with cd.discover_cells(session, cells, events, this_address=this):
                pass

        token.undeclare()


def test_multiple_cells():
    """多个 cell 同时上线，全部被检测到。"""
    cd = CellDiscovery(session_scope=SCOPE)
    this = "host/default"
    addresses = [f"app/test/cell_{i}" for i in range(5)]

    cells = {this: None, **{a: None for a in addresses}}
    events = {this: threading.Event(), **{a: threading.Event() for a in addresses}}

    with zenoh.open(zenoh.Config()) as session:
        with cd.discover_cells(session, cells, events, this_address=this):
            with zenoh.open(zenoh.Config()) as remote:
                tokens = [
                    remote.liveliness().declare_token(cd.liveness_key(a))
                    for a in addresses
                ]
                import time
                for _ in range(100):
                    if all(events[a].is_set() for a in addresses):
                        break
                    time.sleep(0.01)

                for a in addresses:
                    assert events[a].is_set(), f"{a} not detected"

                for t in tokens:
                    t.undeclare()
