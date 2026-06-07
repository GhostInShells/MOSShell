from ghoshell_moss.depends import depend_zenoh

depend_zenoh()

import zenoh
import threading
import time


def test_session_connection():
    """验证是否能成功建立 Session"""
    with zenoh.open(zenoh.Config()) as session:
        assert session.is_closed() is False
        assert str(session.zid())


def test_put_and_subscribe():
    conf = zenoh.Config()
    key_expr = "demo/example/pubsub"
    expected_value = "Instant Message"
    received_data = []
    msg_event = threading.Event()
    started = threading.Event()

    with zenoh.open(conf) as session:
        def subscribe():
            sub: zenoh.Subscriber = session.declare_subscriber(key_expr)
            started.set()
            with sub:
                for sample in sub:
                    received_data.append(sample)
                    msg_event.set()
                    break

        st = threading.Thread(target=subscribe)
        st.start()
        started.wait()
        for _ in range(10):
            session.put(key_expr, expected_value)
            if msg_event.is_set():
                break
            time.sleep(0.01)
        # 3. 等待接收
        assert msg_event.wait(timeout=0.5)
        assert len(received_data) == 1
        assert received_data[0].payload.to_string() == expected_value


def test_session_lifecycle():
    sub: zenoh.Subscriber | None = None
    with zenoh.open(zenoh.Config()) as session:
        sub: zenoh.Subscriber = session.declare_subscriber("demo/example")

    res = []
    for response in sub:
        res.append(response)
    assert len(res) == 0


def test_sub_close_test():
    sub: zenoh.Subscriber | None = None
    responses = []
    errors = []
    with zenoh.open(zenoh.Config()) as session:
        sub: zenoh.Subscriber = session.declare_subscriber("demo/example")

        broker = threading.Event()

        def run_sub():
            try:
                for res in sub:
                    responses.append(res)
            except zenoh.ZError as e:
                errors.append(e)
            finally:
                broker.set()

        def run_pub():
            for _ in range(10):
                if broker.is_set():
                    break
                session.put("demo/example", "hello")
                time.sleep(0.01)

        st = threading.Thread(target=run_sub)
        pt = threading.Thread(target=run_pub)
        # sub undeclare 可以直接退出. iter 挺好用的.
        sub.undeclare()
        st.start()
        pt.start()
        st.join()
        pt.join()
    assert len(responses) == 0
    assert len(errors) == 1


def test_sub_after_session_quit():
    with zenoh.open(zenoh.Config()) as session:
        sub: zenoh.Subscriber = session.declare_subscriber("demo/example")
    responses = []
    for res in sub:
        responses.append(res)
    assert len(responses) == 0


def test_liveness_tokens_baseline():
    with zenoh.open(zenoh.Config()) as session:
        received_liveness_done = threading.Event()
        key_expr = "demo/example/foo.bar"
        heartbeats = []
        heartbeat_failed = []

        def declare_liveness():
            """生成 liveness"""
            token = session.liveliness().declare_token(key_expr)
            received_liveness_done.wait()
            token.undeclare()

        def check_liveness():
            try:
                while True:
                    alive = session.liveliness().get(key_expr)
                    for r in alive:
                        if r.ok:
                            heartbeats.append(r)
                        else:
                            heartbeat_failed.append(r)
                    if len(heartbeats) == 10:
                        break
                    time.sleep(0.01)
            except Exception as e:
                err = e
            finally:
                received_liveness_done.set()

        node_announce = threading.Thread(target=declare_liveness)
        node_checker = threading.Thread(target=check_liveness)
        node_announce.start()
        node_checker.start()
        node_announce.join()
        node_checker.join()
        assert received_liveness_done.is_set()
        assert len(heartbeats) == 10


def test_liveness_tokens_failed():
    with zenoh.open(zenoh.Config()) as session:
        key_expr = "demo/example/foo.bar"
        heartbeats = []
        heartbeat_failed = []
        err = None

        def check_liveness():
            nonlocal err
            try:
                count = 0
                while count < 10:
                    alive = session.liveliness().get(key_expr, timeout=0.03)
                    success = False
                    for r in alive:
                        if r.ok:
                            success = True
                    if success:
                        heartbeats.append(success)
                    else:
                        heartbeat_failed.append(success)
                    count += 1
                    time.sleep(0.01)
            except Exception as e:
                err = e

        node_checker = threading.Thread(target=check_liveness)
        node_checker.start()
        node_checker.join()
        assert err is None
        assert len(heartbeat_failed) == 10


# === queryable-based cell discovery canary tests ===
# 验证 Zenoh queryable-per-cell + wildcard get 组合是否能支撑 cell 动态发现。
# 核心模式: 每个 cell 声明 queryable (handler 闭包带 address)，
# main cell 用 wildcard get (target=ALL, consolidation=NONE) 聚合全部。
# 这些是第三方 API 金丝雀 — Zenoh 升级时早于业务代码暴露 breakage。

import json


def test_queryable_baseline():
    """queryable 基本模式: declare → query → reply."""
    with zenoh.open(zenoh.Config()) as session:
        def handler(query: zenoh.Query):
            query.reply(query.key_expr, "pong")

        q = session.declare_queryable("test/ping", handler)
        try:
            replies = list(session.get("test/ping"))
            assert len(replies) == 1
            assert replies[0].ok is not None
            assert replies[0].ok.payload.to_string() == "pong"
        finally:
            q.undeclare()


def test_cell_queryable_wildcard_get():
    """cell 声明 queryable，wildcard get 查到该 cell 的 info。"""
    prefix = "test/cells"
    with zenoh.open(zenoh.Config()) as session:
        def cell_handler(query: zenoh.Query):
            query.reply(query.key_expr, json.dumps({
                "address": "host/default",
                "type": "host",
            }))

        q = session.declare_queryable(f"{prefix}/host/default", cell_handler)
        try:
            import time
            time.sleep(0.03)
            replies = list(session.get(
                f"{prefix}/**",
                target=zenoh.QueryTarget.ALL,
                consolidation=zenoh.QueryConsolidation(zenoh.ConsolidationMode.NONE),
            ))
            payloads = [json.loads(r.ok.payload.to_string()) for r in replies if r.ok is not None]
            assert len(payloads) >= 1
            cells = {p["address"]: p for p in payloads}
            assert "host/default" in cells
            assert cells["host/default"]["type"] == "host"
        finally:
            q.undeclare()


def test_cell_leave_undeclare_queryable():
    """cell undeclare queryable 后 wildcard get 不再返回该 cell。"""
    prefix = "test/cells"
    with zenoh.open(zenoh.Config()) as session:
        def cell_handler(query: zenoh.Query):
            query.reply(query.key_expr, json.dumps({"address": "app/to_leave"}))

        q = session.declare_queryable(f"{prefix}/app/to_leave", cell_handler)
        import time
        time.sleep(0.03)

        # 确认存在
        replies = list(session.get(
            f"{prefix}/**",
            target=zenoh.QueryTarget.ALL,
            consolidation=zenoh.QueryConsolidation(zenoh.ConsolidationMode.NONE),
        ))
        payloads = [json.loads(r.ok.payload.to_string()) for r in replies if r.ok is not None]
        assert any(p["address"] == "app/to_leave" for p in payloads)

        # 离开
        q.undeclare()
        time.sleep(0.05)

        # 确认消失
        replies = list(session.get(
            f"{prefix}/**",
            target=zenoh.QueryTarget.ALL,
            consolidation=zenoh.QueryConsolidation(zenoh.ConsolidationMode.NONE),
        ))
        payloads = [json.loads(r.ok.payload.to_string()) for r in replies if r.ok is not None]
        assert not any(p["address"] == "app/to_leave" for p in payloads)


def test_multiple_cells_wildcard_get():
    """多个 cell 声明 queryable，wildcard get 全部发现。"""
    prefix = "test/cells"
    with zenoh.open(zenoh.Config()) as session:
        def make_handler(addr: str, typ: str):
            def h(query: zenoh.Query):
                query.reply(query.key_expr, json.dumps({"address": addr, "type": typ}))
            return h

        qs = []
        cells_in = [
            ("host/default", "host"),
            ("app/echo", "app"),
            ("app/vision", "app"),
            ("script/abc123", "script"),
        ]
        for addr, typ in cells_in:
            qs.append(session.declare_queryable(f"{prefix}/{addr}", make_handler(addr, typ)))

        try:
            import time
            time.sleep(0.05)
            replies = list(session.get(
                f"{prefix}/**",
                target=zenoh.QueryTarget.ALL,
                consolidation=zenoh.QueryConsolidation(zenoh.ConsolidationMode.NONE),
            ))
            payloads = [json.loads(r.ok.payload.to_string()) for r in replies if r.ok is not None]
            cells = {p["address"]: p for p in payloads}
            for addr, typ in cells_in:
                assert addr in cells, f"{addr} not discovered"
                assert cells[addr]["type"] == typ
        finally:
            for q in qs:
                q.undeclare()


def test_dynamic_join_and_leave():
    """cell 动态加入/离开 (declare/undeclare queryable)，wildcard get 实时反映。"""
    prefix = "test/cells_dyn"
    with zenoh.open(zenoh.Config()) as session:
        def make_handler(addr: str):
            def h(query: zenoh.Query):
                query.reply(query.key_expr, json.dumps({"address": addr}))
            return h

        import time

        # 初始无 cell
        replies = list(session.get(
            f"{prefix}/**",
            target=zenoh.QueryTarget.ALL,
            consolidation=zenoh.QueryConsolidation(zenoh.ConsolidationMode.NONE),
        ))
        payloads = [json.loads(r.ok.payload.to_string()) for r in replies if r.ok is not None]
        assert len(payloads) == 0

        # 动态加入
        q = session.declare_queryable(f"{prefix}/app/newcomer", make_handler("app/newcomer"))
        time.sleep(0.05)
        replies = list(session.get(
            f"{prefix}/**",
            target=zenoh.QueryTarget.ALL,
            consolidation=zenoh.QueryConsolidation(zenoh.ConsolidationMode.NONE),
        ))
        payloads = [json.loads(r.ok.payload.to_string()) for r in replies if r.ok is not None]
        assert any(p["address"] == "app/newcomer" for p in payloads)

        # 动态离开
        q.undeclare()
        time.sleep(0.05)
        replies = list(session.get(
            f"{prefix}/**",
            target=zenoh.QueryTarget.ALL,
            consolidation=zenoh.QueryConsolidation(zenoh.ConsolidationMode.NONE),
        ))
        payloads = [json.loads(r.ok.payload.to_string()) for r in replies if r.ok is not None]
        assert not any(p["address"] == "app/newcomer" for p in payloads)
