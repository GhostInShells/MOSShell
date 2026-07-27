from ghoshell_moss.depends import depend_matrix

depend_matrix()

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


# === liveness token wildcard canary tests ===
# 验证 liveness token 是否支持 * (单层) / ** (多层) wildcard.
# 这决定 cell 发现能否用 liveness token 按 type 层级过滤.


def test_liveness_wildcard_double_star():
    """liveness token + ** wildcard: declare_token(cells/a/b/c), get(cells/**)."""
    prefix = "test/liveness_wc"
    with zenoh.open(zenoh.Config()) as session:
        tok = session.liveliness().declare_token(f"{prefix}/a/b/c")
        try:
            import time
            time.sleep(0.05)
            replies = list(session.liveliness().get(f"{prefix}/**"))
            keys = [str(r.result.key_expr) for r in replies if r.ok]
            assert f"{prefix}/a/b/c" in keys
        finally:
            tok.undeclare()


def test_liveness_wildcard_single_star():
    """liveness token + * wildcard: declare_token(x/host/main), get(x/host/*)."""
    prefix = "test/liveness_s"
    with zenoh.open(zenoh.Config()) as session:
        tok = session.liveliness().declare_token(f"{prefix}/host/main")
        try:
            import time
            time.sleep(0.05)
            replies = list(session.liveliness().get(f"{prefix}/host/*"))
            keys = [str(r.result.key_expr) for r in replies if r.ok]
            assert f"{prefix}/host/main" in keys
        finally:
            tok.undeclare()


def test_liveness_wildcard_star_vs_double_star_scope():
    """* 只匹配单层: declare_token(x/a/b), get(x/*) 应匹配 a, get(x/a/*) 应匹配 b."""
    prefix = "test/liveness_x"
    with zenoh.open(zenoh.Config()) as session:
        tok_a = session.liveliness().declare_token(f"{prefix}/a/x")
        tok_b = session.liveliness().declare_token(f"{prefix}/b/x")
        try:
            import time
            time.sleep(0.05)

            # get(x/*/x) 应该匹配 a/x 和 b/x 两层
            replies = list(session.liveliness().get(f"{prefix}/*/x"))
            keys = [str(r.result.key_expr) for r in replies if r.ok]
            assert f"{prefix}/a/x" in keys
            assert f"{prefix}/b/x" in keys

            # get(x/*) 不应该匹配到 a/x (那是两层深度)
            replies_single = list(session.liveliness().get(f"{prefix}/*"))
            keys_single = [str(r.result.key_expr) for r in replies_single if r.ok]
            assert f"{prefix}/a/x" not in keys_single, "* should match only 1 segment"
        finally:
            tok_a.undeclare()
            tok_b.undeclare()


def test_liveness_subscribe_wildcard():
    """subscribe liveness with ** wildcard: 监听到 token 的 PUT 和 DELETE."""
    prefix = "test/liveness_sub"
    with zenoh.open(zenoh.Config()) as session:
        received_put = []
        received_delete = []
        started = threading.Event()
        done = threading.Event()

        def _listen():
            sub = session.liveliness().declare_subscriber(f"{prefix}/**")
            started.set()
            with sub:
                for sample in sub:
                    if sample.kind == zenoh.SampleKind.PUT:
                        received_put.append(str(sample.key_expr))
                    elif sample.kind == zenoh.SampleKind.DELETE:
                        received_delete.append(str(sample.key_expr))
                    if received_delete:
                        break

        t = threading.Thread(target=_listen)
        t.start()
        started.wait()
        import time
        time.sleep(0.05)

        tok = session.liveliness().declare_token(f"{prefix}/type/name")
        time.sleep(0.1)
        tok.undeclare()
        time.sleep(0.1)

        done.set()
        t.join(timeout=2)

        assert f"{prefix}/type/name" in received_put
        assert f"{prefix}/type/name" in received_delete


def test_liveness_and_queryable_same_key():
    """同一个 key expression 同时挂 liveness token + queryable，互不干扰."""
    key = "test/duplex/cell"
    with zenoh.open(zenoh.Config()) as session:
        # 1. declare liveness token
        tok = session.liveliness().declare_token(key)

        # 2. declare queryable on same key
        def handler(query: zenoh.Query):
            query.reply(query.key_expr, json.dumps({"status": "alive"}))

        q = session.declare_queryable(key, handler)

        try:
            import time
            time.sleep(0.05)

            # liveness get 能找到
            live_replies = list(session.liveliness().get(key))
            live_keys = [str(r.result.key_expr) for r in live_replies if r.ok]
            assert key in live_keys

            # queryable get 能找到
            data_replies = list(session.get(key))
            data = [json.loads(r.ok.payload.to_string()) for r in data_replies if r.ok is not None]
            assert len(data) == 1
            assert data[0]["status"] == "alive"
        finally:
            q.undeclare()
            tok.undeclare()
