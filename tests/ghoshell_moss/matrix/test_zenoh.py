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
