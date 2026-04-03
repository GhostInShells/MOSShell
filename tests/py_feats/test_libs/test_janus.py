import janus


def test_janus_empty():
    queue = janus.Queue()
    queue.sync_q.put_nowait(1)
    assert not queue.sync_q.empty()
    assert not queue.async_q.empty()

    assert queue.sync_q.get_nowait() == 1
    assert queue.sync_q.empty()
    assert queue.async_q.empty()
