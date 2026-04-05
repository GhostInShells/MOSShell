import pytest
import os
import time
import multiprocessing
from pathlib import Path
from ghoshell_moss.contracts.workspace import LocalWorkspace, FileLocker


def test_workspace_structure(tmp_path: Path):
    """测试工作空间目录自动创建及结构"""
    ws = LocalWorkspace(tmp_path)

    # 测试目录是否存在
    assert ws.root_path() == tmp_path.resolve()
    assert ws.runtime().abspath().exists()
    assert ws.configs().abspath().exists()
    assert ws.assets().abspath().exists()


def test_storage_safe_path(tmp_path: Path):
    """测试路径逃逸防护"""
    ws = LocalWorkspace(tmp_path)
    storage = ws.root()

    # 正常读写
    storage.put("test.txt", b"hello")
    assert storage.get("test.txt") == b"hello"

    # 路径逃逸尝试
    with pytest.raises(PermissionError):
        storage.get("../outside.txt")


def test_lock_basic_acquire_release(tmp_path: Path):
    """测试锁的基本获取与释放"""
    ws = LocalWorkspace(tmp_path)
    lock = ws.lock("test_lock")

    # 正常获取
    assert lock.acquire(timeout=0) is True
    assert lock.is_locked() is True
    assert lock.is_locked(by_self=True)

    # 重复获取（同对象/同进程通常在 FileLocker 中表现为已存在）
    # 注意：FileLocker 暂不支持重入，所以第二次 acquire 会失败
    assert ws.lock("test_lock").acquire(timeout=0) is False

    lock.release()
    assert lock.is_locked() is False
    assert lock.is_locked(by_self=True) is False
    assert lock.acquire(timeout=0) is True


def test_lock_context_manager(tmp_path: Path):
    """测试上下文管理器"""
    ws = LocalWorkspace(tmp_path)

    with ws.lock("ctx_lock"):
        assert ws.lock("ctx_lock").is_locked() is True

    assert ws.lock("ctx_lock").is_locked() is False


def _other_process_lock(lock_path: Path, hold_time: float):
    """子进程辅助函数：获取锁并持有一段时间"""
    from ghoshell_moss.contracts.workspace import FileLocker
    locker = FileLocker(lock_path)
    # 子进程阻塞直到拿到锁
    if locker.acquire(timeout=2.0):
        time.sleep(hold_time)
        locker.release()


def test_multiprocess_lock_competition(tmp_path: Path):
    """测试跨进程锁竞争"""
    ws = LocalWorkspace(tmp_path)
    lock_name = "multi_proc_test"
    # 显式构造锁路径
    lock_dir = ws.runtime().sub_storage("locks").abspath()
    lock_path = lock_dir / f"{lock_name}.lock"

    # 1. 启动子进程
    p = multiprocessing.Process(target=_other_process_lock, args=(lock_path, 0.5))
    p.start()

    # 2. 【关键改进】主动轮询：等待子进程确认占用了锁
    # 替代不靠谱的 time.sleep(0.1)
    max_wait = 2.0
    start_wait = time.time()
    locker = ws.lock(lock_name)

    is_child_locked = False
    while time.time() - start_wait < max_wait:
        if locker.is_locked():
            is_child_locked = True
            break
        time.sleep(0.01)

    if not is_child_locked:
        p.terminate()
        p.join()
        pytest.fail("子进程未能在规定时间内获取锁")

    try:
        # 3. 尝试非阻塞获取，此时子进程拿着锁，主进程应该失败
        assert locker.acquire(timeout=0) is False, "主进程不应在子进程持锁时抢锁成功"

        # 4. 尝试阻塞获取
        # 子进程持有 0.5s，我们给 1.5s 的容错时间，确保它释放后我们能接管
        assert locker.acquire(timeout=1.5) is True, "子进程释放后，主进程应能阻塞获取成功"

    finally:
        # 5. 【关键改进】无论断言是否通过，都确保回收子进程
        locker.release()
        if p.is_alive():
            p.terminate()
        p.join()


def test_stale_lock_cleanup(tmp_path: Path):
    """测试僵尸锁（PID 已不存在）的自动清理"""
    ws = LocalWorkspace(tmp_path)
    lock_storage = ws.runtime().sub_storage("locks")
    lock_file = lock_storage.abspath() / "stale.lock"

    # 模拟一个已经挂掉的进程 PID (假设 999999 不存在)
    lock_file.write_text("999999")

    locker = ws.lock("stale")
    assert locker.is_locked() is False
    assert locker.is_locked(by_self=True) is False
    assert locker.acquire(timeout=0) is True
    locker.release()
