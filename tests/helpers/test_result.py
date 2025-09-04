from ghoshell_moss.helpers.result import ThreadSafeResult

import pytest
import asyncio
import threading
import time


class TestHybridAwait:
    """HybridAwait 类的单元测试"""

    @pytest.mark.asyncio
    async def test_async_wait_with_result(self):
        """测试异步等待并获取结果"""
        waiter = ThreadSafeResult[str]()

        # 在另一个线程中设置结果
        def set_result():
            time.sleep(0.1)
            waiter.resolve("test_result")

        threading.Thread(target=set_result).start()

        # 异步等待结果
        result = await waiter.wait_async(timeout=0.2)
        assert result == "test_result"
        assert waiter.is_done()

    @pytest.mark.asyncio
    async def test_async_wait_with_cancel(self):
        """测试异步等待被取消的情况"""
        waiter = ThreadSafeResult[str]()

        # 在另一个线程中取消操作
        def cancel_operation():
            time.sleep(0.1)
            waiter.cancel("Operation cancelled")

        threading.Thread(target=cancel_operation).start()

        # 异步等待应该抛出 CancelledError
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await waiter.wait_async(timeout=1.0)

        assert "Operation cancelled" in str(exc_info.value)
        assert waiter.is_done()

    @pytest.mark.asyncio
    async def test_async_wait_timeout(self):
        """测试异步等待超时的情况"""
        waiter = ThreadSafeResult[str]()

        # 不设置结果，应该超时
        with pytest.raises(asyncio.TimeoutError):
            await waiter.wait_async(timeout=0.1)

    def test_sync_wait_with_result(self):
        """测试同步等待并获取结果"""
        waiter = ThreadSafeResult[str]()

        # 在另一个线程中设置结果
        def set_result():
            time.sleep(0.1)
            waiter.resolve("sync_result")

        threading.Thread(target=set_result).start()

        # 同步等待结果
        result = waiter.wait(timeout=1.0)
        assert result == "sync_result"
        assert waiter.is_done()

    def test_sync_wait_with_cancel(self):
        """测试同步等待被取消的情况"""
        waiter = ThreadSafeResult[str]()

        # 在另一个线程中取消操作
        def cancel_operation():
            time.sleep(0.1)
            waiter.cancel("Sync operation cancelled")

        threading.Thread(target=cancel_operation).start()

        # 同步等待应该抛出 RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            waiter.wait(timeout=1.0)

        assert "Sync operation cancelled" in str(exc_info.value)
        assert waiter.is_done()

    def test_sync_wait_timeout(self):
        """测试同步等待超时的情况"""
        waiter = ThreadSafeResult[str]()

        # 不设置结果，应该超时
        with pytest.raises(TimeoutError):
            result = waiter.wait(timeout=0.1)
            assert result is None

    @pytest.mark.asyncio
    async def test_multiple_async_waiters(self):
        """测试多个异步等待者"""
        waiter = ThreadSafeResult[str]()
        results = []

        async def async_waiter(_id: int):
            try:
                result = await waiter.wait_async(timeout=2.0)
                results.append(f"waiter_{_id}: {result}")
            except asyncio.CancelledError as e:
                results.append(f"waiter_{_id}: cancelled - {e}")

        # 创建多个异步等待任务
        tasks = [asyncio.create_task(async_waiter(i)) for i in range(3)]

        # 等待一会儿后设置结果
        await asyncio.sleep(0.1)
        waiter.resolve("multiple_result")

        # 等待所有任务完成
        await asyncio.gather(*tasks)

        # 所有等待者都应该收到结果
        assert len(results) == 3
        for i in range(3):
            assert f"waiter_{i}: multiple_result" in results

    @pytest.mark.asyncio
    async def test_mixed_sync_async_waiters(self):
        """测试混合同步和异步等待者"""
        waiter = ThreadSafeResult[str]()
        async_results = []
        sync_results = []

        # 异步等待者
        async def async_waiter():
            try:
                result = await waiter.wait_async(timeout=2.0)
                async_results.append(f"async: {result}")
            except asyncio.CancelledError as e:
                async_results.append(f"async: cancelled - {e}")

        # 同步等待者（在另一个线程中运行）
        def sync_waiter():
            try:
                result = waiter.wait(timeout=2.0)
                sync_results.append(f"sync: {result}")
            except Exception as e:
                sync_results.append(f"sync: error - {e}")

        # 启动异步等待者
        async_task = asyncio.create_task(async_waiter())

        # 启动同步等待者线程
        sync_thread = threading.Thread(target=sync_waiter)
        sync_thread.start()

        # 等待一会儿后设置结果
        await asyncio.sleep(0.1)
        waiter.resolve("mixed_result")

        # 等待所有完成
        await async_task
        sync_thread.join()

        # 检查结果
        assert len(async_results) == 1
        assert len(sync_results) == 1
        assert "async: mixed_result" in async_results
        assert "sync: mixed_result" in sync_results

    def test_double_done_raises_error(self):
        """测试重复设置结果会抛出异常"""
        waiter = ThreadSafeResult[str]()
        waiter.resolve("first_result")

        with pytest.raises(RuntimeError) as exc_info:
            waiter.resolve("second_result")

        assert "Already set result" in str(exc_info.value)

    def test_cancel_after_done_no_effect(self):
        """测试在完成后取消操作不会有任何效果"""
        waiter = ThreadSafeResult[str]()
        waiter.resolve("result")

        # 这不应该抛出异常
        waiter.cancel("too_late")

        # 应该仍然能获取到结果，而不是取消
        result = waiter.wait()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_immediate_result(self):
        """测试在等待前已经设置结果的情况"""
        waiter = ThreadSafeResult[str]()
        waiter.resolve("immediate_result")

        # 异步等待应该立即返回结果
        result = await waiter.wait_async()
        assert result == "immediate_result"

        # 同步等待也应该立即返回结果
        result2 = waiter.wait()
        assert result2 == "immediate_result"

    @pytest.mark.asyncio
    async def test_async_wait_already_done_with_cancel(self):
        """测试在等待前已经取消的情况"""
        waiter = ThreadSafeResult[str]()
        waiter.cancel("pre_cancelled")

        # 异步等待应该抛出 CancelledError
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await waiter.wait_async()

        assert "pre_cancelled" in str(exc_info.value)

        # 同步等待应该抛出 RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            waiter.wait()

        assert "pre_cancelled" in str(exc_info.value)

    def test_multiple_thread_await(self):
        waiter = ThreadSafeResult[str]()

        async def consumer():
            r = await waiter.wait_async()
            assert r == "hello"

        def consumer_main():
            asyncio.run(consumer())

        threads = []
        for i in range(3):
            t = threading.Thread(target=consumer_main)
            t.start()
            threads.append(t)

        waiter.resolve("hello")
        for t in threads:
            t.join()

        assert waiter.wait() == "hello"
