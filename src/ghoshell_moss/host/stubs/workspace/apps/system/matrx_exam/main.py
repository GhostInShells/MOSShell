import asyncio
from ghoshell_moss.host.abcd.matrix import Matrix
from ghoshell_moss.core.concepts.topic import LogTopic, TopicClosedError
from ghoshell_moss.host.abcd.session import ConversationItem
from ghoshell_common.helpers import yaml_pretty_dump


async def matrix_smoke_test(matrix: Matrix):
    """
    环境冒烟测试逻辑
    """
    print("\n" + "=" * 50)
    print("🚀 MOSS Matrix 环境冒烟测试启动")
    print("=" * 50)

    # 1. 验证 Cell 自我识别 (this)
    this = matrix.this
    env_str = yaml_pretty_dump(matrix.cell_env())
    print(f"[{this.type.upper()}] 节点名称: {this.name}")
    print(f"[{this.type.upper()}] 节点地址: {this.address}")
    print(f"[{this.type.upper()}] 工作目录: {this.where}")
    print(f"[{this.type.upper()}] 存活状态: {env_str}")

    print(f"[{this.type.upper()}] ENV 信息: {this.is_alive()}")

    # 2. 验证 Session 基础输出
    print("\n--- 验证 Session 输出 ---")
    session = matrix.session
    print(f"当前 Session ID: {session.session_id}")

    # 定义输出回调，验证 Session 的响应能力
    session.on_output(lambda item: print(f"🔔 [Session Output] 角色: {item.role}, 消息数: {len(item.messages)}"))

    # 模拟发送一个 ConversationItem
    test_item = ConversationItem(role="system").with_message("Matrix smoke test message.")
    session.output(test_item)

    # 3. 验证 Topic Service (生产者/消费者并发验证)
    print("\n--- 验证 Topic Service (Zenoh) ---")
    topics = matrix.topics

    # A. 定义异步消费者任务
    async def log_consumer():
        print("[Consumer] LogTopic 消费者任务已就绪...")
        subscriber = topics.subscribe_model(LogTopic, uid="smoke_test_sub")
        async with subscriber:
            try:
                count = 0
                while count < 2:  # 消费两条后自动退出测试
                    model = await subscriber.poll_model(timeout=5.0)
                    if model:
                        print(f"✅ [Consumer] 捕获日志消息: [{model.level}] {model.message}")
                        count += 1
            except asyncio.TimeoutError:
                print("❌ [Consumer] 等待消息超时")
            except TopicClosedError:
                print("[Consumer] Subscriber 已关闭")

    # B. 定义异步生产者任务
    async def log_producer():
        print("[Producer] LogTopic 生产者任务已启动...")
        # 生产者通常也建议使用 async with 生命周期，但在 TopicService.pub 直接发也可以
        # 这里验证 model_publisher
        publisher = topics.model_publisher(creator=this.address, model=LogTopic)
        async with publisher:
            for i in range(2):
                await asyncio.sleep(0.5)
                publisher.pub(LogTopic(level="info", message=f"这是第 {i + 1} 条冒烟测试日志"))
                print(f"📤 [Producer] 已发布消息 {i + 1}")

    # C. 通过 Matrix.create_task 托管任务，验证任务组管理能力
    matrix.create_task(log_consumer())
    matrix.create_task(log_producer())

    # 4. 模拟运行一段时间，确保任务执行完毕
    # 在这里我们不手动等待 tasks 完成，而是观察 Matrix 在退出时是否会自动回收它们
    print("\n[Wait] 等待 3 秒观察异步任务执行...")
    await asyncio.sleep(3)

    print("\n" + "=" * 50)
    print("✨ 环境冒烟测试阶段性完成")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # 使用 Matrix.run 入口，会自动调用 matrix_smoke_test 并处理生命周期
    # 如果 Ctrl+C，你会看到你在 __aexit__ 中写的清理逻辑
    try:
        Matrix.discover().run(matrix_smoke_test)
    except Exception as e:
        print(f"❌ 运行过程中发生异常: {type(e).__name__}: {e}")
