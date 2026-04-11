import asyncio
import orjson
import zenoh
from ghoshell_moss.host.abcd.matrix import Matrix
from ghoshell_moss.core.concepts.topic import TopicModel


async def global_watcher_app(matrix: Matrix):
    """
    全量观察者：监听 MOSS/** 下的所有 Zenoh 消息
    """
    print("\n" + "=" * 60)
    print("🔍 MOSS 全量观察者启动 (Global Watcher)")
    print(f"当前节点地址: {matrix.this.address}")
    print(f"监听范围: MOSS/**")
    print("=" * 60 + "\n")

    # 1. 直接从容器获取已经 bootstrap 的 zenoh session
    # 这样我们不需要处理它的生命周期，Matrix 退出时会自动关闭它
    z_session = matrix.container.force_fetch(zenoh.Session)

    def on_sample(sample: zenoh.Sample):
        """
        处理所有抓取到的样本
        """
        key = str(sample.key_expr)
        payload_raw = sample.payload.to_bytes()

        # 尝试解析 JSON 提高可读性，解析失败则打印原文字符串
        try:
            data = orjson.loads(payload_raw)
            # 格式化打印
            print(f"📩 [{key}]")
            print(f"   Payload: {orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()}")
        except Exception:
            print(f"📩 [{key}]")
            print(f"   Raw: {sample.payload.to_string()}")
        print("-" * 30)

    # 2. 声明全量订阅者
    # 使用 ** 匹配 MOSS/ 下的所有层级
    print("正在建立 Zenoh 订阅...")
    sub = z_session.declare_subscriber("MOSS/**", on_sample)

    try:
        # 3. 保持运行，直到 Matrix 关闭
        print("✅ 观察者已就绪，正在实时截获总线数据...")
        await matrix.wait_closed()
    except asyncio.CancelledError:
        print("\n[Watcher] 收到取消信号，正在停止监听...")
    finally:
        sub.undeclare()
        print("[Watcher] 订阅已释放。")


if __name__ == "__main__":
    # 使用 Matrix 启动，会自动处理 Host 环境发现
    try:
        Matrix.discover().run(global_watcher_app)
    except KeyboardInterrupt:
        print("\n[Watcher] 用户手动终止测试。")
    except Exception as e:
        print(f"\n[Watcher] 异常退出: {e}")
