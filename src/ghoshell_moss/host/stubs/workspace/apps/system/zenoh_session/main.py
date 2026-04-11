import asyncio
import zenoh
import time
from rich.console import Console
from ghoshell_moss.host import Host

# 实例化 console 用于测试中的可视化反馈
console = Console()
host = Host()


async def test_zenoh_connectivity(matrix):
    """测试 Zenoh 基础连通性"""
    session = matrix.container.force_fetch(zenoh.Session)

    # 1. 基础状态检查
    console.print(f"[cyan]Zenoh Session ID:[/cyan] {session.info().zid()}")
    assert not session.is_closed(), "Zenoh Session 应该是开启状态"

    # 2. 简单的 Pub/Sub 回路测试 (自发自收)
    topic = f"moss/test/connectivity/{int(time.time())}"
    received_event = asyncio.Event()

    def on_put(sample):
        console.print(f"[green]✔ Received test pulse on:[/green] {sample.key_expr}")
        received_event.set()

    # 订阅
    sub = session.declare_subscriber(topic, on_put)

    # 发布
    console.print(f"[yellow]Sending pulse to {topic}...[/yellow]")
    session.put(topic, "pulse")

    try:
        # 等待反馈，超时则认为链路有问题
        await asyncio.wait_for(received_event.wait(), timeout=2.0)
        console.print("[bold green]Matrix Zenoh Communication: OK[/bold green]")
    except asyncio.TimeoutError:
        console.print("[red]❌ Matrix Zenoh Communication: Timeout![/red]")
        raise
    finally:
        sub.undeclare()


async def test_container_singleton(matrix):
    """验证 IoC 容器提供的 Session 是否为单例"""
    s1 = matrix.container.force_fetch(zenoh.Session)
    s2 = matrix.container.force_fetch(zenoh.Session)

    assert s1 is s2, "Container 必须返回同一个 Zenoh Session 实例"
    console.print("[bold green]IoC Singleton Integrity: OK[/bold green]")


async def main():
    console.print(f"[bold magenta]Starting Matrix Integration Test[/bold magenta]")

    # 使用你设计的 async context manager
    async with host.matrix() as matrix:
        # 1. 验证基础 Matrix 属性
        console.print(f"[cyan]Current Cell:[/cyan] {matrix.this}")
        assert matrix.is_running(), "Matrix 应处于运行状态"

        # 2. 运行子项测试
        await test_container_singleton(matrix)
        await test_zenoh_connectivity(matrix)

        # 3. 测试 Workspace 集成
        ws = matrix.workspace
        console.print(f"[cyan]Workspace Root:[/cyan] {ws.root()}")

    console.print(f"\n[bold green]All tests passed! Matrix is healthy.[/bold green]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        console.print(f"\n[bold red]Test Failed:[/bold red] {e}")
        exit(1)