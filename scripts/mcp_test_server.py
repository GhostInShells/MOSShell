"""
MCP Hub 验证用测试 server。

启动后 MCP Hub 可通过 stdio transport 连接此 server，测试 exec/list_servers 等命令。

用法:
  # SSE (默认, 端口 20873)
  python scripts/mcp_test_server.py

  # stdio (MCP Hub 直连)
  python scripts/mcp_test_server.py --stdio
"""

import sys
import time
from datetime import datetime

import click
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MCP Test Server")


@mcp.tool()
def echo(text: str) -> str:
    """回显输入文本。"""
    return f"ECHO: {text}"


@mcp.tool()
def add(a: float, b: float) -> str:
    """两数相加。"""
    return f"{a} + {b} = {a + b}"


@mcp.tool()
def get_time() -> str:
    """获取当前服务器时间。"""
    return datetime.now().isoformat()


@mcp.tool()
def slow_echo(text: str, delay: float = 2.0) -> str:
    """延迟回显，用于测试 @nonblocking exec 的异步行为。"""
    time.sleep(delay)
    return f"SLOW({delay}s): {text}"


@click.command()
@click.option("--stdio", is_flag=True, help="使用 stdio transport（MCP Hub 直连）")
@click.option("--port", default=20873, help="SSE 模式端口")
def main(stdio: bool, port: int):
    if stdio:
        mcp.run(transport="stdio")
    else:
        print(f"Starting MCP test server on http://localhost:{port}/sse")
        mcp.run(transport="sse", port=port)


if __name__ == "__main__":
    main()
