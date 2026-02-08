from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")


@mcp.tool()
async def add(x: int, y: int = 2) -> int:
    """将两个字符串相加。

    Args:
        x: 第一个整数
        y: 第二个整数
    """
    return x + y


@mcp.tool()
async def foo(a: int, b: dict[str, int]) -> int:
    """测试函数。

    Args:
        a: 示例参数
        b: 字典函数
    """
    return a + b.get("i", 0)


@mcp.tool()
async def bar(s: str) -> int:
    """测试函数。

    Args:
        a: 集合参数
    """
    return len(s)


@mcp.tool()
async def multi(a: int, b: int, c: int, d: int) -> int:
    """测试函数。

    Args:
        a: 测试参数
        b: 测试参数
        c: 测试参数
        d: 测试参数
    """
    return a + b + c + d


if __name__ == "__main__":
    # 初始化并运行 server
    mcp.run(transport="stdio")
