from mcp.server.fastmcp import FastMCP
mcp = FastMCP("weather")

@mcp.tool()
async def add(x: int, y: int) -> int:
    """将两个字符串相加。

    Args:
        x: 第一个整数
        y: 第二个整数
    """
    return x + y

if __name__ == "__main__":
    # 初始化并运行 server
    mcp.run(transport='stdio')