import urllib.parse
import webbrowser

from ghoshell_moss import PyChannel

__all__ = ["new_mermaid_chan"]

"""
实现一个 Mermaid Channel, 让 AI 在对话上下文中可以随时通过浏览器绘制 mermaid 图形来表达思路. 

预计在 Beta 版本中实现的功能: 

1. 指定一个目录, 用 markdown 文件的方式存储 mermaid. 
2. 允许 AI 在绘制一个 Mermaid 同时存储它, 方便未来读取. 
3. Channel 上下文列出已经存储的 Mermaid id 和简介, 方便模型直接调用已经绘制过的图. 
4. AI 可以通过 command, 读取一个 mermaid id, 并且修改它.  
"""


def new_mermaid_chan() -> PyChannel:
    channel = PyChannel(
        name="mermaid",
        description="在浏览器中绘制 Mermaid 架构图、流程图等",
        blocking=True,
    )

    channel.build.command()(draw_mermaid)

    return channel


async def draw_mermaid(title: str = "MOSShell Diagram", text__: str = "") -> str:
    """
    在浏览器中绘制 Mermaid 图表

    Args:
        title: 图表标题
        text__: Mermaid 代码
    CTML:
        注意使用 cdata: <draw_mermaid title="xx"><![CDATA[...]]></draw_mermaid>

    Returns:
        状态消息
    """

    # 创建在线编辑器 URL
    url = _create_editor_url(text__, title)

    # 打开浏览器
    webbrowser.open(url)

    return f"已在浏览器中打开 Mermaid 图表: {title}"


def _create_editor_url(code: str, title: str) -> str:
    """创建在线编辑器 URL"""
    # 使用 mermaid.live 编辑器
    base_url = "https://mermaid.live/edit"

    # 构建完整的 HTML 页面
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({{startOnLoad:true}});</script>
    </head>
    <body>
        <div class="mermaid">
            {code}
        </div>
    </body>
    </html>
    """

    # 转换为 data URL
    encoded = urllib.parse.quote(html)
    return f"data:text/html;charset=utf-8,{encoded}"
