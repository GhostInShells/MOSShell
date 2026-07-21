"""在浏览器中渲染 Mermaid 图表 | 交互能力 | alpha

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.mermaid_draw import new_mermaid_channel
    main = new_shell_main_channel()
    main.import_channels(new_mermaid_channel())
"""

import webbrowser

from ghoshell_moss.core import PyChannel

__all__ = ["new_mermaid_channel"]

_MERMAID_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({{
    startOnLoad: true,
    theme: 'default'
  }});
</script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 100%; height: 100%;
    background: #fff;
  }}
  body {{
    display: flex; flex-direction: column;
  }}
  h1 {{
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 1.5rem; padding: 1rem;
    text-align: center;
    flex-shrink: 0;
  }}
  .wrapper {{
    flex: 1;
    display: flex; justify-content: center; align-items: center;
    min-height: 0;
  }}
  .mermaid {{
    height: 100%;
    display: flex; justify-content: center; align-items: center;
  }}
  .mermaid svg {{
    height: 100% !important;
    width: auto !important;
    max-width: none !important;
  }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="wrapper">
    <pre class="mermaid">
{code}
    </pre>
  </div>
</body>
</html>"""


def new_mermaid_channel(
    *,
    name: str = "mermaid",
    description: str = "在浏览器中绘制 Mermaid 架构图、流程图、时序图等",
) -> PyChannel:
    chan = PyChannel(name=name, description=description, blocking=True)

    async def draw(title: str = "MOSS Diagram", text__: str = "") -> str:
        """在浏览器中渲染 Mermaid 图表。

        支持所有 Mermaid 图类型：flowchart、sequence、class、state、er、
        gantt、pie、git、mindmap、timeline、block、architecture 等。

        :param title: 图表标题
        :param text__: Mermaid 代码。必须用 CDATA 包裹，不要用 ```mermaid 代码块包裹。
            节点内换行用 <br/>，不要用 \\n
        """
        import base64
        import re

        code = text__.strip()
        code = re.sub(r"^```(?:mermaid)?\s*\n?", "", code)
        code = re.sub(r"\n?```\s*$", "", code)
        code = code.replace("\\n", "<br/>")

        html = _MERMAID_HTML.replace("{title}", title).replace("{code}", code)
        b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
        url = f"data:text/html;charset=utf-8;base64,{b64}"
        webbrowser.get('chrome').open(url)
        return f"已在浏览器中渲染 Mermaid 图表: {title}"

    chan.build.command(always_observe=False)(draw)

    @chan.build.instruction
    def mermaid_instruction() -> str:
        return (
            "Mermaid 图表绘制通道。通过 draw 命令在浏览器中渲染图表。\n"
            "支持的类型: flowchart, sequence, class, state, er, gantt, pie, "
            "git, mindmap, timeline, block, architecture.\n"
            "text__ 直接写 mermaid 语法，不要用 ```mermaid 代码块包裹。\n"
            "节点内换行用 <br/> 不要用 \\n。\n"
            "CTML 用法: <mermaid:draw title=\"架构图\"><![CDATA[\n"
            "flowchart TD\n"
            "  A[开始] --> B[结束]\n"
            "]]></mermaid:draw>\n"
            "通道名是 mermaid，CTML 写作 <mermaid:draw>。绝对不要写成 <xxx:mermaid>。"
        )

    return chan
