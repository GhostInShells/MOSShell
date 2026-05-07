import asyncio
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.widgets import Frame
from prompt_toolkit.layout.controls import FormattedTextControl
from ghoshell_moss.core.blueprint.matrix import Matrix
from prompt_toolkit.key_binding import KeyBindings

kb = KeyBindings()


@kb.add('c-c')  # 绑定 Ctrl + C
@kb.add('q')  # 或者按下 q 键退出
def exit_app(event):
    event.app.exit()


class MossMonitor:
    def __init__(self, matrix: Matrix):
        self.buffer = matrix.session.output_buffer(maxsize=20)
        # 渲染内容的容器
        self.control = FormattedTextControl(text=self._get_text)

    def _get_text(self):
        lines = []
        for item in self.buffer.values():
            # 这里按照 role 给一点简单的样式前缀
            color = 'ansired' if item.role == 'error' else 'ansigreen'
            lines.append((f'class:{color}', f"[{item.role.upper()}] "))
            for msg in item.messages:
                lines.append(('', f"{msg.to_content_string()}\n"))
        return lines

    async def run(self):
        layout = Layout(Frame(Window(self.control), title="MOSS Real-time Output"))
        app = Application(
            layout=layout,
            full_screen=True,
            key_bindings=kb  # <--- 加上这一行
        )

        # 启动一个异步任务更新界面
        async def updater():
            while True:
                app.invalidate()  # 强制重绘
                await asyncio.sleep(0.5)

        asyncio.create_task(updater())
        await app.run_async()


async def monitor_main(matrix: Matrix):
    monitor = MossMonitor(matrix)
    await monitor.run()


if __name__ == "__main__":
    Matrix.discover().run(monitor_main)
