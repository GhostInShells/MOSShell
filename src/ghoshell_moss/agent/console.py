import asyncio
from typing import List, Dict, Optional, AsyncGenerator, Callable, Any

from ghoshell_moss import CommandTask
from ghoshell_moss.depends import check_agent
from ghoshell_moss.concepts.shell import Output, OutputStream
from ghoshell_common.helpers import uuid
import traceback

if check_agent():
    from prompt_toolkit import Application
    from prompt_toolkit.layout import Layout, HSplit
    from prompt_toolkit.widgets import TextArea, Label
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style


class ChatRenderer:
    def __init__(self):
        # 对话历史区域（只读，支持滚动）
        self.history_area = TextArea(
            text="=== Chat Started ===\n",
            read_only=False,
            focusable=False,
            scrollbar=True
        )

        # 输入区域
        self.input_field = TextArea(
            height=3,
            prompt="You: ",
            multiline=False,
            wrap_lines=True
        )

        # 状态栏
        self.status_bar = Label(text="Ready")

        # 创建布局
        self.layout = Layout(HSplit([
            Label("Simple Chat - Press Ctrl+C to exit", style="class:title"),
            self.history_area,
            Label("─" * 50, style="class:line"),
            self.input_field,
            self.status_bar
        ]))

        # 创建键绑定
        self.kb = KeyBindings()
        self._setup_key_bindings()

        # 创建应用样式
        self.style = Style.from_dict({
            'title': 'bg:#000044 #ffffff bold',
            'line': '#888888',
            'status': 'bg:#000044 #ffffff',
            'user': '#00aa00',
            'assistant': '#0088ff',
            'time': '#888888 italic'
        })

        # 创建应用实例
        self.app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self.style,
            full_screen=True
        )

        # 存储完整的对话历史
        self.conversation_history: List[Dict] = []

        # 当前正在处理的AI回复（用于流式更新）
        self.current_ai_response: Optional[str] = None

        # 输入回调函数（由外部设置）
        self.on_input_callback = None

    def _setup_key_bindings(self):
        """设置键盘快捷键"""

        @self.kb.add('enter')
        def _(event):
            """处理发送消息"""
            if self.input_field.text.strip() and self.on_input_callback:
                # 获取用户输入并清空输入框
                user_input = self.input_field.text
                self.input_field.text = ""

                # 调用外部回调处理输入
                asyncio.create_task(self.on_input_callback(user_input))

        @self.kb.add('c-c')
        def _(event):
            """Ctrl+C 退出"""
            event.app.exit()

    def set_input_callback(self, callback):
        """设置输入处理回调函数"""
        self.on_input_callback = callback

    def print_exception(self, exception: Any, context: str = "") -> None:
        """
        在聊天界面中显示异常信息

        Args:
            exception: 异常对象或字符串
            context: 异常发生的上下文描述
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 格式化异常信息
        if isinstance(exception, Exception):
            # 获取完整的异常堆栈
            exc_info = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
            error_msg = "".join(exc_info)
        else:
            # 直接使用字符串
            error_msg = str(exception)

        # 添加上下文信息
        if context:
            error_msg = f"[{context}]\n{error_msg}"

        # 添加到历史记录
        self.conversation_history.append({
            "role": "System",
            "content": f"ERROR: {error_msg}",
            "timestamp": timestamp,
            "is_temp": False
        })

        # 更新界面显示
        self._refresh_history_display()

        # 更新状态栏
        self.status_bar.text = "Error occurred"

    def add_user_message(self, message: str):
        """添加用户消息到历史记录"""
        self._add_message("User", message)

    def start_ai_response(self):
        """开始AI回复（初始化流式回复）"""
        self.current_ai_response = ""
        # 添加一个空的AI消息占位符
        self._add_message("AI", "", is_temp=True)

    def update_ai_response(self, chunk: str):
        """更新AI的流式回复"""
        if self.current_ai_response is None:
            self.start_ai_response()

        self.current_ai_response += chunk

        # 更新最后一条消息（临时消息）
        if self.conversation_history and self.conversation_history[-1].get("is_temp", False):
            # 更新最后一条临时消息
            self.conversation_history[-1]["content"] = self.current_ai_response
            self._refresh_history_display()

    def finalize_ai_response(self):
        """完成AI回复（移除临时标记）"""
        if self.conversation_history and self.conversation_history[-1].get("is_temp", False):
            self.conversation_history[-1]["is_temp"] = False
            self._refresh_history_display()
        self.current_ai_response = None

    def set_status(self, status: str):
        """更新状态栏"""
        self.status_bar.text = status

    def _add_message(self, role: str, content: str, is_temp: bool = False):
        """内部方法：添加消息到历史记录"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 保存到内存
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "is_temp": is_temp
        })

        # 更新界面显示
        self._refresh_history_display()

    def _refresh_history_display(self):
        """刷新历史区域显示"""
        display_text = ""
        for msg in self.conversation_history:
            timestamp = msg["timestamp"]
            role = msg["role"]
            content = msg["content"]

            # 格式化显示
            if role == "User":
                display_text += f"[{timestamp}] {role}: {content}\n"
            else:
                display_text += f"[{timestamp}] {role}: {content}\n"

        # 更新界面
        self.history_area.buffer.text = display_text
        # 滚动到底部
        self.history_area.buffer.cursor_position = len(self.history_area.buffer.text)

    async def run(self):
        """运行聊天界面"""
        await self.app.run_async()


class ChatRenderOutputStream(OutputStream):

    def __init__(
            self,
            batch_id: str,
            output: Callable[[str], None],
            *,
            on_start: asyncio.Event,
            close: asyncio.Event,
    ):
        self.id = batch_id
        self._output = output
        self._buffered = ""
        self._input_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._started = False
        self._on_start = on_start
        self._close_event = close
        self._main_loop_task: Optional[asyncio.Task] = None

    async def _main_loop(self):
        try:
            while not self._close_event.is_set():
                item = await self._input_queue.get()
                if item is None:
                    break
                self._output(item)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            pass
        finally:
            self._close_event.set()

    def buffer(self, text: str, *, complete: bool = False) -> None:
        if text:
            self._buffered += text
            if self._started:
                self._input_queue.put_nowait(text)
        if complete:
            self._input_queue.put_nowait(None)

    def start(self) -> None:
        if self._started:
            return
        if len(self._buffered) > 0:
            self._input_queue.put_nowait(self._buffered)
        self._started = True
        self._on_start.set()
        self._main_loop_task = asyncio.create_task(self._main_loop())

    def close(self):
        self.commit()
        self._close_event.set()

    def buffered(self) -> str:
        return self._buffered

    async def wait(self) -> None:
        if self._main_loop_task:
            await self._main_loop_task


class ChatRenderOutput(Output):

    def __init__(self, render: ChatRenderer):
        self.render = render
        self.last_stream_close_event = asyncio.Event()
        self._outputted = {}

    def new_stream(self, *, batch_id: Optional[str] = None) -> OutputStream:
        batch_id = batch_id or uuid()
        last_stream_close_event = self.last_stream_close_event
        new_close_event = asyncio.Event()
        self.last_stream_close_event = new_close_event
        self._outputted[batch_id] = []

        def _output(item: str):
            self._outputted[batch_id].append(item)
            self.render.update_ai_response(item)

        return ChatRenderOutputStream(
            batch_id,
            _output,
            on_start=last_stream_close_event,
            close=new_close_event
        )

    def outputted(self) -> List[str]:
        return list(self._outputted.values())

    def clear(self) -> List[str]:
        outputted = self.outputted()
        self._outputted.clear()
        self.last_stream_close_event = asyncio.Event()
        return outputted


# 主循环框架示例
async def main():
    # 创建渲染器
    renderer = ChatRenderer()
    output = ChatRenderOutput(renderer)

    # 设置输入回调
    async def handle_user_input(user_input):
        # 1. 显示用户消息
        renderer.add_user_message(user_input)

        # 2. 开始AI回复
        renderer.start_ai_response()
        renderer.set_status("Thinking...")

        # 3. 模拟流式输出（这里替换为你的实际逻辑）
        response_text = "这是一个模拟的AI回复，逐步显示..."
        stream = output.new_stream()
        stream.start()
        for i in range(len(response_text)):
            chunk = response_text[i]
            await asyncio.sleep(0.1)
            stream.buffer(chunk)
        stream.commit()
        await stream.wait()
        stream.close()

        # 4. 完成回复
        renderer.finalize_ai_response()
        renderer.set_status("Ready")

    renderer.set_input_callback(handle_user_input)

    # 运行界面
    await renderer.run()


if __name__ == "__main__":
    asyncio.run(main())
