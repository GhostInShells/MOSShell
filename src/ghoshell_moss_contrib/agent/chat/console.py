import asyncio
import traceback
from datetime import datetime
from typing import Any, Optional

from ghoshell_common.contracts import LoggerItf

from ghoshell_moss_contrib.agent.chat.base import BaseChat
from ghoshell_moss_contrib.agent.depends import check_agent

if check_agent():
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    RICH_AVAILABLE = True

__all__ = [
    "ConsoleChat",
]


class ConsoleChat(BaseChat):
    def __init__(self, logger: LoggerItf | None = None):
        super().__init__()
        # 存储完整的对话历史
        self.conversation_history: list[dict] = []

        # 当前正在处理的AI回复
        self.current_ai_response: Optional[str] = None

        # 标记是否正在流式输出
        self.is_streaming = False

        # 标记是否被用户中断
        self.interrupted = False

        # Rich控制台
        self.console = Console()

        # 使用PromptSession处理输入
        self.prompt_session = PromptSession()

        # 创建键绑定
        self.kb = KeyBindings()
        self.ai_response_done = asyncio.Event()
        self._setup_key_bindings()

        # 打印启动信息
        self.console.print("=== Chat Started ===")
        self.console.print("Type your message and press Enter to send.")
        self.console.print("Press Enter during AI response to interrupt.\n")
        self.console.print("Press `Ctrl+C` to exit.\n")

    def _setup_key_bindings(self):
        """设置键盘快捷键"""

        @self.kb.add("enter")
        def _(event):
            """处理发送消息或中断流式输出"""
            if self.is_streaming:
                # 中断流式输出
                self.interrupted = True
                self.is_streaming = False

                # 调用中断回调
                if self.on_interrupt_callback:
                    self.on_interrupt_callback()
                    self.console.print("\n[yellow]Output interrupted[/yellow]\n")
            else:
                # 获取当前输入缓冲区内容
                buf = event.app.current_buffer
                user_input = buf.text
                buf.reset()

                # 调用输入处理回调
                if user_input.strip() and self.on_input_callback:
                    self.add_user_message(user_input)
                    self.on_input_callback(user_input)

    def set_input_callback(self, callback):
        """设置输入处理回调函数"""
        self.on_input_callback = callback

    def set_interrupt_callback(self, callback):
        """设置中断处理回调函数"""
        self.on_interrupt_callback = callback

    def add_user_message(self, message: str):
        """添加用户消息到历史记录"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"\n\n[green][{timestamp}] User: {message}[/green]")
        self.conversation_history.append({"role": "user", "content": message, "timestamp": timestamp})

    def start_ai_response(self):
        """开始AI回复"""
        self.ai_response_done.clear()
        self.current_ai_response = ""
        self.is_streaming = True
        self.interrupted = False
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"\n\n[white][{timestamp}] AI: [/white]")

    def update_ai_response(self, chunk: str, is_thinking: bool = False):
        """更新AI的流式回复"""
        if self.interrupted:
            return False  # 立即停止

        if not self.is_streaming:
            self.start_ai_response()

        self.current_ai_response += chunk

        # 根据 is_gray 参数选择颜色
        if is_thinking:
            self.console.print(f"[grey50]{chunk}[/grey50]", end="")
        else:
            self.console.print(f"[white]{chunk}[/white]", end="")

        # 检查是否被中断
        if self.interrupted:
            return False

        return True

    def finalize_ai_response(self):
        """完成AI回复"""
        if self.current_ai_response:
            timestamp = datetime.now().strftime("%H:%M:%S")

            # 添加换行
            self.console.print()

            # 保存到历史记录
            self.conversation_history.append(
                {"role": "assistant", "content": self.current_ai_response, "timestamp": timestamp}
            )

            # 如果rich可用且没有被中断，添加Markdown渲染
            if not self.interrupted:
                # self._add_markdown_rendering(self.current_ai_response)
                pass

        self.console.print("\n")
        self.current_ai_response = None
        self.is_streaming = False
        self.interrupted = False
        self.ai_response_done.set()
        self.console.print("> You: ")

    def _add_markdown_rendering(self, content: str):
        """使用rich渲染Markdown"""
        try:
            # 创建Markdown面板
            markdown = Markdown(content)
            panel = Panel(markdown, title="AI Response", border_style="blue", padding=(1, 2))

            # 打印Markdown面板
            self.console.print(panel)

        except Exception as e:
            self.print_exception(e, "Failed to render Markdown")

    def print_exception(self, exception: Any, context: str = ""):
        """打印异常信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 格式化异常信息
        if isinstance(exception, Exception):
            exc_info = traceback.format_exception(type(exception), exception, exception.__traceback__)
            error_msg = "".join(exc_info)
        else:
            error_msg = str(exception)

        # 添加上下文信息
        if context:
            error_msg = f"[{context}]\n{error_msg}"

        # 打印错误信息（红色）
        self.console.print(f"[red][{timestamp}] ERROR: {error_msg}[/red]")

    async def run(self):
        """运行聊天界面主循环"""

        try:
            while True:
                # 使用PromptSession获取用户输入（无颜色提示）
                try:
                    user_input = await self.prompt_session.prompt_async("> You: ", key_bindings=self.kb)

                except (EOFError, KeyboardInterrupt):
                    self.console.print("[yellow]Exiting...[/yellow]")
                    break
                except Exception as e:
                    self.print_exception(e, "Error getting input")

        finally:
            # 清理资源
            self.is_streaming = False
            self.interrupted = False
