from abc import ABC, abstractmethod
from typing import Any


class BaseChat(ABC):
    def __init__(self):
        # 输入回调函数
        self.on_input_callback = None
        # 中断回调函数
        self.on_interrupt_callback = None

    def set_input_callback(self, callback):
        """设置输入处理回调函数"""
        self.on_input_callback = callback

    def set_interrupt_callback(self, callback):
        """设置中断处理回调函数"""
        self.on_interrupt_callback = callback

    @abstractmethod
    def add_user_message(self, message: str):
        """添加用户消息到历史记录"""

    @abstractmethod
    def start_ai_response(self):
        """开始AI回复"""

    @abstractmethod
    def update_ai_response(self, chunk: str, is_thinking: bool = False):
        """更新AI的流式回复"""

    @abstractmethod
    def finalize_ai_response(self):
        """完成AI回复"""

    @abstractmethod
    def print_exception(self, exception: Any, context: str = ""):
        """打印异常信息"""

    @abstractmethod
    async def run(self):
        pass
