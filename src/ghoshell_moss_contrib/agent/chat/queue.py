import asyncio
from datetime import datetime
import traceback
from typing import Any, Dict, List, Optional, Literal

from ghoshell_moss.message import Text, Message
from ghoshell_moss_contrib.agent.chat.base import BaseChat

import logging

logger = logging.getLogger("QueueChat")


class QueueChat(BaseChat):
    """基于 asyncio.Queue 的聊天实现"""

    def __init__(self, input_queue: asyncio.Queue[Message], output_queue: asyncio.Queue[Message]):
        """
        初始化 QueueChat

        Args:
            input_queue: 输入队列，外部通过此队列发送输入
            output_queue: 输出队列，聊天通过此队列发送输出
        """
        self.input_queue = input_queue
        self.output_queue = output_queue

        # 存储完整的对话历史
        self.conversation_history: List[Dict] = []

        # 当前正在处理的AI回复
        self.current_ai_response: str = ""

        # 标记是否正在流式输出
        self.is_streaming = False

        # 标记是否被用户中断
        self.interrupted = False

        # 生命周期
        self.is_closed = asyncio.Event()

        super().__init__()

    def _send_output(self, role, text: str = "", is_final: bool = False):
        """发送消息到输出队列"""
        # 放入输出队列（非阻塞方式）
        message = Message.new(role=role, name="__queue__").with_content(Text(text=text))
        if not is_final:
            message.seq = "incomplete"
        try:
            self.output_queue.put_nowait(message)
        except asyncio.QueueFull:
            # 如果队列满，尝试创建任务异步放入
            asyncio.create_task(self.output_queue.put(message))

    def add_user_message(self, message: str):
        """添加用户消息到历史记录"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 保存到历史记录
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": timestamp
        })

        # 发送到输出队列
        self._send_output(role="user", text=message)

    def start_ai_response(self):
        """开始AI回复"""
        self.current_ai_response = ""
        self.is_streaming = True
        self.interrupted = False

        self._send_output(role="system", text="ai_start")

    def update_ai_response(self, chunk: str, is_thinking: bool = False):
        """更新AI的流式回复"""
        if not self.is_streaming:
            self.start_ai_response()

        self.current_ai_response += chunk

        self._send_output(role="assistant", text=chunk)

        # 检查是否被中断
        if self.interrupted:
            return False

        return True

    def finalize_ai_response(self):
        """完成AI回复"""
        if self.current_ai_response:
            timestamp = datetime.now().strftime("%H:%M:%S")

            self._send_output(role="assistant", text=self.current_ai_response, is_final=True)

            # 保存到历史记录
            self.conversation_history.append({
                "role": "assistant",
                "content": self.current_ai_response,
                "timestamp": timestamp
            })

        self.current_ai_response = ""
        self.is_streaming = False
        self.interrupted = False

    def print_exception(self, exception: Any, context: str = ""):
        """打印异常信息"""
        # 格式化异常信息
        if isinstance(exception, Exception):
            exc_info = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
            error_msg = "".join(exc_info)
        else:
            error_msg = str(exception)

        # 添加上下文信息
        if context:
            error_msg = f"[{context}]\n{error_msg}"

        # 发送错误信息
        self._send_output(
            role="system",
            text=error_msg,
        )

    async def run(self):
        """运行聊天界面 - 阻塞循环，处理输入队列"""
        logger.info("队列聊天已启动")
        # 发送连接成功消息
        self._send_output(
            role="system",
            text="队列聊天已启动",
        )

        while not self.is_closed.is_set():
            try:
                # 阻塞等待输入队列的消息
                request = await self.input_queue.get()
                logger.info(f"收到用户输入: {request}")

                # 处理空输入
                if request is None:
                    continue

                # 处理用户输入
                for content in request.contents:
                    text = Text.from_content(content)
                    # 目前只处理文本内容
                    if not text:
                        continue

                    # 处理中断信号
                    if text.text == "":
                        self.interrupted = True
                        self.is_streaming = False
                        # 发送中断消息

                        self._send_output(
                            role="system",
                            text="输出已被中断",
                        )

                        if self.on_interrupt_callback:
                            await self.on_interrupt_callback()
                        continue

                    self.add_user_message(text.text)
                    if self.on_input_callback:
                        self.on_input_callback(text.text)

                # 处理其他未知请求类型
                else:
                    logger.warning(f"未知请求类型: {request}")
                    self.print_exception("未知请求类型")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"处理输入消息时出错: {e}")
                self.print_exception(e, "处理输入消息时出错")
            finally:
                self.input_queue.task_done()

    def get_conversation_history(self) -> List[Dict]:
        """获取对话历史"""
        return self.conversation_history.copy()

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()

    def close(self):
        self.is_closed.set()
        self.input_queue.put_nowait(None) # 发送关闭信号
