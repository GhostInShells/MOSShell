import logging
import time
from typing import Optional

import mss
from ghoshell_common.contracts import LoggerItf
from PIL import Image

from ghoshell_moss import PyChannel
from ghoshell_moss.message import Base64Image, Message, Text

__all__ = ["ScreenCapture"]

"""
实现一个基本的电脑屏幕截图 channel. 

通过截图, 可以让模型看到指定的屏幕或者窗口的展示. 也是一种读屏的方案. 

预计 Beta 版本实现的功能: 
1. 让 AI 更清晰地理解电脑哪些屏幕可以读屏, 截屏
2. 通过 channel 配置, 渲染一个 pyqt6 界面, 可以随时查看 AI 看到的截图是什么. 
"""


class ScreenCapture:
    """
    屏幕截图感知Channel
    只在模型思考时捕获关键帧（按需截图）
    """

    def __init__(self, logger: Optional[LoggerItf] = None):
        self.logger = logger or logging.getLogger(__name__)

        # 状态控制
        self._capturing = False

        # mss 相关
        self._mss_initialized = False
        self._mss = None
        self._monitors_info: dict[int, dict] = {}

        # 性能统计
        self._capture_count = 0
        self._last_capture_time = 0.0
        self._total_capture_time = 0.0

    def _init_mss(self):
        """初始化 mss 截图库"""
        if self._mss_initialized:
            return
        try:
            self._mss = mss.mss()
            self._mss_initialized = True

            # 获取显示器信息
            for i, monitor in enumerate(self._mss.monitors[1:], start=1):
                self._monitors_info[i] = {
                    "left": monitor["left"],
                    "top": monitor["top"],
                    "width": monitor["width"],
                    "height": monitor["height"],
                    "index": i,
                }

            self.logger.info("mss 初始化成功，检测到 %s 个显示器", len(self._monitors_info))

        except ImportError:
            self.logger.exception("请安装 mss: pip install mss")
            self._mss_initialized = False
        except Exception:
            self.logger.exception("mss 初始化失败")
            self._mss_initialized = False

    def status_description(self) -> str:
        """
        返回当前组件状态的描述
        """
        if not self._capturing:
            return "屏幕截图功能关闭状态"
        self._init_mss()

        monitor_count = len(self._monitors_info)
        cutting_status = "开启" if self._capturing else "关闭"

        # 显示器信息
        monitor_info = []
        for idx, info in self._monitors_info.items():
            monitor_info.append(f"显示器 {idx}: {info['width']}×{info['height']}")

        monitors_str = "\n".join(monitor_info) if monitor_info else "无"

        # 性能统计
        avg_time = self._total_capture_time / self._capture_count if self._capture_count > 0 else 0

        description = f"""
屏幕截图模块状态：
- 截图功能: {cutting_status}
- 可用显示器: {monitor_count} 个
- 累计截图: {self._capture_count} 次
- 平均耗时: {avg_time:.3f} 秒

显示器信息：
{monitors_str}
        """.strip()

        return description

    def is_screen_capturing(self) -> bool:
        """
        当前是否开启了屏幕截图
        """
        return self._capturing

    async def set_capturing(self, toggle: bool = True) -> None:
        """
        开启或关闭屏幕截图
        """
        old_status = self._capturing
        self._capturing = toggle

        if old_status != toggle:
            status = "开启" if toggle else "关闭"
            self.logger.info("屏幕截图功能已%s", status)

    async def capture(self) -> dict[int, Image.Image]:
        """
        捕获一帧屏幕截图（按需调用）
        返回：{显示器编号: PIL Image}
        """
        if not self._mss_initialized:
            self.logger.warning("mss 未初始化，无法截图")
            return {}

        start_time = time.time()
        screenshots = {}

        try:
            # 捕获所有显示器
            for monitor_idx, monitor_info in self._monitors_info.items():
                # 使用 mss 截图
                screenshot = self._mss.grab(monitor_info)

                # 转换为 PIL Image
                # 注意：mss 返回的是 BGRA，需要转换为 RGB
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

                screenshots[monitor_idx] = img

            # 更新统计信息
            capture_time = time.time() - start_time
            self._capture_count += 1
            self._total_capture_time += capture_time
            self._last_capture_time = start_time

            self.logger.debug("截图完成，耗时: %.3f秒，捕获 %s 个显示器", capture_time, len(screenshots))

            return screenshots

        except Exception:
            self.logger.exception("截图失败")
            return {}

    async def screen_messages(self) -> list[Message]:
        """
        生成屏幕截图上下文消息
        在模型思考时被调用
        """
        # 创建基础消息
        desc = self.status_description()
        message = Message.new(
            role="user",
            name="__screen_cutting__",
        ).with_content(Text(text=desc))

        # 如果截图功能未开启，只返回描述
        if not self.is_screen_capturing():
            return [message]

        # 捕获屏幕截图
        screenshots = await self.capture()

        if not screenshots:
            # 截图失败，返回描述
            return [message]

        # 添加截图到消息
        for monitor_idx, screenshot in screenshots.items():
            monitor_info = self._monitors_info.get(monitor_idx, {})
            width = monitor_info.get("width", "未知")
            height = monitor_info.get("height", "未知")

            caption = f"显示器 {monitor_idx} 的截图 ({width}×{height})"

            message.with_content(Text(text=caption), Base64Image.from_pil_image(screenshot))

        return [message]

    def as_channel(self) -> PyChannel:
        """
        包装为 MOSShell Channel
        """
        channel = PyChannel(
            name="screen_capture",
            description="获取当前所在设备的屏幕上的视觉信息，通过屏幕截图。只有用户要求的时候才使用. ",
            dynamic=True,
        )

        # 注册上下文消息生成器
        channel.build.with_context_messages(self.screen_messages)

        # 注册控制命令
        channel.build.command()(self.set_capturing)

        # 添加状态查询命令
        @channel.build.command()
        async def screen_status() -> str:
            """查询屏幕截图模块状态"""
            return self.status_description()

        return channel

    def close(self):
        """清理资源"""
        if self._mss is not None:
            try:
                self._mss.close()
                self.logger.info("mss 资源已释放")
            except Exception:
                self.logger.exception("释放 mss 资源失败")

        self._mss_initialized = False
