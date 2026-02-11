import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple
from PIL import Image
import cv2

from ghoshell_moss.message import Message, Base64Image, Text
from ghoshell_common.contracts import LoggerItf
from ghoshell_container import IoCContainer, get_container
from ghoshell_moss import PyChannel
from ghoshell_moss.transports.zmq_channel.zmq_channel import ZMQChannelProvider
import logging


class OpenCVVision:
    """基于OpenCV的视觉感知Channel，提供实时图像上下文给模型"""

    def __init__(self, container: Optional[IoCContainer] = None):
        self._container = container or get_container()
        self._logger: LoggerItf = container.get(LoggerItf) if container else logging.getLogger(__name__)

        # 线程安全的数据结构
        self._cached_image_lock = threading.Lock()
        self._cached_image: Optional[Image.Image] = None
        self._cached_timestamp: float = 0.0

        # 控制标志
        self._is_caching_image = True
        self._is_running = False

        # OpenCV相关
        self._cap: Optional[cv2.VideoCapture] = None
        self._window_name = "MOSShell Vision"

        # 性能参数
        self._target_fps = 2  # 目标帧率：2帧/秒
        self._frame_interval = 1.0 / self._target_fps
        self._last_capture_time = 0.0

        # 错误处理
        self._last_error: Optional[str] = None
        self._error_count = 0
        self._max_errors = 10

    def _initialize_camera(self) -> bool:
        """初始化摄像头，返回是否成功"""
        try:
            # 尝试不同的摄像头索引
            for camera_index in [0, 1, 2]:
                self._cap = cv2.VideoCapture(camera_index)
                if self._cap.isOpened():
                    # 设置合适的分辨率以提高性能
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self._cap.set(cv2.CAP_PROP_FPS, 30)

                    # 测试读取一帧
                    ret, _ = self._cap.read()
                    if ret:
                        self._logger.info(f"摄像头初始化成功，使用索引 {camera_index}")
                        return True
                    else:
                        self._cap.release()

            self._logger.error("无法初始化任何摄像头")
            return False

        except Exception as e:
            self._logger.error(f"摄像头初始化失败: {e}")
            return False

    def _capture_frame(self) -> bool:
        """捕获一帧图像并更新缓存，返回是否成功"""
        if self._cap is None or not self._cap.isOpened():
            return False

        ret, frame = self._cap.read()
        if not ret:
            self._error_count += 1
            if self._error_count >= self._max_errors:
                self._last_error = "连续读取帧失败"
                self._logger.error(self._last_error)
            return False

        # 重置错误计数
        self._error_count = 0

        # 显示实时画面
        cv2.imshow(self._window_name, frame)

        # 处理按键（'q'退出）
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._logger.info("用户按q键，视觉模块退出")
            return False

        current_time = time.time()

        # 按目标帧率更新缓存
        if self._is_caching_image and current_time - self._last_capture_time >= self._frame_interval:
            # 转换颜色空间：BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 转换为PIL Image
            pil_image = Image.fromarray(rgb_frame)

            # 线程安全地更新缓存
            with self._cached_image_lock:
                self._cached_image = pil_image
                self._cached_timestamp = current_time

            self._last_capture_time = current_time

        return True

    def _cleanup_camera(self) -> None:
        """清理摄像头资源"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()

    def start_capture(self) -> bool:
        """启动视觉捕获"""
        if self._is_running:
            self._logger.warning("视觉捕获已在运行中")
            return True

        if not self._initialize_camera():
            self._logger.error("视觉模块初始化失败")
            return False

        # 创建显示窗口
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, 640, 480)

        self._is_running = True
        self._logger.info("视觉模块已启动")
        return True

    def close(self) -> None:
        """关闭视觉模块"""
        self._logger.info("正在关闭视觉模块...")
        self._is_running = False

        # 清理摄像头资源
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        # 确保窗口被销毁
        try:
            cv2.destroyWindow(self._window_name)
        except:
            pass

        # 最后调用 destroyAllWindows 确保清理所有窗口
        cv2.destroyAllWindows()

        self._logger.info("视觉模块已关闭")

    def get_cached_image(self) -> Tuple[Optional[Image.Image], float]:
        """
        线程安全地获取缓存的图像和时间戳

        返回:
            (图像, 时间戳) 元组，如果没有缓存图像则返回 (None, 0.0)
        """
        with self._cached_image_lock:
            if self._cached_image is not None:
                # 创建图像的副本，避免线程间共享可变对象
                image_copy = self._cached_image.copy()
                return image_copy, self._cached_timestamp
            return None, 0.0

    async def stop_looking(self) -> None:
        """
        关闭视觉，停止缓存新图像并清空当前缓存
        """
        self._logger.info("模型命令：关闭视觉")
        self._is_caching_image = False

        # 清空缓存
        with self._cached_image_lock:
            self._cached_image = None
            self._cached_timestamp = 0.0

    async def start_looking(self) -> None:
        """
        开启视觉，开始缓存新图像
        """
        self._logger.info("模型命令：开启视觉")
        self._is_caching_image = True
        self._last_capture_time = 0.0  # 重置时间，立即捕获下一帧

    async def context_messages(self) -> List[Message]:
        """
        返回最新的视觉信息作为上下文消息

        注意：这个方法在模型每次思考时被调用，应该快速返回
        """
        # 如果视觉未开启，返回空列表
        if not self._is_caching_image:
            return []

        # 获取缓存的图像
        image, timestamp = self.get_cached_image()

        if image is None:
            # 如果有错误信息，可以返回错误提示（可选）
            if self._last_error:
                error_msg = Message.new(role="system", name="__vision_error__").with_content(
                    Text(text=f"视觉模块错误: {self._last_error}")
                )
                return [error_msg]
            return []

        # 创建视觉消息
        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%d.%m.%Y %H:%M:%S")

        message = Message.new(role="user", name="__vision_system__").with_content(
            Text(text=f"这是你最新看到的视觉信息，来自你的视野。时间: {timestamp_str}"),
            Base64Image.from_pil_image(image),
        )

        return [message]

    def description(self) -> str:
        status = "已开启视觉" if self._is_caching_image else "视觉已经关闭."
        desc = f"基于OpenCV的视觉感知模块，提供实时图像输入. 当前状态: {status}"
        return desc

    def as_channel(self) -> PyChannel:
        """将视觉模块包装为MOSShell Channel"""
        _channel = PyChannel(
            name="vision",
            description="基于OpenCV的视觉感知模块，提供实时图像输入",
            block=True,  # 这是一个非阻塞的感知Channel
        )

        # 注册上下文消息生成器
        _channel.build.with_context_messages(self.context_messages)
        _channel.build.with_description()(self.description)

        # 注册控制命令
        _channel.build.command()(self.start_looking)
        _channel.build.command()(self.stop_looking)

        # 添加状态查询命令（可选，用于调试）
        @_channel.build.command()
        async def vision_status() -> str:
            """查询视觉模块的当前状态"""
            image, timestamp = self.get_cached_image()
            status = {
                "is_running": self._is_running,
                "is_caching": self._is_caching_image,
                "has_cached_image": image is not None,
                "last_image_time": datetime.fromtimestamp(timestamp).strftime("%H:%M:%S") if timestamp > 0 else "无",
                "error": self._last_error or "无",
            }
            return f"视觉模块状态: {status}"

        return _channel

    def __del__(self):
        """析构函数确保资源释放"""
        if self._is_running:
            self.close()

    def run_opencv_loop(self) -> None:
        """
        启动视觉模块的主循环

        注意：这个方法会阻塞，直到视觉模块被关闭
        """
        if not self.start_capture():
            self._logger.error("无法启动视觉捕获")
            return

        self._logger.info("视觉模块已启动，按Ctrl+C或窗口内按'q'键退出")

        try:
            # 主循环：捕获、显示、处理事件
            while self._is_running:
                if not self._capture_frame():
                    self._is_running = False  # 确保状态同步
                    break

                # 检查窗口是否被用户关闭（通过关闭按钮）
                try:
                    # 如果窗口不存在，getWindowProperty 会返回 -1
                    if cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
                        self._logger.info("用户关闭了视觉窗口")
                        self._is_running = False
                        break
                except:
                    # 窗口可能已经被销毁
                    self._is_running = False
                    break

        except KeyboardInterrupt:
            self._logger.info("收到键盘中断信号")
        except Exception as e:
            self._logger.error(f"视觉模块运行异常: {e}")
        finally:
            self.close()
