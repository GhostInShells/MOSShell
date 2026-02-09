from ghoshell_moss import get_container
from ghoshell_moss.transports.zmq_channel import ZMQChannelProvider
from ghoshell_moss_contrib.channels.opencv_vision import OpenCVVision

if __name__ == "__main__":
    # 初始化容器
    _container = get_container()

    # 创建视觉模块
    vision = OpenCVVision(_container)

    # 创建ZMQ Channel Provider
    provider = ZMQChannelProvider(
        address="tcp://localhost:5557",
        container=_container,
    )

    try:
        # 将视觉模块包装为Channel并运行
        channel = vision.as_channel()
        provider.run_in_thread(channel)

        # 启动视觉捕获循环（会阻塞）
        vision.run_opencv_loop()

    finally:
        # 确保清理顺序正确
        vision.close()
        provider.close()
        provider.wait_closed_sync()
