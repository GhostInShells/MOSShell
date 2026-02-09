# ZMQ Transport

用来验证 Channel 通过 provider->broker 范式进行跨进程通讯, 并且能同步上下文.

测试方式:

1. 确保已经运行 `uv sync --active --all-extras` 完成依赖同步.
1. 确认 python 在 uv 的 venv 环境: `which python`
1. 直接运行 vision_provider: `python ./examples/vision_exam/vision_provider.py`
   - 这会启动 vision channel 的 provider 进程, 监听 5557 端口允许被访问.
   - 会打开一个 opencv 的端口, 可以看到视觉信息.
1. 运行 vision_proxy: `python ./examples/vision_exam/vision_proxy.py`
   - 会与 vision provider 建立双工通讯, 可以控制对方.
   - 打开一个 pyqt6 的 simple viewer, 周期性同步 vision channel 的信息.

当 vision provider 的视觉消息同步给了 vision proxy, 能正确展示图片, 则测试成功.
