# ScriptChannel Transport (Vision Demo)

这个示例复刻 `examples/vision_exam/` 的行为，但把底层的传输从 `ZMQChannelProvider/ZMQChannelProxy`
替换为 `ScriptChannelProvider/ScriptChannelProxy`（stdio 子进程双工协议）。

与 ZMQ 版本的关键差异：

- provider 进程不再通过 TCP 端口提供服务，而是由 proxy 脚本**直接拉起子进程**并通过 stdin/stdout 通信。

## Run

1. 确保依赖已安装：`uv sync --active --all-extras`
1. 运行 proxy（它会自动启动 provider 子进程）：

```bash
python examples/vision_exam_script_channel/vision_proxy.py
```

成功标志：

- 会弹出 OpenCV 的摄像头窗口（按 `q` 或关闭窗口退出）
- 会弹出一个 PyQt 的图片 viewer，周期性显示 provider 同步过来的最新画面
