# ScriptChannelHub Demo

这个例子演示如何使用 `ScriptChannelHub` 来管理多个“脚本通道”（每个子通道在独立子进程里运行）。

关键点：

- `ScriptChannelHub` 本身不直接手动启动子进程
- 每个子通道都是一个 `ScriptChannelProxy`，在它的 runtime 启动时自动拉起 provider 子进程

## Run

在仓库根目录执行：

```bash
python examples/script_channel_hub_demo/main.py
```
