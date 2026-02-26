# ScriptChannel Demo

这个例子演示 `script_channel`（stdin/stdout 双工协议）如何用类似 `uvicorn.run(...)` 的方式把**指定的 Python 脚本文件**启动成一个可调用的 MOSS Channel：

- 父进程（proxy）通过 `ScriptChannelProxy` 启动子进程 provider
- 子进程 provider 使用 `ModuleChannel` 将目标脚本包装成 Channel
- 父进程可以像调用本地 command 一样调用子进程脚本导出的函数

注意：因为底层传输复用 stdout 作为协议通道，provider 进程的业务代码不要往 stdout 打印内容；日志请输出到 stderr。


## Run

在仓库根目录执行：

```bash
python examples/script_channel_demo/main.py
```

另外你也可以直接用门面函数（会阻塞运行，类似 `uvicorn.run(...)`）：

```python
from ghoshell_moss.transports import script_channel

script_channel.run("examples/script_channel_demo/target_script.py", name="demo_script")
```

如果希望在阻塞期间还能手动调用命令，可以打开内置 REPL：

```python
from ghoshell_moss.transports import script_channel

script_channel.run(
    "examples/script_channel_demo/target_script.py",
    name="demo_script",
    interactive=True,
)
```
