import asyncio
from typing import Any, Optional

from ghoshell_moss.core import PyChannel

__all__ = ["new_mac_control_channel"]


class JXAError(Exception):
    """JXA 执行错误"""

    pass


async def run(*, timeout: Optional[float] = 30.0, text__: str = "") -> Any:
    """
    在当前 MAC 上执行一个 JXA 函数, 子进程阻塞到执行完毕.
    如果你需要与 mac 应用做交互, 又没有别的手段时, 可以用这个办法.

    :param timeout: 过期时间.
    :param text__: jxa 脚本的源码.

    CTML 使用举例:
        <chan_name:run><![CDATA[
        (function() {
            'use strict';

            // 1. 使用 Bundle Identifier 获取日历应用对象
            var Calendar = Application('com.apple.iCal');

            // 2. 检查是否正在运行
            var wasRunning = Calendar.running();

            // 3. 尝试激活 (如果没有运行，这会启动它)
            Calendar.activate();

            // 4. 等待一小会儿，让系统处理激活请求
            delay(0.5); // JXA 中的 delay 函数，单位秒

            // 5. 再次检查运行状态和窗口状态
            var isRunning = Calendar.running();
            var windowCount = Calendar.windows ? Calendar.windows.length : 0;

            // 6. 返回结构化的调试信息
            return {
                success: true,
                action: 'activate_calendar',
                bundleId: 'com.apple.iCal',
                wasRunning: wasRunning,
                isRunning: isRunning,
                windowCount: windowCount,
                timestamp: Date.now()
            };
        })();
        ]]></chan_name:run>

    :return: 返回操作结果. 你必须等操作结果到手后, 才能知道它运行的效果如何.
    """
    cmd = ["osascript", "-l", "JavaScript", "-"]

    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(input=text__.encode("utf-8")), timeout=timeout)

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            raise JXAError(f"JXA 执行失败 (code: {process.returncode}): {error_msg}")

        output = stdout.decode("utf-8", errors="replace").strip()

        # 尝试解析 JSON
        return output

    except asyncio.TimeoutError:
        if process and process.returncode is None:
            process.kill()
            await process.wait()
        raise


def new_mac_control_channel(
    name: str = "mac_control",
    description: str = "使用 jxa 语法来操作当前所在 mac",
) -> PyChannel:
    """
    创建一个控制 mac 的 channel.
    """
    mac_jxa_channel = PyChannel(
        name=name,
        description=description,
        blocking=True,
    )

    mac_jxa_channel.build.command()(run)
    return mac_jxa_channel
