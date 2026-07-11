"""ProcessManager 内部工具函数 — 跨平台进程组治理."""

import os

__all__ = ["killpg"]


def killpg(process_group: int, sig: int) -> str | None:
    """kill 进程组. 返回 None 表示成功, 返回 str 表示错误.

    平台行为:
    - Linux/macOS: os.killpg — 原子消灭进程组内所有进程
    - Windows: 无进程组概念, 降级为 os.kill 单个 pid.
      返回错误字符串描述降级, 不抛异常.

    主动 setsid / setpgid 脱离的守护进程不受影响.
    """
    if hasattr(os, "killpg"):
        try:
            os.killpg(process_group, sig)
            return None
        except ProcessLookupError:
            return None
        except PermissionError as e:
            return f"killpg({process_group}, {sig}) denied: {e}"
        except OSError as e:
            return f"killpg({process_group}, {sig}) failed: {e}"
    else:
        # Windows: degraded to single-process kill
        try:
            os.kill(process_group, sig)
            return f"killpg degraded: os.killpg unavailable, killed pid={process_group} only"
        except ProcessLookupError:
            return None
        except PermissionError as e:
            return f"killpg({process_group}, {sig}) denied: {e}"
        except OSError as e:
            return f"killpg({process_group}, {sig}) failed: {e}"
