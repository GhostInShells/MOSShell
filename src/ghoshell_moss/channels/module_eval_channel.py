"""Module 求值容器：exec / vars / api | 系统管理 | alpha

将 Python 模块源码包装为 CTML Channel——模型通过 exec 命令直接在模块命名空间中
写代码执行，变量跨调用累积，像 Python REPL。模块源码即 instruction。

依赖 Sandbox 提供安全 exec + stdout 捕获 + 命名空间管理。

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.module_eval_channel import new_module_eval_channel

    module_source = '''
    from collections import Counter
    data = Counter(["a", "b", "a"])
    '''

    main = new_shell_main_channel()
    main.import_channels(new_module_eval_channel(module_source, channel_name="counter"))
"""

import inspect
from typing import Optional

from ghoshell_moss.core.codex.sandbox import Sandbox, SANDBOX_BUILTINS
from ghoshell_moss.core.blueprint.channel_builder import new_channel, MutableChannel

__all__ = ["new_module_eval_channel"]


def _summarize_value(val: object) -> str:
    """简短描述值：类型 + 截断的 repr。"""
    type_name = type(val).__name__
    raw = repr(val)
    if len(raw) > 120:
        raw = raw[:117] + "..."
    return f"{type_name}: {raw}"


def new_module_eval_channel(
    module_source: str,
    *,
    channel_name: str = "module_eval",
    description: Optional[str] = None,
    sandbox_builtins: dict | None = SANDBOX_BUILTINS,
) -> MutableChannel:
    """创建 ModuleEvalChannel。

    module_source 在 Sandbox 中执行以初始化领域对象，同时作为 Channel 的 instruction
    直接展示给模型——模型看到模块源码即知道有哪些对象和 API。

    :param module_source: 模块的 Python 源码，定义领域对象和初始化逻辑
    :param channel_name: Channel 名（CTML 标签名）
    :param description: Channel 描述
    :param sandbox_builtins: Sandbox 的 builtins 白名单，默认屏蔽危险函数
    """
    # 初始化沙盒用完整 builtins，让 module_source 可以 import
    init_sandbox = Sandbox(name=channel_name, builtins=None)
    init_sandbox.exec(module_source)
    # 模型执行沙盒用受限 builtins，共享 init 的命名空间
    sandbox = Sandbox(name=channel_name, parent=init_sandbox, builtins=sandbox_builtins)

    desc = description or f"Module eval container — exec code in persistent namespace"
    chan = new_channel(name=channel_name, description=desc)

    @chan.build.instruction
    def instruction() -> str:
        return module_source

    @chan.build.close
    async def cleanup():
        sandbox.close()

    @chan.build.command(name="exec", always_observe=True)
    async def exec_code(text__: str) -> str:
        """执行 Python 代码。text__: 在持久化命名空间中执行的完整代码文本。

        代码通过 CTML 开闭标签传入：
            <module_eval:exec>
            x = 1 + 2
            print(x)
            </module_eval:exec>

        print() 输出被捕获返回。变量跨调用累积——像一个 Python REPL。
        抛异常时返回完整 traceback，命名空间状态保留。
        """
        output = sandbox.exec(text__)
        return output.rstrip() if output else "(executed, no output)"

    @chan.build.command(name="vars", always_observe=True)
    async def list_vars(*names: str) -> str:
        """查看命名空间中的变量。*names: 可选，指定变量名；无参数时列出所有公共变量名。

        无参数列出所有不以 '_' 开头的变量名。
        有参数时逐个显示类型与截断的 repr 值。
        """
        ns = sandbox._module.__dict__
        if not names:
            public = sorted(k for k in ns if not k.startswith("_"))
            return "\n".join(public) if public else "(empty namespace)"

        lines = []
        for name in names:
            if name in ns:
                lines.append(f"{name}: {_summarize_value(ns[name])}")
            else:
                lines.append(f"{name}: not found")
        return "\n".join(lines)

    @chan.build.command(name="api", always_observe=True)
    async def reflect_api(name: str, *methods: str) -> str:
        """反射命名空间中对象的方法签名。name: 对象变量名。*methods: 可选，指定方法名。

        无 methods 时列出对象的所有公开可调用方法。
        有 methods 时逐个显示方法签名与 docstring。
        """
        ns = sandbox._module.__dict__
        if name not in ns:
            return f"'{name}' not found in namespace"

        obj = ns[name]
        obj_type = type(obj).__name__

        if not methods:
            public = sorted(
                m for m in dir(obj)
                if not m.startswith("_") and callable(getattr(obj, m, None))
            )
            header = f"Public methods of '{name}' ({obj_type}):"
            return header + "\n" + "\n".join(f"  {m}" for m in public) if public else header + " (none)"

        lines = [f"'{name}' ({obj_type}):"]
        for method_name in methods:
            method = getattr(obj, method_name, None)
            if method is None:
                lines.append(f"  {method_name}: not found")
            elif not callable(method):
                lines.append(f"  {method_name}: not callable ({type(method).__name__})")
            else:
                try:
                    sig = str(inspect.signature(method))
                except (ValueError, TypeError):
                    sig = "(...)"

                doc = inspect.getdoc(method)
                lines.append(f"  {method_name}{sig}")
                if doc:
                    lines.append(f"    {doc}")
        return "\n".join(lines)

    return chan
