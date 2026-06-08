"""Module 求值容器：exec / vars / api | 系统管理 | alpha

将 Python 模块源码包装为 CTML Channel——模型通过 exec 命令直接在模块命名空间中
写代码执行，变量跨调用累积，像 Python REPL。模块源码即 instruction。

依赖 Sandbox 提供安全 exec + ExecutionResult + Reflector 反射。

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


def new_module_eval_channel(
    module_source: str,
    *,
    channel_name: str = "module_eval",
    description: Optional[str] = None,
    sandbox_builtins: dict | None = SANDBOX_BUILTINS,
) -> MutableChannel:
    """创建 ModuleEvalChannel。

    module_source 在 Sandbox 中执行以初始化领域对象，同时作为 instruction 和
    Reflector source——本地对象在 source 中可见，import 对象通过 get_interface()
    反射。

    :param module_source: 模块的 Python 源码，定义领域对象和初始化逻辑
    :param channel_name: Channel 名（CTML 标签名）
    :param description: Channel 描述
    :param sandbox_builtins: 模型执行时的 builtins 白名单，默认屏蔽危险函数
    """
    # 初始化沙盒用完整 builtins，让 module_source 可以 import
    init_sandbox = Sandbox(name=channel_name, builtins=None, source=module_source)
    init_sandbox.exec(module_source)
    # 模型执行沙盒用受限 builtins，共享 init 的命名空间和 source
    sandbox = Sandbox(
        name=channel_name,
        parent=init_sandbox,
        builtins=sandbox_builtins,
        source=module_source,
    )

    desc = description or f"Module eval container — exec code in persistent namespace"
    chan = new_channel(name=channel_name, description=desc)

    @chan.build.instruction
    def instruction() -> str:
        return module_source

    @chan.build.close
    async def cleanup():
        init_sandbox.close()

    @chan.build.command(name="exec", always_observe=True)
    async def exec_code(text__: str) -> str:
        """执行 Python 代码。text__: 在持久化命名空间中执行的完整代码文本。

        代码通过 CTML 开闭标签传入：
            <module_eval:exec>
            x = 1 + 2
            print(x)
            </module_eval:exec>

        print() 输出被捕获返回。异常返回完整 traceback。
        变量跨调用累积——像一个 Python REPL。
        __result__ 赋值可作为返回值。
        """
        result = sandbox.exec(text__)
        parts = []
        if result.std_output:
            parts.append(result.std_output.rstrip())
        if result.exception:
            parts.append(f"Error: {result.exception}")
            if result.traceback:
                parts.append(result.traceback.rstrip())
        if result.returns is not None:
            parts.append(f"__result__: {result.returns!r}")
        return "\n".join(parts) if parts else "(executed, no output)"

    @chan.build.command(name="vars", always_observe=True)
    async def list_vars() -> str:
        """查看命名空间。委托给 Reflector——返回 module source + import 对象反射。

        输出与 moss codex get-interface 一致：source 在前，<attr> 块在后。
        本地对象在 source 中可见，import 对象在 <attr> 块中展开。
        """
        return sandbox.get_interface()

    @chan.build.command(name="api", always_observe=True)
    async def reflect_api(name: str, *methods: str) -> str:
        """反射命名空间中对象的方法签名。name: 对象变量名。*methods: 可选，指定方法名。

        无 methods 时委托给 sandbox.get_interface(name)——import 对象返回完整源码，
        本地对象返回 signature + docstring + 方法列表。
        有 methods 时逐个显示方法签名与 docstring。
        """
        if not methods:
            return sandbox.get_interface(name)

        ns = sandbox.module.__dict__
        if name not in ns:
            return f"'{name}' not found in namespace"

        obj = ns[name]
        obj_type = type(obj).__name__
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
