import inspect
from typing import Dict, Any, Iterable, Optional, TypedDict
from ghoshell_moss.core.helpers.func import parse_function_interface
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

__all__ = ['ToolRegistrar']


class Metadata(TypedDict):
    name: str
    sig: inspect.Signature
    doc: str
    help: str
    obj: Any
    interface: str | None


class ToolRegistrar(Completer):
    def __init__(
            self,
            tool_objects: Dict[str, Any],
            *,
            command_mark: str = '/',
            help_mark: str = '?',
    ):
        self._tool_objects = tool_objects
        self._tool_objects_docs = {
            name: inspect.getdoc(type(obj)) or 'No doc'
            for name, obj in self._tool_objects.items()
        }
        self._command_mark = command_mark
        self._help_mark = help_mark
        # 缓存结构: { "robot.arm": { "move": Metadata(...), ... } }
        self._metadata_cache: Dict[str, Dict[str, Metadata]] = {}
        self._build_cache()

    def _build_cache(self):
        """递归扫描所有对象及其方法，构建补全与提示字典"""

        def _scan(obj, path):
            self._metadata_cache[path] = {}
            for name in dir(obj):
                if name.startswith('_'): continue
                try:
                    attr = getattr(obj, name)
                except Exception:
                    continue

                # 记录该成员信息
                is_method = inspect.ismethod(attr)
                sig = inspect.signature(attr) if is_method else None
                interface = None
                if sig:
                    func_itf = parse_function_interface(attr)
                    interface = func_itf.to_interface()
                doc = inspect.getdoc(attr) or "No doc"

                self._metadata_cache[path][name] = Metadata(
                    name=name,
                    sig=sig,
                    doc=doc,
                    obj=attr,
                    help=doc.splitlines()[0],
                    interface=interface,
                )
                # 如果是对象，继续递归 (限制深度防止死循环)
                if not inspect.ismethod(attr) and not isinstance(attr, (int, str, float, bool)):
                    _scan(attr, f"{path}.{name}")

        for name, obj in self._tool_objects.items():
            _scan(obj, name)

    def _lookup_by_path(self, path: str) -> Optional[Metadata | dict]:
        """根据路径字符串查找缓存中的元数据"""
        parts = path.split('.')
        parent = ".".join(parts[:-1]) if len(parts) > 1 else None
        name = parts[-1]

        # 查找逻辑
        if parent is None:
            # 可能是根对象
            if name in self._metadata_cache:
                # 注意：根对象需要特殊处理以获取 doc
                return {
                    'name': name,
                    'sig': None,
                    'doc': inspect.getdoc(self._tool_objects[name]),
                    'obj': self._tool_objects[name],
                }
        elif parent in self._metadata_cache:
            return self._metadata_cache[parent].get(name)
        return None

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        # 1. 拦截 '?' 帮助请求
        if self._help_mark and text.startswith(self._help_mark):
            path = text[len(self._help_mark):].strip()
            yield from self._get_help_completions(path)

        if self._command_mark and text.startswith(self._command_mark):
            text = text[len(self._command_mark):]
            yield from self._get_command_completions(text)
        return

    def _get_help_completions(self, text: str) -> Iterable[Completion]:
        # 2. 路径/方法补全模式
        parts = text.split('.')
        prefix = parts[-1]
        parent_path = ".".join(parts[:-1]) if len(parts) > 1 else None

        # 补全根节点
        if parent_path is None:
            for name, doc in self._tool_objects_docs.items():
                if name.startswith(prefix):
                    yield Completion(
                        name,
                        start_position=-len(prefix),
                        display_meta=doc.splitlines()[0] if doc else '',
                    )
            return

        # 补全嵌套节点
        if parent_path in self._metadata_cache:
            for name, meta in self._metadata_cache[parent_path].items():
                if name.startswith(prefix):
                    display = f"{name}{meta['sig']}" if meta['sig'] else name
                    yield Completion(
                        name,
                        start_position=-len(prefix),
                        display=display,
                        display_meta=meta['help'],
                    )

    def _get_command_completions(self, text: str) -> Iterable[Completion]:

        # 1. 参数补全模式
        if "(" in text and (text.rstrip()[-1] in (',', '(')):
            func_path = text.split('(')[0]
            parts = func_path.split('.')
            parent_path = ".".join(parts[:-1])
            func_name = parts[-1]

            # 从 cache 中获取该函数签名
            meta = self._metadata_cache.get(parent_path, {}).get(func_name)
            if meta and meta['sig']:
                # 获取已输入的参数，避免重复补全
                args_part = text.split('(')[-1]
                existing = {p.split('=')[0].strip() for p in args_part.split(',') if '=' in p}

                for p_name, param in meta['sig'].parameters.items():
                    if p_name not in existing:
                        yield Completion(
                            f"{p_name}=",
                            display=p_name,
                            display_meta=str(param.annotation),
                        )
            return

        # 2. 路径/方法补全模式
        parts = text.split('.')
        prefix = parts[-1]
        parent_path = ".".join(parts[:-1]) if len(parts) > 1 else None

        # 补全根节点
        if parent_path is None:
            for name, doc in self._tool_objects_docs.items():
                if name.startswith(prefix):
                    yield Completion(
                        name,
                        start_position=-len(prefix),
                        display_meta=doc.splitlines()[0] if doc else '',
                    )
            return

        # 补全嵌套节点
        if parent_path in self._metadata_cache:
            for name, meta in self._metadata_cache[parent_path].items():
                if name.startswith(prefix):
                    display = f"{name}{meta['sig']}" if meta['sig'] else name
                    yield Completion(
                        name,
                        start_position=-len(prefix),
                        display=display,
                        display_meta=meta['help'],
                    )

    def is_command(self, line: str) -> bool:
        return line.startswith(self._command_mark)

    def is_help(self, line: str) -> bool:
        return line.startswith(self._help_mark)

    def eval_input(self, line: str) -> Any:
        """
        执行输入命令，支持嵌套属性路径及函数调用
        """
        if self._help_mark and line.startswith(self._help_mark):
            obj = self._lookup_by_path(line[len(self._help_mark):])
            if obj is None:
                raise ValueError(f'help for `{line}` not found')
            elif interface := obj.get('interface'):
                return interface
            elif sig := obj.get('sig'):
                return (
                    f"\ndef {obj['name']}({sig}):\n"
                    f"    {obj['doc']}"
                )
            else:
                return obj.get('doc', 'No doc')
        elif self._command_mark and not line.startswith(self._command_mark):
            raise ValueError(f'`{line}` is not a command, need start with `{self._command_mark}`')

        # 去除前缀
        cmd = line.strip().lstrip(self._command_mark)
        found = self._lookup_by_path(cmd.split('(', 1)[0])
        if found is None:
            raise ValueError(f'Command for `{line}` not found')

        # 定义执行环境
        # 这里只允许访问传入的 tool_objects，且禁用 __builtins__
        allowed_globals = {"__builtins__": None}
        allowed_locals = self._tool_objects

        try:
            # 使用 eval 执行表达式
            # 这种方式支持完整的 Python 表达式语法，例如:
            # /robot.arm.move(10, 20)
            # /robot.say("test")
            return eval(cmd, allowed_globals, allowed_locals)

        except SyntaxError as e:
            raise ValueError(f"Syntax Error: {e.msg}")
        except AttributeError as e:
            raise ValueError(f"Attribute Error: {e}")
        except Exception as e:
            raise Exception(f"Eval failed: {str(e)}")


if __name__ == "__main__":
    import asyncio
    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout
    import asyncio


    class Arm:
        """arm doc"""

        def move(self, x: int, y: int):
            """move doc"""
            return f"Arm moved to ({x}, {y})"


    class Robot:
        """robot doc"""

        def __init__(self):
            self.arm = Arm()

        async def say(self, message: str):
            """ say doc"""
            await asyncio.sleep(0.1)  # 模拟异步
            return f"Robot says: {message}"


    # 实例化对象
    _robot = Robot()


    async def run_interactive_debugger(registrar: ToolRegistrar):
        # 初始化 PromptSession，直接注入 Registrar 作为 Completer
        session = PromptSession(completer=registrar)
        print("--- MOSS Debugger Loaded ---")
        print("Usage example: /robot.arm.move(10, 20) or /robot.say('Hello')")
        print("Type 'exit' to quit.\n")

        while True:
            try:
                # 使用 patch_stdout 确保即使在异步任务中 print 也能正常对齐
                with patch_stdout():
                    line = await session.prompt_async("moss> ")

                line = line.strip()
                if line in ['exit', 'quit']:
                    break
                if not line:
                    continue

                # 执行输入
                try:
                    result = registrar.eval_input(line)

                    # 统一处理同步值与异步协程
                    if asyncio.iscoroutine(result):
                        result = await result

                    if result is not None:
                        print(f"Result: {result}")
                except Exception as e:
                    print(f"Execution Error: {e}")

            except (EOFError, KeyboardInterrupt):
                break


    # 运行测试
    _arm = Arm()
    registrar = ToolRegistrar(tool_objects={"robot": _robot, "arm": _arm}, command_mark="/")

    asyncio.run(run_interactive_debugger(registrar))
