from typing import Any, Type, Tuple, List, Dict, TypeVar, Awaitable, List, Callable, Generic, Optional
from ast import literal_eval
from dataclasses import dataclass
from functools import wraps
import inspect

__all__ = [
    'prepare_kwargs_by_signature',
    'parse_function_interface',
    'awaitable_caller',
]


def prepare_kwargs_by_signature(sig: inspect.Signature, args: tuple, kwargs: dict) -> dict:
    """
    parse args and kwargs into a dict of kwargs.
    Written with help from deepseek:v3
    """
    # 绑定参数
    try:
        bound_args = sig.bind(*args, **kwargs)
    except TypeError as e:
        raise TypeError(f"invalid params, args {args}, kwargs {kwargs} for {sig}")

    # 应用默认值
    bound_args.apply_defaults()

    # 可以在这里进行参数类型转换（例如 str -> int）
    for name, value in bound_args.arguments.items():
        param = sig.parameters[name]
        if param.annotation != inspect.Parameter.empty:
            try:
                if param.kind == inspect.Parameter.VAR_POSITIONAL or param.kind == inspect.Parameter.VAR_KEYWORD:
                    continue

                if isinstance(value, str) and param.annotation is not str:
                    if param.annotation is bool:
                        value = value.lower() is 'true'
                    elif param.annotation is dict or param.annotation is list:
                        # 支持 dict 和 list 的 python 风格默认转换.
                        # 理论上 Command Token 的协议需要先设计好转换.
                        value = literal_eval(value)
                bound_args.arguments[name] = param.annotation(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"argument {name} with annotation {param.annotation} is invalid: {e}")
    return bound_args.arguments


@dataclass(frozen=False)
class FunctionReflection:
    """
    Reflection generated from function signature, and can also generate function signature.
    """
    name: str
    signature: inspect.Signature
    docstring: str
    is_coroutine_function: bool
    comments: str

    def prepare_kwargs(self, *args, **kwargs) -> Dict[str, Any]:
        return prepare_kwargs_by_signature(self.signature, args, kwargs)

    def to_interface(self, name: str = "", doc: str = "", comments: str = "") -> str:
        def_syntax = "async def" if self.is_coroutine_function else "def"
        indent = " " * 4
        name = name or self.name
        docstring = doc or self.docstring
        comments = comments or self.comments

        sig = str(self.signature)
        definition = f"{def_syntax} {name}{sig}:"
        lines = [definition]
        if docstring:
            docstring_lines = to_function_docstring_lines(docstring)
            for line in docstring_lines:
                lines.append(indent + line)

        if comments:
            for comment_line in comments.split("\n"):
                lines.append(indent + '# ' + comment_line)
        lines.append(indent + "pass")
        return "\n".join(lines)


def to_function_docstring_lines(doc: str) -> List[str]:
    """
    将一个字符串变成函数的 docstring 形式的文本块. 并且添加上必要的 indent.
    """
    quote = "'''"
    replace_quote = "\\" + quote  # 转义后的三引号：`\'''`
    doc_lines = doc.split('\n')
    result_lines = [quote]  # 开始 docstring
    for line in doc_lines:
        stripped = line.strip()
        if not stripped:
            continue  # 跳过空行
        # 将字符串中的 `'''` 替换为 `\'''`
        replaced = line.replace(quote, replace_quote)
        result_lines.append(replaced)
    result_lines.append(quote)  # 结束 docstring
    return result_lines


def parse_function_interface(fn: Callable) -> FunctionReflection:
    name = fn.__name__
    sig = inspect.Signature.from_callable(fn)
    is_cofunc = inspect.iscoroutinefunction(fn)
    docstring = inspect.getdoc(fn)
    return FunctionReflection(
        name=name,
        signature=sig,
        is_coroutine_function=is_cofunc,
        docstring=docstring,
        comments="",
    )


R = TypeVar('R')


def unwrap_callable_or_value(func: Callable[[], R] | R) -> R:
    if callable(func):
        return func()
    return func


def awaitable_caller(
        fn: Callable[..., R] | Callable[..., Awaitable[R]] | R,
        *,
        default: Optional[Any] = None,
) -> Callable[..., Awaitable[R]]:
    if not callable(fn):
        async def return_result(*args, **kwargs):
            return fn if fn is not None else default  # as result

        return return_result

    if inspect.iscoroutinefunction(fn):
        return fn

    @wraps(fn)
    async def wrapper(*args, **kwargs):
        r = fn(*args, **kwargs)
        if inspect.isawaitable(r):
            r = await r
        return r

    return wrapper
