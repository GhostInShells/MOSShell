from enum import Enum

__all__ = ["CommandError", "CommandErrorCode", "FatalError", "InterpretError"]


class FatalError(Exception):
    """
    致命错误, 会导致 Shell 停摆, 状态也需要清空.

    todo: 还没有用起来.
    """

    pass


class InterpretError(Exception):
    """
    解释器解释异常, 是可以恢复的异常.

    todo: 还没有用起来
    """

    pass


class CommandError(Exception):
    """
    Command 运行时异常的封装, 所有的 command 的最佳实践都是用 CommandError 替代原来的 error.
    方便 AI 运行时理解异常.
    """

    def __init__(self, code: int = -1, message: str = ""):
        self.code = code
        self.message = message
        super().__init__(f"Command failed with code `{code}`: {message}")


class CommandErrorCode(int, Enum):
    """
    语法糖, 用来快速生成 command error. 采用了 golang 的语法糖习惯.

    >>> raise CommandErrorCode.CANCELLED.error("error info")

    CommandCode 有特殊的约定习惯.
    < 400 是正常行为逻辑中的异常. 不会中断解释过程.
    >= 400 是不可接受的异常, 会立刻中断 interpreter 的执行逻辑. 并且清空整批规划.

    todo: 参数要重新整理一遍. 缩小到 3位数. 约定百位整数作为基础异常分类.
        需要增加的异常类型:
        - CANCELED:  被各种行为取消了.
        - CLEARED:  被主动清空了. 通常是 shell 和 interpreter 的逻辑.
        - INTERRUPTTED: 被中断了, 从而无法运行.
        第二类是 AI 生成异常:
        - NOT_FOUND: 命令其实不存在.
        - NOT_AVAILABLE: 命名其实无法调用.
        - VALUE_ERROR: 入参不正确
        第三类是链路异常:
        - TIMEOUT_ERROR: 超时
        - DISCONNECTED_ERROR: 通讯中断
        - CLOSED_ERROR: Channel 已经被终止调用了.
        第四类是运行时异常:
        - RUNTIME_ERROR
        - FAILED
        - UNKONW
    """

    SUCCESS = 0
    CANCELLED = 10010

    # todo: 合并重复的参数
    INVALID_USAGE = 40300
    INVALID_PARAMETER = 40100
    VALUE_ERROR = 400000
    NOT_AVAILABLE = 40200
    NOT_FOUND = 40400

    FAILED = 50000
    TIMEOUT = 50010
    UNKNOWN_CODE = -1

    def error(self, message: str) -> CommandError:
        return CommandError(self.value, message)

    @classmethod
    def get_error_code_name(cls, value: int) -> str:
        """将错误代码值映射到对应的枚举名称"""
        try:
            return cls(value).name
        except ValueError:
            # 如果值不在枚举中，返回未知代码的名称
            return cls.UNKNOWN_CODE.name

    @classmethod
    def description(cls, errcode: int, errmsg: str | None = None) -> str:
        if errcode == cls.SUCCESS:
            return "success"
        name = cls.get_error_code_name(errcode)
        return "failed `{}`: {}".format(name, errmsg or "no errmsg")
