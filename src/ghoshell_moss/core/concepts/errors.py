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
    """

    # AI 需要感知到的普通运行结果.
    SUCCESS = 0
    # 最常用的异常方式, 建议用它包装所有的 AI 需要感知的异常.
    FAILED = 100

    # --- 不需要立刻响应, 而且 AI 也不需要关心的异常. 通常是系统调度结果. --- #

    # 命令被取消.
    CANCELLED = 200
    # 命令被清空.
    CLEARED = 201
    # 命令超时被设置失败.
    TIMEOUT = 202
    # 命令被中断.
    INTERRUPTED = 203

    # --- 不合法的异常, 需要 AI 立刻去响应. --- #

    # 不合法的使用时机.
    INVALID_USAGE = 401
    # 参数不正确.
    VALUE_ERROR = 402
    # 命令不可用.
    NOT_AVAILABLE = 403
    # 命令不存在.
    NOT_FOUND = 404
    # channel 没有启动.
    NOT_RUNNING = 405
    # channel 未连接.
    NOT_CONNECTED = 406

    # --- 命令执行不可接受的异常 --- #
    # 对于 AI 而言必须要立刻感知的致命异常.
    FATAL = 500
    UNKNOWN_ERROR = 505

    def error(self, message: str) -> CommandError:
        return CommandError(self.value, message)

    @classmethod
    def interpretation_fatal(cls, err: Exception) -> bool:
        if err is None:
            return False
        if not isinstance(err, CommandError):
            return True
        # 400 以上的异常对解释流程是致命的.
        return err.code >= 400

    def match(self, error: Exception | None) -> bool:
        if not error:
            return False
        if not isinstance(error, CommandError):
            return False
        return error.code == self.value

    @classmethod
    def get_error_code_name(cls, value: int) -> str:
        """将错误代码值映射到对应的枚举名称"""
        try:
            return cls(value).name
        except ValueError:
            # 如果值不在枚举中，返回未知代码的名称
            return cls.UNKNOWN_ERROR.name

    @classmethod
    def description(cls, errcode: int, errmsg: str | None = None) -> str:
        if errcode == cls.SUCCESS:
            return "success"
        name = cls.get_error_code_name(errcode)
        return "failed `{}`: {}".format(name, errmsg or "no errmsg")
