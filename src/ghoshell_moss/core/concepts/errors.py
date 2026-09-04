"""
MOSS 架构中可复用的异常类型, 主要是 CommandError
"""

from enum import IntEnum

from ghoshell_container import get_caller_info
from typing_extensions import Self

__all__ = [
    "CommandError", "CommandErrorCode", "FatalError", "InterpretError", 'PausedError',
]


class FatalError(Exception):
    """
    致命错误, 会导致 Shell 停摆, 状态也需要清空.

    todo: 还没有用起来.
    """

    pass


class CommandError(Exception):
    """
    Command 运行时异常的封装, 所有的 command 的最佳实践都是用 CommandError 替代原来的 error.
    方便 AI 运行时理解异常.
    """

    def __init__(self, code: int = -1, message: str = "", at_line: str = "", error_name: str = '') -> None:
        self.code = code
        self.message = message
        self.at_line = at_line or get_caller_info(2)
        error_msg = CommandErrorCode.description(code, message, error_name=error_name)
        super().__init__(error_msg)

    def __repr__(self):
        return f"{self} at {self.at_line}"

    @classmethod
    def from_error(cls, err: Exception) -> Self:
        import asyncio
        if err is None or not isinstance(err, Exception):
            errcode = CommandErrorCode.UNKNOWN_ERROR.value
            errmsg = f"raise error from invalid type {type(err)}"

        elif isinstance(err, CommandError):
            errcode = err.code
            errmsg = err.message
        elif isinstance(err, InterpretError):
            # 识别解释器异常, 保留 INTERPRET_ERROR 语义而不降级为 UNKNOWN_ERROR.
            errcode = CommandErrorCode.INTERPRET_ERROR.value
            errmsg = err.message
        elif isinstance(err, asyncio.CancelledError):
            errcode = CommandErrorCode.CANCELLED.value
            errmsg = ""
        elif isinstance(err, asyncio.TimeoutError):
            errcode = CommandErrorCode.TIMEOUT.value
            errmsg = ""
        elif isinstance(err, AttributeError):
            errcode = CommandErrorCode.INVALID_USAGE.value
            errmsg = ""
        elif isinstance(err, Exception):
            errcode = CommandErrorCode.UNKNOWN_ERROR.value
            # 忽略回调.
            errmsg = str(err)
        else:
            errcode = CommandErrorCode.UNKNOWN_ERROR.value
            errmsg = str(err)
        return cls(errcode, errmsg)


class InterpretError(Exception):
    """
    解释器解释异常, 是可以恢复的异常.

    恒以 ``CommandErrorCode.INTERPRET_ERROR`` 归口 (code = 407):
    跨包转换 (CommandError.from_error / CommandTask.fail) 时, 若直接对
    InterpretError 走通用 Exception 分支, 会被降级成 UNKNOWN_ERROR (505),
    丢语义. 因此这里显式携带 code + 归一化的原始 message (不含 code 前缀),
    供转换处无损读取.
    """

    def __init__(self, message: str | Exception = ""):
        # code 恒为 INTERPRET_ERROR: 解释器解析异常统一归口, 不随内层来源散失.
        # (CommandErrorCode 定义在模块后方, 作类属性默认值会 NameError, 故用实例属性.)
        self.code: int = CommandErrorCode.INTERPRET_ERROR.value
        if isinstance(message, InterpretError):
            self.message = message.message
        elif isinstance(message, CommandError):
            # 保留内层原始 message; code 恒为 INTERPRET_ERROR.
            self.message = message.message
        elif isinstance(message, Exception):
            self.message = str(message)
        else:
            self.message = message
        # 保持 str(error) 稳定为 "INTERPRET_ERROR: <message>".
        super().__init__(CommandErrorCode.description(self.code, self.message))

    @classmethod
    def from_error(cls, err: Exception) -> Self:
        error = CommandError.from_error(err)
        return cls(error)


class PausedError(Exception):
    """
    system is paused
    """
    pass


class CommandErrorCode(IntEnum):
    """
    语法糖, 用来快速生成 command error. 采用了 golang 的语法糖习惯.

    >>> raise CommandErrorCode.CANCELLED.error("error info")

    CommandCode 有特殊的约定习惯.
    < 400 是正常行为逻辑中的异常. 不会中断解释过程.
    >= 400 是不可接受的异常, 会立刻中断 interpreter 的执行逻辑. 并且清空整批规划.
    """

    # AI 需要感知到的普通运行结果.
    SUCCESS = 0

    # --- 不需要立刻响应, 而且 AI 也不需要关心的异常. 通常是系统调度结果. --- #

    # 命令被取消.
    CANCELLED = 200
    # 命令被清空.
    CLEARED = 201
    # 命令超时被设置失败.
    TIMEOUT = 202
    # 命令被中断.
    INTERRUPTED = 203

    # --- 需要 AI 感知的异常. --- #
    FAILED = 300
    # 命令被取消/中断但携带可读的进度信息 (如"播放了 n 秒, 停在 xxx").
    # is_notifiable 会把它记录成可读 message, 但 code < 400 不触发 observe 也不中断.
    STOPPED = 301

    # --- 不合法的异常, 需要 AI 立刻去响应. --- #

    # 返回值实际上是 OBSERVE 动作, 仍然用 error 来通知.
    OBSERVE = 400

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
    INTERPRET_ERROR = 407

    # --- 命令执行不可接受的异常 --- #
    # 对于 AI 而言必须要立刻感知的致命异常.
    CRITICAL = 500
    UNKNOWN_ERROR = 505
    FATAL = 600

    def error(self, message: str, error_name: str = '') -> CommandError:
        at_line = get_caller_info(2)
        return CommandError(self.value, message, at_line=at_line, error_name=error_name)

    @classmethod
    def is_cancelled(cls, err: Exception | int) -> bool:
        if err is None:
            return False
        if isinstance(err, Exception):
            if not isinstance(err, CommandError):
                return False
            code = err.code
        elif isinstance(err, int):
            code = err
        else:
            return False
        return 200 <= code < 300

    @classmethod
    def is_notifiable(cls, err: Exception | int) -> bool:
        """需要被通知的异常."""
        if err is None:
            return False
        if isinstance(err, Exception):
            if not isinstance(err, CommandError):
                return True
            code = err.code
        elif isinstance(err, int):
            code = err
        else:
            return False
        return code >= 300

    @classmethod
    def is_critical(cls, err: Exception | int) -> bool:
        if err is None:
            return False
        if isinstance(err, Exception):
            if not isinstance(err, CommandError):
                return True
            code = err.code
        elif isinstance(err, int):
            code = err
        else:
            return False
        # 400 以上的异常对解释流程是致命的.
        return code >= 400

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
    def description(cls, errcode: int, errmsg: str | None = None, error_name: str = '') -> str:
        if errcode == cls.SUCCESS:
            return "success"
        if not error_name:
            error_name = cls.get_error_code_name(errcode)
        return "{}: {}".format(error_name, errmsg or "no errmsg")
