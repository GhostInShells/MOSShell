from typing import Dict
from enum import Enum

__all__ = ['FatalError', 'InterpretError', 'CommandErrorCode', 'CommandError']


class FatalError(Exception):
    """
    致命错误, 会导致 Shell 停摆, 状态也需要清空.
    """
    pass


class InterpretError(Exception):
    """
    解释器解释异常, 是可以恢复的异常.
    """
    pass


class CommandError(Exception):

    def __init__(self, code: int = -1, message: str = ""):
        self.code = code
        self.message = message
        super().__init__(f"Command failed with code `{code}`: {message}")


class CommandErrorCode(int, Enum):
    SUCCESS = 0
    CANCELLED = 10010
    NOT_AVAILABLE = 10020
    INVALID_USAGE = 10030
    NOT_FOUND = 10040
    FAILED = 10050
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
