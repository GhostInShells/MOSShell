from enum import Enum

__all__ = ['FatalError', 'InterpretError', 'CommandErrorCode', 'CommandError']


class FatalError(Exception):
    pass


class InterpretError(Exception):
    pass


class CommandError(Exception):
    CANCEL_CODE = 10010
    NOT_AVAILABLE = 10020
    INVALID_USAGE = 10030
    UNKNOWN_CODE = -1

    def __init__(self, code: int = -1, message: str = ""):
        self.code = code
        self.message = message
        super().__init__(f"Command failed with code `{code}`: {message}")


class CommandErrorCode(int, Enum):
    CANCEL_CODE = 10010
    NOT_AVAILABLE = 10020
    INVALID_USAGE = 10030
    UNKNOWN_CODE = -1

    def error(self, message: str) -> CommandError:
        return CommandError(self.value, message)
