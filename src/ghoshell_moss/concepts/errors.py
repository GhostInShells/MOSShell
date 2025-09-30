class FatalError(Exception):
    pass


class InterpretError(Exception):
    pass


class CommandError(Exception):
    CANCEL_CODE = 10010
    NOT_AVAILABLE = 10020
    INVALID_USAGE = 10030
    CHILDREN_FAILED = 10040
    UNKNOWN_CODE = -1

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Command failed with code {code}: {message}")


class LoopStoppedError(Exception):
    """
    不需要记录的异常.
    """
    pass
