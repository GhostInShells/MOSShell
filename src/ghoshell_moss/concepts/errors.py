class FatalError(Exception):
    pass


class InterpretError(Exception):
    pass


class CommandError(Exception):

    CANCEL_CODE = 10010
    UNKNOWN_CODE = -1

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Command failed with code {code}: {message}")


class StopTheLoop(Exception):
    pass
