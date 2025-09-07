class FatalError(Exception):
    pass


class CommandError(Exception):

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Command failed with code {code}: {message}")


class StopTheLoop(Exception):
    pass
