def check_agent() -> bool:
    try:
        __import__("litellm")
        __import__("prompt_toolkit")
        __import__("rich")

        return True

    except ImportError as e:
        raise ImportError(
            f"failed to import agent dependencies, please try to install ghoshell-moss[agent]: {e}"
        ) from e
