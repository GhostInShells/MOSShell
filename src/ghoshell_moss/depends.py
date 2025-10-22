def check_agent() -> bool:
    try:
        import litellm
        import prompt_toolkit
        import rich
        return True

    except ImportError as e:
        raise ImportError(f"failed to import agent dependencies, please try to install ghoshell-moss[agent]: {e}")


def check_demo() -> bool:
    try:
        import live2d
        return True
    except ImportError as e:
        raise ImportError(f"failed to import demo dependencies, please try to install ghoshell-moss[demo]: {e}")


def check_mcp() -> bool:
    try:
        import mcp
        return True
    except ImportError as e:
        raise ImportError(f"failed to import mcp, please try to install ghoshell-moss[mcp]: {e}")
