"""
管理 ghoshell_moss 第三方依赖的检查.
"""

def depend_cli():
    try:
        import typer, rich, dotenv  # noqa: F401
    except ImportError:
        raise ImportError("install ghoshell_moss[cli]")

def depend_matrix():
    depend_cli()
    try:
        import zenoh  # noqa: F401
    except ImportError:
        raise ImportError("install ghoshell_moss[matrix]")

def depend_host():
    depend_matrix()
    try:
        import prompt_toolkit  # noqa: F401
        import pexpect  # noqa: F401
    except ImportError:
        raise ImportError("install ghoshell_moss[host]")

def depend_ghost():
    try:
        import pydantic_ai, anthropic  # noqa: F401
    except ImportError:
        raise ImportError("install ghoshell_moss[ghost]")
