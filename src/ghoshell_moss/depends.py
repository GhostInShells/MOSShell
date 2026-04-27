"""
管理 ghoshell moss 第三方依赖的检查.
"""


def depend_zenoh():
    try:
        import zenoh
    except ImportError:
        raise ImportError(f"Depend zenoh, please install by 'pip install ghoshell_moss[matrix]'")


def depend_circus():
    try:
        import circus
    except ImportError:
        raise ImportError(f"Depend circus, please install by 'pip install ghoshell_moss[matrix]'")


def depend_cli():
    try:
        import typer
    except ImportError:
        raise ImportError(f"Depend typer, please install by 'pip install ghoshell_moss[cli'")


def depend_pyaudio():
    try:
        import pyaudio
    except ImportError:
        raise ImportError(f"Depend pyaudio, please install by 'pip install ghoshell_moss[audio]'")
