"""Text Blocks node entry point.

Start:  moss nodes run nodes/webview_apps/text_blocks
Debug:  python main.py
"""

import subprocess
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
_VENV_PYTHON = str(_NODE_DIR / ".venv" / "bin" / "python")


def main():
    subprocess.run([_VENV_PYTHON, "-m", "reflex", "run"], check=True)


if __name__ == "__main__":
    main()
