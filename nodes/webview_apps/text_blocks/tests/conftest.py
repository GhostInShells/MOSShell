import sys
from pathlib import Path

# add source to path so tests can import ghoshell_text_blocks
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
