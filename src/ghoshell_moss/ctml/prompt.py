from pathlib import Path

VERSION = "v1"


def get_moss_meta_prompt() -> str:
    path = Path(__file__).parent.joinpath(f"prompt_{VERSION}.md")
    with path.open() as f:
        return f.read()
