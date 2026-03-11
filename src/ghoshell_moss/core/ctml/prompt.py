from pathlib import Path

VERSION = "v2.zh"


def get_moss_meta_prompt(version: str = VERSION) -> str:
    path = Path(__file__).parent.joinpath(f"prompts/ctml_{version}.md")
    with path.open() as f:
        return f.read()
