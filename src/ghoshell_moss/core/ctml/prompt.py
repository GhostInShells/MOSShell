from pathlib import Path

VERSION = "v0_2_0.zh"

__all__ = [
    'get_moss_ctml_meta_instruction',
]


def get_moss_ctml_meta_instruction(version: str = VERSION) -> str:
    path = Path(__file__).parent.joinpath(f"prompts/ctml_{version}.md")
    with path.open() as f:
        return f.read()
