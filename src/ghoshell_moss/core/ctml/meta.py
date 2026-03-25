from pathlib import Path

CTML_VERSION = "v1_0_0.zh"

__all__ = [
    'get_moss_ctml_meta_instruction',
    'CTML_VERSION',
]


def get_moss_ctml_meta_instruction(version: str = CTML_VERSION) -> str:
    path = Path(__file__).parent.joinpath(f"prompts/ctml_{version}.md")
    with path.open() as f:
        return f.read()
