from pathlib import Path

CTML_VERSION = "v1_0_0.zh"

__all__ = [
    'get_moss_ctml_meta_instruction',
    'CTML_VERSION',
]

__instructions = {}


def get_moss_ctml_meta_instruction(version: str = CTML_VERSION) -> str:
    global __instructions
    version_file = f"prompts/ctml_{version}.md"
    if version in __instructions:
        return __instructions[version]

    path = Path(__file__).parent.joinpath(version_file)
    text = path.read_text(encoding="utf-8")
    # 总共也不会有多少个版本, 直接放字典了. 有可能变多时, 再用 cache 吧.
    __instructions[version] = text
    return text
