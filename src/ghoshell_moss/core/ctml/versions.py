from typing import Iterable
from pathlib import Path

CTML_VERSION = "v1_0_0.zh"

__all__ = [
    'get_moss_ctml_meta_instruction',
    'expecting_version_file',
    'default_moss_ctml_meta_instruction_directory',
    'search_version_file_in_dir',
    'get_version_from_filename',
    'CTML_VERSION',
]

__instructions = {}


def default_moss_ctml_meta_instruction_directory() -> Path:
    return Path(__file__).parent.joinpath("prompts")


def search_version_file_in_dir(directory: Path) -> Iterable[Path]:
    if directory.is_dir():
        for file in directory.rglob("*.md"):
            yield file
    else:
        yield from []


def get_version_from_filename(filename: str) -> str:
    if filename.endswith(".md"):
        return filename[:-3]
    return filename


def expecting_version_file(ctml_version: str) -> Path:
    if ctml_version.endswith('.md'):
        ctml_version = ctml_version[:-3]
    directory = default_moss_ctml_meta_instruction_directory()
    version_file = directory.joinpath(f"{ctml_version}.md")
    return version_file


def get_moss_ctml_meta_instruction(version: str = CTML_VERSION) -> str:
    global __instructions
    version_file = expecting_version_file(version)
    filename = version_file.name
    if filename in __instructions:
        return __instructions[filename]

    if not version_file.exists():
        raise FileNotFoundError(f"File not found at: {version_file}")
    text = version_file.read_text(encoding="utf-8")
    # 总共也不会有多少个版本, 直接放字典了. 有可能变多时, 再用 cache 吧.
    __instructions[filename] = text
    return text
