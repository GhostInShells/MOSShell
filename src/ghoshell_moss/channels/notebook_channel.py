import os
from pathlib import Path

from ghoshell_moss.core.blueprint.channel_builder import new_channel, MutableChannel
from ghoshell_moss.message import Message

__all__ = ["new_notebook_channel"]


def _safe_path(root: Path, name: str) -> Path:
    """将 name 解析到 root 下的安全路径。拒绝穿透和绝对路径."""
    if os.path.isabs(name) or ".." in name.split(os.sep):
        raise ValueError(f"Invalid page name: {name!r}")
    resolved = (root / name).resolve()
    if not str(resolved).startswith(str(root.resolve())):
        raise ValueError(f"Path traversal denied: {name!r}")
    return resolved


def _tree(root: Path, prefix: str = "") -> list[str]:
    """生成目录树的缩进行."""
    lines = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    except OSError:
        return lines
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            ext = "    " if is_last else "│   "
            lines.extend(_tree(entry, prefix + ext))
        else:
            lines.append(f"{prefix}{connector}{entry.name}")
    return lines


def new_notebook_channel(
    dirpath: str,
    *,
    name: str = "notebook",
    description: str = "",
) -> MutableChannel:
    """基于文件系统目录创建一个可丢弃的 notebook channel.

    每个文件是一页。支持 write / read / append / list / delete。
    context_messages 自动展示当前目录树。

    :param dirpath: 笔记本根目录路径 (如 /tmp/my_notes)
    :param name: Channel 名称
    :param description: Channel 描述
    """
    root = Path(dirpath).resolve()
    root.mkdir(parents=True, exist_ok=True)

    desc = description or f"Disposable notebook at {root}"
    chan = new_channel(name=name, description=desc)

    # -- context: 目录树 --

    @chan.build.context_messages
    def show_tree() -> list[Message]:
        lines = _tree(root)
        if not lines:
            return [Message.new().with_content(f"(empty notebook at {root})")]
        tree_str = f"Notebook {root.name}/\n" + "\n".join(lines)
        return [Message.new().with_content(tree_str)]

    # -- commands --

    @chan.build.command()
    async def write(name: str, *, text__: str = "") -> str:
        """写一页笔记，覆盖已有内容.

        :param name: 文件名 (如 "analysis.md")
        :param text__: 笔记内容，支持多行文本/代码
        """
        path = _safe_path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text__, encoding="utf-8")
        return f"Wrote {name} ({len(text__)} chars)"

    @chan.build.command()
    async def read(name: str) -> str:
        """读取一页笔记.

        :param name: 文件名
        """
        path = _safe_path(root, name)
        if not path.exists():
            return f"Page not found: {name}"
        return path.read_text(encoding="utf-8")

    @chan.build.command()
    async def append(name: str, *, text__: str = "") -> str:
        """追加内容到已有笔记页.

        :param name: 文件名
        :param text__: 追加的内容
        """
        path = _safe_path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text__)
        return f"Appended to {name} ({len(text__)} chars)"

    @chan.build.command()
    async def list_pages() -> list[str]:
        """列出所有笔记页的文件路径."""
        pages = []
        for entry in sorted(root.rglob("*")):
            if entry.is_file():
                pages.append(str(entry.relative_to(root)))
        return pages

    @chan.build.command()
    async def delete(name: str) -> str:
        """删除一页笔记.

        :param name: 文件名
        """
        path = _safe_path(root, name)
        if not path.exists():
            return f"Page not found: {name}"
        path.unlink()
        return f"Deleted {name}"

    return chan
