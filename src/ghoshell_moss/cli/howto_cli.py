"""
How-To CLI — auto-reflected from the how-to knowledge base resource.

Directory structure maps to typer hierarchy:
  how_to/README.md             → help text
  how_to/*.md (non-README)     → moss how-to <stem>
  how_to/<dir>/README.md       → sub-typer help text
  how_to/<dir>/*.md            → moss how-to <dir> <stem>

All reads go through MarkdownKnowledgeBase (the ResourceStorage).
"""

import asyncio
import typer
from pathlib import Path
from .utils import console, print_error, print_info, print_simple_table

HOW_TO_ROOT = Path(__file__).resolve().parent / "how_tos"


def load_markdown_knowledge_base(_path: Path):
    """
    Initialize and scan the knowledge base (sync, fast).
    :return: MarkdownKnowledgeBase
    """
    from ghoshell_moss.core.resources.markdown_kb import MarkdownKnowledgeBase
    _path = _path
    _kb = MarkdownKnowledgeBase(host="moss-howto", root=_path)
    _kb.scan()
    return _kb


# -- Build the app ----------------------------------------------------------

kb = load_markdown_knowledge_base(HOW_TO_ROOT)

howto_app = typer.Typer(
    name="howtos",
    # Use README first line as help, falling back to a default
    help=[m.description for m in kb.metas if m.path == "README.md"][0]
    if any(m.path == "README.md" for m in kb.metas)
    else "MOSS How-To knowledge base",
    no_args_is_help=True,
)


@howto_app.command(name="list")
def list_docs(
        query: str = typer.Option(None, "--query", "-q", help="Keyword filter"),
        json_out: bool = typer.Option(False, "--json", help="JSON output for AI consumption."),
        limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
):
    """List how-to documents from the knowledge base."""
    metas = asyncio.run(kb.list_infos(query=query, limit=limit if limit >= 0 else 9999))

    if not metas:
        print_info("No how-to documents found.")
        return

    if json_out:
        import json as j
        console.print(j.dumps([{
            "locator": m.locator,
            "host": m.host,
            "path": m.path,
            "title": m.title,
            "description": m.description
        } for m in metas], ensure_ascii=False, indent=2))
        return

    if not query:
        console.print("Tip: 添加或修改 howto 前, 先读 README.md — 判定入口三问 + 反模式清单.\n")

    rows = [
        [
            m.path,
            m.title,
            (m.description[:60] + "...") + "\n" if len(m.description) > 60 else m.description + "\n"
        ]
        for m in metas if not m.path.endswith("README.md")
    ]
    print_simple_table(
        data=rows,
        headers=["Path", "Title", "Description"],
        title=f"MOSS How-To ({len(rows)} docs)",
    )


@howto_app.command(name="read")
def read_doc(
        path: str = typer.Argument(help="Document path."),
        raw: bool = typer.Option(False, "--raw", help="Output raw markdown without syntax highlighting."),
):
    """Read a how-to document by path."""
    item = asyncio.run(kb.get(path))
    if item is None and not path.endswith(".md"):
        item = asyncio.run(kb.get(path + ".md"))
    if item is None:
        print_error(f"Document not found: {path}")
        print_info("Use 'moss howtos list' to see available documents.")
        raise typer.Exit(code=1)

    text = asyncio.run(item.get())
    if raw:
        console.print(text)
    else:
        from rich.syntax import Syntax
        console.print(f"[bold blue]{kb.host}://{path}[/bold blue]\n")
        syntax = Syntax(text, "markdown", theme="monokai", line_numbers=True)
        console.print(syntax)
