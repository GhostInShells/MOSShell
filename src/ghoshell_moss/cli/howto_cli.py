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
from .utils import console, print_error, print_info, print_simple_table, show_status

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
    name="how-tos",
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
    metas = asyncio.run(kb.list_metas(query=query, limit=limit if limit >= 0 else 9999))

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
        title=f"MOSS How-To ({len(metas)} docs)",
    )


@howto_app.command(name="recall")
def recall(
        query: str = typer.Argument(help="recall how-tos by agent with pydantic ai and anthropic api"),
):
    """Ask a question via the recall agent (requires ANTHROPIC_SMALL_FAST_MODEL)."""
    from ghoshell_moss.core.resources.markdown_kb import recall_available

    if not recall_available():
        print_error("ANTHROPIC_SMALL_FAST_MODEL not set.")
        print_info("Set ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY, and ANTHROPIC_SMALL_FAST_MODEL to enable.")
        raise typer.Exit(code=1)

    with show_status("recalling how-to documents..."):
        result = asyncio.run(kb.recall((query, None)))

    console.print(f"[bold]Q:[/bold] {query}\n")

    if result.done:
        if result.locators:
            console.print(f"[bold green]Matched {len(result.locators)} document(s):[/bold green]")
            for loc in result.locators:
                console.print(f"  {loc}")
            console.print(f"\n[dim]Reasoning: {result.reasoning}[/dim]")
        else:
            console.print("[yellow]No matching documents found.[/yellow]")
    else:
        console.print(f"[bold yellow]Need clarification:[/bold yellow] {result.prompt}")
        if result.choices:
            console.print()
            for c in result.choices:
                console.print(f"  - {c}")


@howto_app.command(name="read")
def read_doc(
        path: str = typer.Argument(help="Document path, e.g. 'how-to-make-how-to.md'"),
        raw: bool = typer.Option(False, "--raw", help="Output raw markdown without syntax highlighting."),
):
    """Read a how-to document by path."""
    item = asyncio.run(kb.get(path))
    if item is None:
        print_error(f"Document not found: {path}")
        print_info("Use 'moss how-tos list' to see available documents.")
        raise typer.Exit(code=1)

    text = asyncio.run(item.get())
    if raw:
        console.print(text)
    else:
        from rich.syntax import Syntax
        console.print(f"[bold blue]{kb.host}://{path}[/bold blue]\n")
        syntax = Syntax(text, "markdown", theme="monokai", line_numbers=True)
        console.print(syntax)
