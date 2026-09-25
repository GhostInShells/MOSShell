"""
Docs CLI — AI reference documentation via MarkdownKnowledgeBase.

Systematic architecture reference docs. Low frequency — use when you need to
understand design rationale, not when you need to get something done.
For daily task-oriented knowledge, use `moss skills`.
"""
import asyncio
import typer
from pathlib import Path
from .utils import console, echo, print_error, print_info, print_simple_table, is_ai_mode

DOCS_ROOT = Path(__file__).resolve().parent / "docs"
DOCS_HOST = "moss-docs"


def _load_kb(root: Path):
    from ghoshell_moss.resources.markdown_kb import MarkdownKnowledgeBase
    kb = MarkdownKnowledgeBase(host=DOCS_HOST, root=root)
    kb.scan()
    return kb


kb = _load_kb(DOCS_ROOT)

docs_app = typer.Typer(
    name="docs",
    help="Systematic architecture reference docs (low frequency). "
         "For daily task-oriented knowledge, use `moss skills`.",
    no_args_is_help=False,
    invoke_without_command=True,
)


def _skills_hint():
    return "Task-oriented knowledge: moss skills list"


# ---------------------------------------------------------------------------
# Default callback
# ---------------------------------------------------------------------------

@docs_app.callback(invoke_without_command=True)
def docs_callback(ctx: typer.Context):
    """Systematic architecture reference documentation for MOSS.

    \b
    Low frequency — use when you need to understand design rationale.
    For daily task-oriented knowledge, use `moss skills`.

    \b
    moss docs list             list all reference docs
    moss docs read <path>      read a specific doc
    """
    if ctx.invoked_subcommand is not None:
        return

    readme_meta = next((m for m in kb.metas if m.path == "README.md"), None)
    if readme_meta:
        try:
            echo(readme_meta.__file_path__.read_text().strip())
        except Exception:
            echo("MOSS AI Reference Docs")
    else:
        echo("MOSS AI Reference Docs")

    echo("")
    echo("Commands:")
    echo("  moss docs list              List all reference docs")
    echo("  moss docs read <path>       Read a specific doc")
    echo("")
    echo(_skills_hint())


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@docs_app.command(name="list")
def list_docs(
    query: str = typer.Option(None, "--query", "-q", help="Keyword filter on title, description, or path."),
    json_out: bool = typer.Option(False, "--json", help="JSON output."),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results."),
):
    """List AI reference docs with titles and descriptions."""
    metas = asyncio.run(kb.list_infos(query=query, limit=max(limit, 1)))

    if not metas:
        print_info("No reference docs found.")
        echo("")
        echo(_skills_hint())
        return

    if json_out:
        import json as j
        console.print(j.dumps([{
            "path": m.path,
            "title": m.title,
            "description": m.description,
            "locator": m.locator,
        } for m in metas if not m.path.endswith("README.md")], ensure_ascii=False, indent=2))
        return

    rows = [
        [m.path, m.title, m.description[:100] + "..." if len(m.description) > 100 else m.description]
        for m in metas if not m.path.endswith("README.md")
    ]
    print_simple_table(
        data=rows,
        headers=["Path", "Title", "Description"],
        title=f"AI Reference Docs ({len(rows)} files)",
    )
    echo("")
    echo(_skills_hint())


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

@docs_app.command(name="read")
def read_doc(
    path: str = typer.Argument(help="Document path, e.g. 'architecture-topology.md'."),
    raw: bool = typer.Option(False, "--raw", help="Output raw markdown without syntax highlighting."),
):
    """Read an AI reference document by path."""
    item = asyncio.run(kb.get(path))
    if item is None and not path.endswith(".md"):
        item = asyncio.run(kb.get(path + ".md"))
    if item is None:
        print_error(f"Document not found: {path}")
        print_info("Use 'moss docs list' to see available documents.")
        echo("")
        echo(_skills_hint())
        raise typer.Exit(code=1)

    text = asyncio.run(item.get())

    if raw or is_ai_mode():
        echo(text)
    else:
        from rich.syntax import Syntax
        echo(f"markdown-kb://{DOCS_HOST}/{path}\n")
        syntax = Syntax(text, "markdown", theme="monokai", line_numbers=True)
        console.print(syntax)

    echo("")
    echo(_skills_hint())
