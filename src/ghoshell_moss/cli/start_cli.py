"""moss start — orient yourself. Always begin here.

Like CLAUDE.md loads project context for AI, moss start loads the MOSS cognitive
map for every session: what MOSS is, what you can do with it, and where to go next.

Single markdown file, no MarkdownKnowledgeBase dependency.
"""

import typer
from pathlib import Path
from .utils import console, echo, print_error, print_info, is_ai_mode

START_DOC_PATH = Path(__file__).resolve().parent / "start.md"

start_app = typer.Typer(
    name="start",
    help="Orient yourself — loads the MOSS cognitive map, like CLAUDE.md for the CLI.",
    no_args_is_help=False,
    invoke_without_command=True,
)


def _render_markdown(text: str) -> None:
    if is_ai_mode():
        echo(text)
    else:
        from rich.markdown import Markdown
        from rich.panel import Panel
        console.print(Panel.fit(Markdown(text), border_style="blue"))


@start_app.callback(invoke_without_command=True)
def start(ctx: typer.Context):
    """Load the MOSS cognitive map — start every session here.

    Like CLAUDE.md orients an AI to a project, this command orients you
    to MOSS: what it is, what you can do with it, and what to try next.

    Examples:
        moss start              # rich rendering
        moss --ai start         # plain text for AI consumption
    """
    if ctx.invoked_subcommand is not None:
        return

    if not START_DOC_PATH.exists():
        print_error(f"Start document not found: {START_DOC_PATH}")
        print_info("This is likely a packaging issue — please report it.")
        raise typer.Exit(code=1)

    text = START_DOC_PATH.read_text(encoding="utf-8").strip()
    _render_markdown(text)

    echo("")
    echo(f"Source: {START_DOC_PATH}")
    echo("")
    echo("Next:")
    echo("  moss --ai all-commands      See everything MOSS can do")
    echo("  moss skills list            Action-oriented skills (发现/召回)")
    echo("  moss docs list               Architecture reference docs")
