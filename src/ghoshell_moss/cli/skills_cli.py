"""Skills CLI — MOSS 复合任务技能 (SKILL.md) 的发现、读取与语义召回.

取代 moss howtos. skills 随 CLI 分发 (cli/skills/), 默认扫描包内技能,
--root 覆盖探索任意目录的 skills. recall 语义召回需 LLM 配置,
取不到 LLMFuncs 时给出清晰提示而非静默失败.
"""

import asyncio
import json
import typer
from pathlib import Path

from .utils import console, print_error, print_info, print_simple_table

SKILLS_ROOT = Path(__file__).resolve().parent / "skills"
SKILLS_HOST = "moss-skills"

skills_app = typer.Typer(
    name="skills",
    help="MOSS skills — 复合任务的行动导向技能 (发现/读取/召回).",
    no_args_is_help=True,
)


def _load_kb(root: Path, llm_funcs=None):
    from ghoshell_moss.resources.markdown_kb import MarkdownKnowledgeBase
    kb = MarkdownKnowledgeBase(
        host=SKILLS_HOST, root=root,
        pattern="*/SKILL.md", keys=["name", "description"],
        llm_funcs=llm_funcs,
    )
    kb.scan()
    return kb


def _resolve_root(override: Path | None) -> Path:
    return (override or SKILLS_ROOT).resolve()


def _load_llm_funcs():
    """从 project container 取 LLMFuncs; 失败返回 None (recall 依赖宽容)."""
    try:
        from ghoshell_moss.contracts.llms import LLMFuncs
        from ghoshell_moss.core.blueprint.project import Project
        project = Project.discover()
        project.bootstrap()
        return project.container.force_fetch(LLMFuncs)
    except Exception:
        return None


def _skill_name(path: str) -> str:
    """从 meta path 提取技能名, 去掉 /SKILL.md."""
    if path.endswith("/SKILL.md"):
        return path[:-len("/SKILL.md")]
    return path.removesuffix(".md")


def _skill_locator_name(locator: str) -> str:
    """从 locator (scheme://host/path) 提取可读技能名."""
    # scheme://host/<path>
    idx = locator.find("://")
    if idx < 0:
        return locator
    rest = locator[idx + 3:]
    slash = rest.find("/")
    if slash < 0:
        return locator
    path = rest[slash + 1:]
    return _skill_name(path)


def _resolve_read_path(path: str) -> str:
    """补全用户输入到 SKILL.md 路径."""
    if path.endswith(".md"):
        return path
    return f"{path.rstrip('/')}/SKILL.md"


@skills_app.command(name="list")
def list_skills(
        query: str = typer.Option(None, "--query", "-q", help="Keyword filter"),
        root: Path = typer.Option(
            None, "--root",
            help="Explore skills in this directory (default: bundled cli/skills)",
        ),
        json_out: bool = typer.Option(False, "--json", help="JSON output for AI consumption."),
        limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
):
    """Discover skills from the skills knowledge base."""
    resolved = _resolve_root(root)
    kb = _load_kb(resolved)
    metas = asyncio.run(kb.list_infos(query=query, limit=limit if limit >= 0 else 9999))

    if not metas:
        print_info("No skills found.")
        return

    if json_out:
        console.print(json.dumps([{
            "locator": m.locator,
            "host": m.host,
            "path": m.path,
            "name": m.title,
            "description": m.description,
        } for m in metas], ensure_ascii=False, indent=2))
        return

    if not query:
        print_info(f"Skills root: {resolved}")
        print_info("Tip: `moss skills read <path>` 查看技能内容.")

    rows = [[_skill_name(m.path), m.description[:120] + ("..." if len(m.description) > 120 else "")]
            for m in metas]
    print_simple_table(
        data=rows,
        headers=["Skill", "Description"],
        title=f"MOSS Skills ({len(rows)})",
    )


@skills_app.command(name="read")
def read_skill(
        path: str = typer.Argument(..., help="Skill path (e.g. build-a-gui-app)."),
        raw: bool = typer.Option(False, "--raw", help="Output raw markdown without syntax highlighting."),
        root: Path = typer.Option(
            None, "--root",
            help="Explore skills in this directory (default: bundled cli/skills)",
        ),
):
    """Read a skill by path. Accepts short name (build-a-gui-app) or full path."""
    resolved_path = _resolve_read_path(path)
    kb = _load_kb(_resolve_root(root))
    item = asyncio.run(kb.get(resolved_path))
    if item is None:
        print_error(f"Skill not found: {path}")
        print_info("Use 'moss skills list' to see available skills.")
        raise typer.Exit(code=1)

    text = asyncio.run(item.get())
    if raw:
        console.print(text)
    else:
        from rich.syntax import Syntax
        console.print(f"[bold blue]{kb.host}://{resolved_path}[/bold blue]\n")
        syntax = Syntax(text, "markdown", theme="monokai", line_numbers=True)
        console.print(syntax)


@skills_app.command(name="recall")
def recall_skill(
        query: str = typer.Argument(..., help="Task / query to recall skills for."),
        root: Path = typer.Option(
            None, "--root",
            help="Explore skills in this directory (default: bundled cli/skills)",
        ),
        json_out: bool = typer.Option(False, "--json", help="JSON output."),
):
    """Semantic recall — LLMFuncs multi-label classification over skill metas."""
    llm_funcs = _load_llm_funcs()
    if llm_funcs is None:
        print_error("recall 需要 LLM 配置 — 未能从 project container 获取 LLMFuncs.")
        print_info("配置 LLM 见 `moss llms list` / `moss project env-init`; 无 LLM 时用 `moss skills list` 手动发现.")
        raise typer.Exit(code=1)

    kb = _load_kb(_resolve_root(root), llm_funcs=llm_funcs)
    try:
        rec = asyncio.run(kb.recall(query))
    except NotImplementedError as e:
        print_error(f"recall 不可用: {e}")
        raise typer.Exit(code=1)

    if json_out:
        console.print(json.dumps(rec.model_dump(), ensure_ascii=False, indent=2))
        return

    if not rec.locators:
        print_info("No skills recalled for the query.")
        return
    rows = [[_skill_locator_name(loc)] for loc in rec.locators]
    print_simple_table(
        data=rows,
        headers=["Skill"],
        title="Recalled Skills",
    )
    if rec.reasoning:
        print_info(rec.reasoning)
