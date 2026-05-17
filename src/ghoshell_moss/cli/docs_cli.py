"""
Docs CLI — browse reference documentation.

Single root command. No subcommands. The docs group will never grow commands.

Default behavior (no flags):
  Show root README.md + hints about available doc sets.

With --ai:
  Default to docs/ai/, show tree.

With --lang:
  Default to docs/<lang>/, show tree (or README if empty).

With --path:
  Override all defaults, scan the given path.
"""

import subprocess
import typer
from pathlib import Path
from typing import Optional
from .utils import console, echo, print_error, print_info, print_panel, is_ai_mode

DOCS_ROOT = Path(__file__).resolve().parent / "docs"


# ---------------------------------------------------------------------------
# .gitignore helpers
# ---------------------------------------------------------------------------

def _read_gitignore(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return []
    patterns = []
    with open(gitignore) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def _is_ignored(rel_path: str, patterns: list[str]) -> bool:
    import fnmatch
    for p in patterns:
        if fnmatch.fnmatch(rel_path, p) or fnmatch.fnmatch(rel_path, p.rstrip("/") + "/*"):
            return True
        if "/" in p or p.endswith("/"):
            if fnmatch.fnmatch(rel_path + "/", p) or fnmatch.fnmatch(rel_path, p.rstrip("/")):
                return True
    return False


# ---------------------------------------------------------------------------
# Heading extraction
# ---------------------------------------------------------------------------

def _first_heading(filepath: Path) -> str:
    try:
        with open(filepath) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    return stripped[2:].strip()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Git mtime
# ---------------------------------------------------------------------------

def _git_mtime(root: Path, rel_path: str) -> int:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", rel_path],
            capture_output=True, text=True, cwd=root, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass
    try:
        return int((root / rel_path).stat().st_mtime)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------

def _build_tree(root: Path, patterns: list[str]) -> dict:
    tree: dict = {"dirs": {}, "files": []}

    for entry in sorted(root.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
        rel = str(entry.relative_to(root))
        if _is_ignored(rel, patterns):
            continue
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            subtree = _build_tree(entry, patterns)
            if subtree["dirs"] or subtree["files"]:
                tree["dirs"][entry.name] = subtree
        elif entry.suffix == ".md":
            tree["files"].append({
                "path": rel,
                "name": entry.name,
                "heading": _first_heading(entry),
                "mtime": _git_mtime(root, rel),
            })

    tree["files"].sort(key=lambda f: f["mtime"], reverse=True)
    return tree


# ---------------------------------------------------------------------------
# Tree output
# ---------------------------------------------------------------------------

def _print_tree(root: Path, tree: dict):
    echo(str(root))
    for line in _tree_lines(tree):
        echo(line)


def _tree_lines(tree: dict, prefix: str = "") -> list[str]:
    lines = []
    dir_names = sorted(tree.get("dirs", {}).keys())
    files = tree.get("files", [])

    for i, name in enumerate(dir_names):
        is_last_dir = (i == len(dir_names) - 1) and not files
        connector = "└── " if is_last_dir else "├── "
        lines.append(f"{prefix}{connector}{name}/")
        sub_prefix = "    " if is_last_dir else "│   "
        lines.extend(_tree_lines(tree["dirs"][name], prefix + sub_prefix))

    for i, f in enumerate(files):
        connector = "└── " if i == len(files) - 1 else "├── "
        heading = f" — {f['heading']}" if f["heading"] else ""
        lines.append(f"{prefix}{connector}{f['name']}{heading}")

    return lines


# ---------------------------------------------------------------------------
# Hints
# ---------------------------------------------------------------------------

def _show_default(root: Path):
    """Show root README + hints about available doc sets."""
    readme = root / "README.md"
    if readme.exists():
        content = readme.read_text().strip()
        echo(content)
        echo("")

    # Discover available subdirs
    subdirs = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            subdirs.append(d.name)

    if subdirs:
        echo("Available doc sets:")
        for name in subdirs:
            label = {
                "en": "English documentation (default)",
                "zh": "Chinese documentation (--lang zh)",
                "ai": "AI model documentation (--ai)",
            }.get(name, name)
            echo(f"  {name}/  — {label}")

    echo("")
    echo("Usage: moss docs [--ai] [--lang en|zh] [--path <path>]")


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------

def docs_cmd(
        path: Optional[str] = typer.Option(
            None, "--path", "-p",
            help="Custom docs root path (bypasses --ai/--lang defaults).",
        ),
        lang: str = typer.Option(
            "en", "--lang", "-l",
            help="Language subdirectory for human docs (default: en).",
        ),
):
    """Browse MOSS reference documentation as a directory tree."""
    # --path overrides everything
    if path:
        root = Path(path).resolve()
        if not root.is_dir():
            print_error(f"Docs directory not found: {root}")
            raise typer.Exit(code=1)
        patterns = _read_gitignore(root)
        tree = _build_tree(root, patterns)
        if not tree["dirs"] and not tree["files"]:
            print_info(f"No .md files found under {root}")
            return
        _print_tree(root, tree)
        return

    # AI mode: default to ai/ docs
    if is_ai_mode():
        root = DOCS_ROOT / "ai"
        if not root.is_dir():
            print_info(f"AI docs directory not found: {root}")
            print_info("Run 'moss docs' without --ai for available doc sets.")
            raise typer.Exit(code=1)
        patterns = _read_gitignore(root)
        tree = _build_tree(root, patterns)
        if not tree["dirs"] and not tree["files"]:
            print_info(f"No .md files found under {root}")
            return
        _print_tree(root, tree)
        return

    # Human mode with --lang: show lang-specific tree
    if lang and lang != "en":
        root = DOCS_ROOT / lang
        if not root.is_dir():
            print_error(f"Language docs directory not found: {root}")
            print_info("Available languages: en, zh")
            raise typer.Exit(code=1)
        patterns = _read_gitignore(root)
        tree = _build_tree(root, patterns)
        if not tree["dirs"] and not tree["files"]:
            print_info(f"No .md files found under {root}")
            return
        _print_tree(root, tree)
        return

    # Default human mode (en): show README + hints
    _show_default(DOCS_ROOT)
