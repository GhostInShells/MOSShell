import os
import sys
import typer
import click as _click
from pathlib import Path
from typing import Optional, List
from ghoshell_moss.cli.utils import (
    print_error, print_warning,
    print_panel, echo, set_ai_mode, is_ai_mode
)
from ghoshell_moss.cli import (
    codex_cli, project_cli, manifests_cli,
    ctml_cli, howto_cli, features_cli, docs_cli,
    start_cli, modes_cli, ghosts_cli, nodes_cli, networks_cli,
    ground_cli, memento_cli, llms_cli,
)
from ghoshell_moss.depends import depend_matrix
from typer.main import get_command
from typer.models import DefaultPlaceholder

__version__ = "0.1.0-beta"

# 创建 app 对象
# help_option_names 依然有效
app = typer.Typer(
    name="moss",
    help="MOSS - command line tool for managing and operating the MOSShell system.",
    rich_markup_mode=None,  # 如果你将来想用 rich，可以改为 "rich"
    no_args_is_help=True  # 没传子命令时自动显示帮助
)

app.add_typer(start_cli.start_app, name="start", short_help="Orient yourself — loads the MOSS cognitive map")

app.add_typer(codex_cli.codex_app, name="codex", short_help="Runtime introspection and code evaluation tools")
app.add_typer(project_cli.project_app, name="project", short_help="MOSS Project tools")
app.add_typer(ctml_cli.ctml_app, name="ctml", short_help="environment ctml manager")
app.add_typer(howto_cli.howto_app, name="howtos", short_help="MOSS How-To knowledge base")
app.add_typer(features_cli.features_app, name="features", short_help="AI-native feature tracking")
app.add_typer(docs_cli.docs_app, name="docs", short_help="Systematic architecture reference docs (low frequency)")
app.add_typer(modes_cli.modes_app, name="modes", short_help="List and inspect available runtime modes")
app.add_typer(ghosts_cli.ghosts_app, name="ghosts", short_help="List and inspect available ghosts")
app.add_typer(ground_cli.ground_app, name="ground", short_help="Cognitive ground — pin addresses to a directory")
app.add_typer(memento_cli.memento_app, name="memento", short_help="Memento — cognitive-trajectory system (commit anchors, fork, annotate)")
app.add_typer(llms_cli.llms_app, name="llms", short_help="Inspect and call LLM configs — list models, verify availability.")

# Matrix-dependent groups: only register when zenoh is available.
# All CLI modules import safely without zenoh — the check gates registration,
# not import. A clean ``moss --help`` shouldn't list commands that can't run.
try:
    depend_matrix()
except ImportError:
    pass
else:
    app.add_typer(nodes_cli.nodes_app, name="nodes", short_help="Discover, create, launch, and maintain node cells")
    app.add_typer(networks_cli.networks_app, name="networks", short_help="List and inspect available network configurations")
    app.add_typer(manifests_cli.manifest_app, name="manifests", short_help="MOSS workspace manifest tools")


@app.callback(invoke_without_command=True)
def main(
        ctx: typer.Context,
        version: Optional[bool] = typer.Option(
            None, "--version", "-V", help="Show version information", is_eager=True
        ),
        ai: bool = typer.Option(
            False, "--ai", help="Plain text output for AI consumption (no rich formatting)",
        ),
        mode: Optional[str] = typer.Option(
            None, "--mode", "-m",
            help="MOSS mode (overrides MOSS.md default)",
        ),
        ghost: Optional[str] = typer.Option(
            None, "--ghost", "-g",
            help="MOSS ghost (overrides MOSS.md default)",
        ),
        network: Optional[str] = typer.Option(
            None, "--network",
            help="Network driver (overrides MOSS.md default)",
        ),
        scope: Optional[str] = typer.Option(
            None, "--scope",
            help="Network scope (overrides MOSS.md default)",
        ),
        workspace: Optional[Path] = typer.Option(
            None, "--workspace", "-w",
            help="MOSS workspace path (overrides auto-discovery)",
        ),
):
    """
    MOSS - command line tool

    This is a command line tool for MOSS (Model-oriented Operating System Shell).
    """
    if ai:
        set_ai_mode(True)

    if version:
        print_panel(
            f"MOSS CLI v{__version__}\n"
            f"MOSS (Model-oriented Operating System Shell)\n"
            f"Python: {sys.version.split()[0]}",
            title="Version Information"
        )
        raise typer.Exit()

    # 将全局环境选项注入到 Environment 单例
    # 只在用户显式提供参数时才触发，失败不影响不需要环境的命令（如 codex, concepts）
    if mode is not None or ghost is not None or network is not None or scope is not None or workspace is not None:
        _set_global_environment(mode, ghost, network, scope, workspace)


# ---------------------------------------------------------------------------
# 全局环境注入
# ---------------------------------------------------------------------------

def _set_global_environment(
    mode: str | None,
    ghost: str | None,
    network: str | None,
    scope: str | None,
    workspace: Path | None,
) -> None:
    """
    CLI callback: 显式构造 Environment + seal, 注册为进程 singleton.

    §UU-1 seal 定案纪律: Environment 无 set_* (mid-life 修改), 参数一次性塞
    __init__, seal 一次性事实. CLI 入口负责在任何 Environment.discover() 之前
    完成 "构造 + seal", 让 subcommand 的 discover 直接拿到含 CLI 参数的 singleton.

    老实现是 "discover 后 set_mode" — set 接口被 §UU-1 拿掉后, 参数无声无息全部
    丢失, 是 CLI 参数化不生效的根因. commit 4c75f76b 已 flag 本入口为"下一次
    fix pass" 待办, 本次修完.

    workspace 不存在时静默返回 — 无 workspace 需求的命令 (codex, ctml, howtos,
    features) 不需要 env; 需要的命令走 Environment.discover 会自触发裸构造并暴露
    具体错误.
    """
    from ghoshell_moss.core.blueprint.environment import Environment

    if workspace is not None:
        # workspace 通过环境变量注入 —— Environment.find_workspace_path() 第一优先级
        # (这不是隐式约定, 是 workspace 探测独立机制, 与 CLI 参数化正交)
        os.environ["MOSS_WORKSPACE"] = str(workspace.resolve())

    try:
        env = Environment(
            mode=mode,
            ghost=ghost,
            network=network,
            scope=scope,
        )
    except EnvironmentError:
        # workspace 不存在等. 需环境的命令会在自己 discover 时报出具体错误.
        return
    env.seal()


@app.command(
    "help",
    short_help="Show help for one or more commands",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cli_help(ctx: typer.Context):
    """
    Show help for one or more commands.

    Without arguments, shows the top-level help.
    With arguments, each sequence resolves to a command path (group + subcommand)
    and prints help for each resolved command.

    Examples:
        moss --ai help                          # top-level help
        moss --ai help codex get-interface      # help for one command
        moss --ai help codex get-interface concepts core  # help for two commands
    """
    if ctx.args:
        _show_help_for_paths(app, list(ctx.args))
    else:
        _show_group_help(app, ["moss"])


@app.command("all-commands", short_help="Show all available commands as a tree")
def all_commands(
    depth: int = typer.Option(2, "--depth", "-d", help="Depth: 1=groups, 2=groups+commands, 3=+parameters"),
    group: Optional[str] = typer.Option(None, "--group", "-g", help="Limit to a specific command group subtree"),
):
    """
    Show all available commands as a hierarchical tree.

    Designed for AI discovery: reduces CLI exploration from 40+ rounds to 1-2.
    Use --depth to control detail level.
    Use --group to focus on a specific subtree.
    """
    if group:
        target = _find_group_by_name(app, group)
        if target is None:
            print_error(f"Unknown group: {group}")
            raise typer.Exit(code=1)
        _print_command_tree(target, depth, prefix=group)
    else:
        _print_command_tree(app, depth)


@app.command(
    name="init",
    short_help="Initialize a MOSS project — creates .moss workspace in the target directory",
)
def init_project(
        path: Optional[Path] = typer.Argument(
            None,
            help="Project directory. The .moss workspace is created inside it.",
        ),
        cwd: bool = typer.Option(
            False, "--cwd", "-c",
            help="Use current directory as the project root.",
        ),
        yes: bool = typer.Option(
            False, "--yes", "-y",
            help="Skip all confirmation prompts (for non-interactive / AI use).",
        ),
        force: bool = typer.Option(
            False, "--force", "-f",
            help="Overwrite existing .moss with latest stub templates.",
        ),
) -> None:
    from ghoshell_moss.core.blueprint.environment import Environment, DEFAULT_WORKSPACE_DIR_NAME
    from ghoshell_moss.cli.utils import console as _console, print_info, print_success

    # 1. Resolve project directory
    if path is not None:
        project_dir = path.resolve()
    elif cwd:
        project_dir = Path.cwd()
    else:
        # Interactive
        cwd_path = Path.cwd()
        _console.print("\n[bold cyan]MOSS Project Setup[/bold cyan]")
        _console.print(f" 1) Current directory: [dim]{cwd_path}[/dim]")
        _console.print(f" 2) Custom path")

        if is_ai_mode():
            print_error("Interactive mode not available with --ai flag.")
            print_info("Use --cwd or provide a path argument (with --yes to skip confirmation).")
            raise typer.Exit(code=1)

        choice = typer.prompt("\nSelect an option", default="1", type=str)

        if choice == "1":
            project_dir = cwd_path
        elif choice == "2":
            custom_path = typer.prompt("Enter project path", type=Path)
            project_dir = custom_path.resolve()
        else:
            print_error("Invalid selection.")
            raise typer.Exit(code=1)

    workspace_dir = project_dir / DEFAULT_WORKSPACE_DIR_NAME
    is_reinit = workspace_dir.exists() and (workspace_dir / 'MOSS.md').exists()

    # 2. Confirmation
    if force:
        pass
    elif is_reinit:
        if not yes and not typer.confirm(
            f"'.moss' already exists in '{project_dir.name}'. Force re-initialize?",
            default=False,
        ):
            print_warning("Aborted.")
            return
    elif not project_dir.exists():
        if not yes and not typer.confirm(
            f"Create project directory '{project_dir}' and initialize .moss?",
            default=True,
        ):
            print_warning("Aborted.")
            return
    else:
        if not yes and not typer.confirm(
            f"Create .moss workspace in '{project_dir}'?",
            default=True,
        ):
            print_warning("Aborted.")
            return

    # 3. Execute
    action = "Re-initializing" if is_reinit else "Initializing"
    print_info(f"{action} .moss in: {project_dir}")
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        Environment.init_workspace(workspace_dir, force=force)
        print_success(f"{action} completed successfully.")
        print_info("Next step: use 'moss project env-init' to see environment template.")
    except Exception as e:
        print_error(f"Failed to initialize: {e}")
        raise typer.Exit(code=1)


@app.command(
    name="shell-init",
    short_help="Export MOSS environment variables for shell evaluation",
)
def shell_init() -> None:
    """Output shell-compatible export statements for the current MOSS environment.

    Usage:
        eval $(moss shell-init)
        eval $(moss --mode desktop --ghost echo shell-init)

    All values come from MOSS.md defaults or CLI override parameters.
    """
    from ghoshell_moss.core.blueprint.environment import Environment

    try:
        env = Environment.discover()
    except EnvironmentError as e:
        print_error(f"Environment discovery failed: {e}")
        raise typer.Exit(code=1)

    for key, value in env.dump_runtime_scope().items():
        if value:
            echo(f"export {key}='{value}'")


# ---------------------------------------------------------------------------
# Helper functions for all-commands and help
# ---------------------------------------------------------------------------

def _resolve(v, default=None):
    """Resolve Typer DefaultPlaceholder to a real value or default."""
    if isinstance(v, DefaultPlaceholder):
        return default
    return v


def _is_hidden_group(info) -> bool:
    """Check if a TyperInfo group is hidden."""
    return _resolve(info.hidden, False)


def _is_hidden_cmd(info) -> bool:
    """Check if a CommandInfo command is hidden."""
    return _resolve(info.hidden, False)


def _short_help(info) -> str:
    """Get effective short_help from TyperInfo or CommandInfo."""
    return _resolve(info.short_help, "") or ""


def _find_group_by_name(typer_app, name: str):
    """Find a registered group by name."""
    for g in typer_app.registered_groups:
        if g.name == name and not _is_hidden_group(g):
            return g.typer_instance
    return None


# --- all-commands ---

def _print_command_tree(typer_app, depth: int, prefix: str = ""):
    """Print command tree at given depth."""
    if is_ai_mode():
        _print_command_tree_ai(typer_app, depth, prefix)
    else:
        _print_command_tree_human(typer_app, depth, prefix)


def _print_command_tree_ai(typer_app, depth: int, prefix: str = ""):
    """AI mode: plain text command tree."""
    header = f"## moss command tree (depth={depth})"
    if prefix:
        header += f" --group {prefix}"
    echo(header)

    # Collect non-hidden groups and root-level commands
    groups = [g for g in typer_app.registered_groups if not _is_hidden_group(g)]
    root_cmds = [c for c in typer_app.registered_commands if not _is_hidden_cmd(c)]

    for grp in groups:
        echo("")
        sh = _short_help(grp) or "—"
        echo(f"### {grp.name} — {sh}")
        if depth >= 2:
            sub = grp.typer_instance
            sub_cmds = [c for c in sub.registered_commands if not _is_hidden_cmd(c)]
            max_name = max((len(c.name) for c in sub_cmds), default=0)
            for cmd in sub_cmds:
                help_line = _get_command_help(sub, cmd.name)
                echo(f"  {cmd.name.ljust(max_name + 2)}{help_line}")
            if depth >= 3:
                for cmd in sub_cmds:
                    params = _get_command_params(sub, cmd.name)
                    if params:
                        echo(f"  {cmd.name} parameters:")
                        for p in params:
                            echo(f"    {p}")

    # Root-level commands (if any)
    if root_cmds:
        echo("")
        echo("### (root commands)")
        for cmd in root_cmds:
            help_line = _get_command_help(typer_app, cmd.name)
            echo(f"  {cmd.name}  {help_line}")
            if depth >= 3:
                params = _get_command_params(typer_app, cmd.name)
                if params:
                    echo(f"  {cmd.name} parameters:")
                    for p in params:
                        echo(f"    {p}")


def _print_command_tree_human(typer_app, depth: int, prefix: str = ""):
    """Human mode: rich formatted command tree."""
    from ghoshell_moss.cli.utils import console as _console

    header = f"[bold bright_cyan]moss command tree (depth={depth})[/bold bright_cyan]"
    if prefix:
        header += f" [dim]--group {prefix}[/dim]"
    _console.print(header)

    groups = [g for g in typer_app.registered_groups if not _is_hidden_group(g)]
    root_cmds = [c for c in typer_app.registered_commands if not _is_hidden_cmd(c)]

    for grp in groups:
        sh = _short_help(grp) or "—"
        _console.print(f"\n[bold green]{grp.name}[/bold green] — [dim]{sh}[/dim]")
        if depth >= 2:
            sub = grp.typer_instance
            sub_cmds = [c for c in sub.registered_commands if not _is_hidden_cmd(c)]
            for cmd in sub_cmds:
                help_line = _get_command_help(sub, cmd.name)
                _console.print(f"  [bold cyan]{cmd.name}[/bold cyan]  {help_line}")
            if depth >= 3:
                for cmd in sub_cmds:
                    params = _get_command_params(sub, cmd.name)
                    if params:
                        _console.print(f"  [bold]{cmd.name} parameters:[/bold]")
                        for p in params:
                            _console.print(f"    [dim]{p}[/dim]")

    if root_cmds:
        _console.print("\n[bold green](root commands)[/bold green]")
        for cmd in root_cmds:
            help_line = _get_command_help(typer_app, cmd.name)
            _console.print(f"  [bold cyan]{cmd.name}[/bold cyan]  {help_line}")
            if depth >= 3:
                params = _get_command_params(typer_app, cmd.name)
                if params:
                    _console.print(f"  [bold]{cmd.name} parameters:[/bold]")
                    for p in params:
                        _console.print(f"    [dim]{p}[/dim]")


def _get_command_help(typer_app, cmd_name: str) -> str:
    """Get display help text for a command, preferring short_help over help."""
    try:
        click_group = get_command(typer_app)
        sub_cmd = click_group.commands.get(cmd_name) if hasattr(click_group, 'commands') else None
        if not sub_cmd:
            return ""
        return (sub_cmd.short_help or sub_cmd.help or "").split("\n")[0].strip()
    except Exception:
        return ""


def _get_command_params(typer_app, cmd_name: str) -> List[str]:
    """Get formatted parameter descriptions for a command. Returns list of strings."""
    try:
        click_group = get_command(typer_app)
        sub_cmd = click_group.commands.get(cmd_name) if hasattr(click_group, 'commands') else None
        if not sub_cmd:
            return []
        result = []
        for p in sub_cmd.params:
            opts = "|".join(p.opts) if p.opts else p.name
            parts = [opts]
            if p.type and hasattr(p.type, 'name'):
                parts.append(f"<{p.type.name.lower()}>")
            if p.help:
                parts.append(f"— {p.help}")
            if p.default is not None and p.default is not False:
                parts.append(f"(default: {p.default})")
            if p.required:
                parts.append("[required]")
            result.append(" ".join(parts))
        return result
    except Exception:
        return []


# --- help with multi-path ---

def _show_help_for_paths(typer_app, args: List[str]):
    """Parse command paths from args and show help for each resolved command."""
    current = typer_app
    group_path = []  # track path for error messages

    for arg in args:
        # Try to match as a command on the current app
        cmd_info = _find_command(current, arg)
        if cmd_info is not None:
            # Found a terminal command — show its help
            _show_command_help(current, arg, cmd_info)
            current = typer_app  # reset to root
            group_path = []
            continue

        # Try to match as a sub-group
        sub_typer = _find_group_by_name(current, arg)
        if sub_typer is not None:
            current = sub_typer
            group_path.append(arg)
            continue

        # If we're inside a sub-group and can't match anything,
        # show the group's own help as a fallback
        if group_path:
            _show_group_help(current, group_path)
            print_warning(f"'{arg}' is not a valid subcommand of '{' '.join(group_path)}'")
        else:
            print_warning(f"Unknown command or group: '{arg}' — skipping")
        current = typer_app
        group_path = []

    # If we ended inside a group without a terminal command, show group help
    if group_path and current is not typer_app:
        _show_group_help(current, group_path)


def _find_command(typer_app, name: str):
    """Find a non-hidden command by name. Returns CommandInfo or None."""
    for c in typer_app.registered_commands:
        if c.name == name and not _is_hidden_cmd(c):
            return c
    return None


def _show_command_help(typer_app, cmd_name: str, cmd_info):
    """Show Click-level help for a specific command."""
    try:
        click_group = get_command(typer_app)
        ctx = _click.Context(click_group, info_name=click_group.name)
        sub_cmd = click_group.get_command(ctx, cmd_name)
        if sub_cmd is None:
            print_warning(f"Cannot resolve Click command: {cmd_name}")
            return
        sub_ctx = _click.Context(sub_cmd, parent=ctx, info_name=cmd_name)
        echo(sub_ctx.get_help())
    except Exception as e:
        print_error(f"Failed to get help for '{cmd_name}': {e}")


def _show_group_help(typer_app, group_path: List[str]):
    """Show help for a command group (Typer app)."""
    try:
        click_group = get_command(typer_app)
        ctx = _click.Context(click_group, info_name=" ".join(group_path))
        echo(ctx.get_help())
    except Exception as e:
        print_error(f"Failed to get help for '{' '.join(group_path)}': {e}")


def main_entry():
    """Command line entry point"""
    try:
        # Typer 的启动方式
        app()
    except Exception as e:
        print_error(f"Command execution failed: {str(e)}")
        sys.exit(1)
