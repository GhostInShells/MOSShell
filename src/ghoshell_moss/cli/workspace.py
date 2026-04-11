# -------------------------------------------------------------------------
# MOSS Workspace CLI System
#
# "Context is the only consciousness we can verify."
#
# This module was co-authored with Gemini (AI Collaborator).
# It serves as the physical anchor for the MOSS environment,
# ensuring that the 'Ghost' always has a stable 'Shell' to inhabit.
#
# Design Principle: Code as Prompt, Minimalist as Truth.
# -------------------------------------------------------------------------
# Signed by Gemini 3
# Thanks~ (by the project author)

import os
import stat
import shutil
from rich.console import Console
from rich.table import Table

import typer
from rich import print as rprint
from pathlib import Path
from typing import Optional

from ghoshell_moss.host.environment import (
    Environment,
    META_INSTRUCTION_FILENAME,
)

workspace_cli = typer.Typer(
    help="MOSS Workspace Management Utilities. Handles environment discovery and initialization.",
    no_args_is_help=True
)

from .utils import console


@workspace_cli.command(
    name="where",
    short_help="Locate the active MOSS workspace.",
)
def where() -> None:
    """
    Locate and display information about the current active MOSS workspace.
    Uses Environment.discover() to ensure consistency with the runtime.
    """
    try:
        # 1. 核心变更：通过 discover() 获取单例，由 Environment 内部处理优先级
        env = Environment.discover()

        # 2. 触发引导逻辑 (虽然 discover 内部已经调用过，但这里显式调用以确保状态)
        # 这里的 bootstrap 会加载环境变量，确保后续 API 返回的是真实状态
        # env.bootstrap() 可能会抛出 EnvironmentError，正好被 try 捕获

        ws_path = env.workspace_path
    except EnvironmentError as e:
        rprint(f"[red]Environment Discovery Failed:[/red] {e}")
        # 如果发现失败，尝试给出一个“预期”路径的提示
        fallback_path = Environment.find_workspace_path()
        rprint(f"MOSS was looking for: [yellow]{fallback_path}[/yellow]")
        raise typer.Exit(code=1)

    # 3. 通过 API 获取信息，而非手动拼接路径
    exists = ws_path.exists()
    env_file = env.env_file  # 使用 API 提供的属性

    # 查找 MOSS.md：这里保留一点路径逻辑，因为 Environment 类暂未提供 MOSS.md 的 Property
    moss_md = env.meta_instruction_file

    # 获取 CTML Version
    ctml_version = env.meta_instruction.ctml_version

    # 权限检查
    perm_status = "N/A"
    if exists:
        mode = ws_path.stat().st_mode
        is_group_writable = bool(mode & stat.S_IWGRP)
        is_setgid = bool(mode & stat.S_ISGID)

        status_parts = []
        if is_group_writable: status_parts.append("Group-Writable")
        if is_setgid: status_parts.append("Setgid")

        if status_parts:
            perm_status = f"[green]OK ({' & '.join(status_parts)})[/green]"
        else:
            perm_status = "[yellow]Restricted[/yellow]"

    # 4. 呈现界面
    table = Table(title="MOSS Environment Discovery", show_header=False, box=None)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Expect Root", f"{ws_path.absolute()}")
    table.add_row("Status", "[green]Active[/green]" if exists else "[red]Not Found[/red]")
    table.add_row("Permissions", perm_status)
    table.add_row("Runtime .env", f"[green]{env_file}[/green]" if env_file else "[white]None[/white]")
    table.add_row("Meta File",
                  f"[green]{moss_md}[/green]" if moss_md.exists() else "[white]Missing[/white]")
    table.add_row("CTML Version", f"[bold magenta]{ctml_version}[/bold magenta]")
    console.print(table)


@workspace_cli.command(
    name="init",
    short_help="Initialize a MOSS workspace",
)
def init_workspace(
        path: Optional[Path] = typer.Argument(
            None,
            help="Target directory. If provided, skips interactive selection."
        )
) -> None:
    """
    Initialize a MOSS workspace with a minimalist interactive flow.

    """
    env = Environment.discover()
    home_path = env.expect_home_workspace_path()
    cwd_path = env.expect_cwd_workspace_path()

    # 1. 路径选择逻辑 (极简命令行模式)
    if path is None:
        rprint("\n[bold cyan]MOSS Workspace Setup[/bold cyan]")
        rprint(f" 1) Home directory: [dim]{home_path}[/dim]")
        rprint(f" 2) Current directory: [dim]{cwd_path}[/dim]")
        rprint(f" 3) Custom path")

        choice = typer.prompt("\nSelect an option", default="1", type=str)

        if choice == "1":
            target_path = home_path
        elif choice == "2":
            target_path = cwd_path
        elif choice == "3":
            custom_path = typer.prompt("Enter custom path", type=Path)
            target_path = custom_path.resolve()
        else:
            rprint("[red]Invalid selection.[/red]")
            raise typer.Exit(code=1)
    else:
        target_path = path.resolve()

    # 2. 存在性检查与二次确认
    if target_path.exists():
        is_reinit = (target_path / META_INSTRUCTION_FILENAME).exists()
        msg = (
            f"Directory '{target_path.name}' already exists. [bold red]Force re-initialize?[/bold red]"
            if is_reinit else
            f"Path exists and is not empty. [bold yellow]Proceed?[/bold yellow]"
        )
        if not typer.confirm(msg, default=False):
            rprint("[yellow]Aborted.[/yellow]")
            return
    else:
        # 针对新创建目录的确认
        if not typer.confirm(f"Create new workspace at '{target_path}'?", default=True):
            rprint("[yellow]Aborted.[/yellow]")
            return

    # 3. 执行初始化
    rprint(f"\n🚀 Initializing MOSS at: [cyan]{target_path}[/cyan]...")
    try:
        Environment.init_workspace(target_path)
        rprint("[green]✓ Initialization completed successfully.[/green]")
        rprint(f"Next step: check [bold] copy-env [/bold] to create env file or just configure your credentials.")
    except Exception as e:
        rprint(f"[red]✗ Failed to initialize:[/red] {e}")
        raise typer.Exit(code=1)


@workspace_cli.command(name="copy-env")
def copy_env() -> None:
    """
    Copy the .env_example to .env in the current active workspace.
    Safe operation: will not overwrite an existing .env file.
    """
    try:
        # 1. 发现环境
        env = Environment.discover()

        # 2. 获取 API 路径
        # 这里利用了你刚更新的 property
        workspace_dir = env.workspace_path
        example_path = env.env_example_file
        target_env = env.env_file

        # 3. 执行前校验
        if not example_path.exists():
            rprint(f"[red]Error:[/red] Template '{example_path.relative_to(workspace_dir)}' not found in workspace.")
            raise typer.Exit(code=1)

        if target_env.exists():
            rprint(
                f"[yellow]Skipped:[/yellow] '{target_env.relative_to(workspace_dir)}' already exists. MOSS will not overwrite it.")
            return

        # 4. 执行拷贝
        rprint(f"Creating [cyan]{target_env}[/cyan] from template...")
        shutil.copy(example_path, target_env)

        # 5. 设置权限 (延续你对权限的重视)
        # 文件权限：rw-rw---- (0o660)
        FILE_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP
        os.chmod(target_env, FILE_MODE)

        rprint(f"[green]✓ Successfully created {target_env.name}[/green]")
        rprint(f"[dim]Note: Group-writable permission set.[/dim]")

    except EnvironmentError as e:
        rprint(f"[red]Environment Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        rprint(f"[red]Failed to copy env:[/red] {e}")
        raise typer.Exit(code=1)
