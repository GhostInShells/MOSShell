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

import typer
from pathlib import Path
from typing import Optional

from ghoshell_moss.core.blueprint.environment import (
    Environment,
    META_CONFIG_FILENAME,
)

workspace_app = typer.Typer(
    help="MOSS Workspace Management Utilities. Handles environment discovery and initialization.",
    no_args_is_help=True
)

from .utils import (
    console,
    print_simple_table,
    print_success,
    print_error,
    print_warning,
    print_info,
    is_ai_mode,
)


@workspace_app.command(
    name="where",
    short_help="Locate the active MOSS workspace.",
)
def where() -> None:
    """
    Locate and display information about the current active MOSS workspace.
    Uses Environment.discover() to ensure consistency with the runtime.
    """
    try:
        env = Environment.discover()
        ws_path = env.workspace_path
    except EnvironmentError as e:
        print_error(f"Environment Discovery Failed: {e}")
        fallback_path = Environment.find_workspace_path()
        print_info(f"MOSS was looking for: {fallback_path}")
        raise typer.Exit(code=1)

    exists = ws_path.exists()
    env_file = env.env_file
    moss_md = env.meta_instruction_file
    ctml_version = env.meta_config.ctml_version

    # permissions check
    perm_status = "N/A"
    if exists:
        mode = ws_path.stat().st_mode
        is_group_writable = bool(mode & stat.S_IWGRP)
        is_setgid = bool(mode & stat.S_ISGID)
        status_parts = []
        if is_group_writable:
            status_parts.append("Group-Writable")
        if is_setgid:
            status_parts.append("Setgid")
        if status_parts:
            perm_status = f"OK ({' & '.join(status_parts)})"
        else:
            perm_status = "Restricted"

    status = "Active" if exists else "Not Found"
    env_file_str = str(env_file) if env_file else "None"
    moss_md_str = str(moss_md) if moss_md.exists() else "Missing"

    print_simple_table(
        data=[
            ["Expect Root", str(ws_path.absolute())],
            ["Status", status],
            ["Permissions", perm_status],
            ["Runtime .env", env_file_str],
            ["Meta File", moss_md_str],
            ["CTML Version", ctml_version],
        ],
        headers=["Property", "Value"],
        title="MOSS Environment Discovery",
    )


@workspace_app.command(
    name="init",
    short_help="Initialize a MOSS workspace",
)
def init_workspace(
        path: Optional[Path] = typer.Argument(
            None,
            help="Target directory. If provided, skips interactive selection."
        ),
        cwd: bool = typer.Option(
            False, "--cwd", "-c",
            help="Use current directory as workspace target."
        ),
        home: bool = typer.Option(
            False, "--home",
            help="Use home directory as workspace target."
        ),
        yes: bool = typer.Option(
            False, "--yes", "-y",
            help="Skip all confirmation prompts (for non-interactive / AI use)."
        ),
) -> None:
    """
    Initialize a MOSS workspace.

    Interactive mode (default):
        moss ws init          # prompts for path and confirmation

    Non-interactive mode (for scripts and AI):
        moss ws init /path    # explicit path, skips path prompt
        moss ws init /path -y # skip all prompts including confirmation
        moss ws init --cwd -y # current directory, no prompts
        moss ws init --home -y # home directory, no prompts
    """
    env = Environment.discover()
    home_path = env.expect_home_workspace_path()
    cwd_path = env.expect_cwd_workspace_path()

    # 1. Resolve target path
    if path is not None:
        target_path = path.resolve()
    elif cwd:
        target_path = cwd_path
    elif home:
        target_path = home_path
    else:
        # Interactive path selection
        console.print("\n[bold cyan]MOSS Workspace Setup[/bold cyan]")
        console.print(f" 1) Current directory: [dim]{cwd_path}[/dim]")
        console.print(f" 2) Home directory: [dim]{home_path}[/dim]")
        console.print(f" 3) Custom path")

        if is_ai_mode():
            print_error("Interactive mode not available with --ai flag.")
            print_info("Use one of: --cwd, --home, or provide a path argument.")
            print_info("Add --yes to skip confirmation prompts.")
            raise typer.Exit(code=1)

        choice = typer.prompt("\nSelect an option", default="1", type=str)

        if choice == "1":
            target_path = cwd_path
        elif choice == "2":
            target_path = home_path
        elif choice == "3":
            custom_path = typer.prompt("Enter custom path", type=Path)
            target_path = custom_path.resolve()
        else:
            print_error("Invalid selection.")
            raise typer.Exit(code=1)

    # 2. Confirmation
    if target_path.exists():
        is_reinit = (target_path / META_CONFIG_FILENAME).exists()
        if is_reinit:
            if not yes and not typer.confirm(
                f"Directory '{target_path.name}' already exists. Force re-initialize?",
                default=False
            ):
                print_warning("Aborted.")
                return
        else:
            if not yes and not typer.confirm(
                f"Path exists and is not empty. Proceed?",
                default=False
            ):
                print_warning("Aborted.")
                return
    else:
        if not yes and not typer.confirm(
            f"Create new workspace at '{target_path}'?", default=True
        ):
            print_warning("Aborted.")
            return

    # 3. Execute
    print_info(f"Initializing MOSS at: {target_path}")
    try:
        Environment.init_workspace(target_path)
        print_success("Initialization completed successfully.")
        print_info("Next step: use 'moss ws copy-env' to create env file, then configure credentials.")
    except Exception as e:
        print_error(f"Failed to initialize: {e}")
        raise typer.Exit(code=1)


@workspace_app.command(
    name="override",
    short_help="Override an existing workspace with the latest stub files.",
)
def override_workspace(
        yes: bool = typer.Option(
            False, "--yes", "-y",
            help="Skip confirmation prompt (for non-interactive / AI use)."
        ),
) -> None:
    """
    Override the active workspace with the latest stub template files.

    This is used when the MOSS source has been upgraded and you want to
    sync stub changes into an existing workspace. Existing files that match
    the stub will be overwritten; extra user files are left untouched.

    Tip: if your workspace is a git repo, run 'git diff' after override to
    review changes and selectively restore any files you want to keep.
    """
    try:
        env = Environment.discover()
        ws_path = env.workspace_path
    except EnvironmentError as e:
        print_error(f"Environment Discovery Failed: {e}")
        raise typer.Exit(code=1)

    if not ws_path.exists() or not (ws_path / META_CONFIG_FILENAME).exists():
        print_error(f"No existing MOSS workspace found at '{ws_path}'.")
        print_info("Use 'moss ws init' to create a new workspace.")
        raise typer.Exit(code=1)

    if not yes and not typer.confirm(
        f"This will overwrite stub files in '{ws_path.name}' with the latest version.\n"
        f"User-created files will be left untouched. Continue?",
        default=False
    ):
        print_warning("Aborted.")
        return

    print_info(f"Overriding workspace at: {ws_path}")
    try:
        Environment.init_workspace(ws_path, force=True)
        print_success("Override completed. Stub files updated to latest version.")
        print_info("Tip: use 'git diff' to review changes if this workspace is a git repo.")
    except Exception as e:
        print_error(f"Failed to override: {e}")
        raise typer.Exit(code=1)


@workspace_app.command(name="copy-env")
def copy_env(
        force: bool = typer.Option(
            False, "--force", "-f",
            help="Overwrite existing .env file if present."
        ),
) -> None:
    """
    Copy the .env_example to .env in the current active workspace.

    By default, will not overwrite an existing .env file.
    Use --force to overwrite.
    """
    try:
        env = Environment.discover()
        workspace_dir = env.workspace_path
        example_path = env.env_example_file
        target_env = env.env_file

        if not example_path.exists():
            print_error(
                f"Template '{example_path.relative_to(workspace_dir)}' not found in workspace."
            )
            raise typer.Exit(code=1)

        existed = target_env.exists()
        if existed and not force:
            print_warning(
                f"'{target_env.relative_to(workspace_dir)}' already exists. Use --force to overwrite."
            )
            return

        print_info(f"Creating {target_env} from template...")
        shutil.copy(example_path, target_env)

        # file permissions: rw-rw---- (0o660)
        FILE_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP
        os.chmod(target_env, FILE_MODE)

        action = "Overwritten" if existed else "Created"
        print_success(f"{action} {target_env.name}")
        print_info("Group-writable permission set.")

    except EnvironmentError as e:
        print_error(f"Environment Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        print_error(f"Failed to copy env: {e}")
        raise typer.Exit(code=1)
