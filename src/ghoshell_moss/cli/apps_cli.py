from typing import List
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
from ghoshell_moss.host.abcd.app import AppInfo
from ghoshell_common.helpers import yaml_pretty_dump
from ghoshell_moss.host import Host
from .utils import console, print_host_mode_info, print_simple_table, print_simple_panel
import subprocess
import shlex
import typer

app_store_app = typer.Typer(
    help="MOSS App Store: Manage and introspect environment applications.",
    no_args_is_help=True
)


@app_store_app.command(name="list")
def list_apps(
        include: List[str] = typer.Argument(None, help="Include patterns (e.g. 'core/*', '*/web')"),
        exclude: List[str] = typer.Option(None, "--exclude", "-e", help="Exclude patterns"),
        json_out: bool = typer.Option(False, "--json", help="Output raw JSON for AI consumption."),
        verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose mode."),
):
    """
    List all discovered apps in the MOSS environment.
    """
    import os
    if include is not None and any(os.path.exists(p) for p in include):
        console.print(
            "[yellow]Warning: Some patterns match local files. Did you forget to use quotes? (e.g., '*/' )[/yellow]")

    host = Host()
    if verbose:
        print_host_mode_info(host)
    # 刷新并获取所有 apps
    all_apps = list(host.apps.list_apps(refresh=True))

    # 调用新的过滤逻辑
    results = list(host.apps.match_apps(all_apps, include, exclude))

    if not results:
        console.print(f"[yellow]No apps found matching: '{include}'[/yellow]")
        return

    # AI 模式输出
    if json_out:
        data = [app.model_dump() for app in results]
        console.json(data=data)
        return

    _display_app_table(results, is_filtered=bool(include))
    if verbose:
        console.print(f"[dim]App store: {host.apps.app_store_directory}[/dim]")


@app_store_app.command(name="show")
def show_app(
        fullname: str = typer.Argument(..., help="The full address of the app (e.g., group/name)"),
        json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
        verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose mode."),
):
    """
    Show detailed information of a specific app by its address.
    """
    host = Host()
    if verbose:
        print_host_mode_info(host)

    app = host.apps.get_app_info(fullname)

    if not app:
        console.print(f"[red]Error: App with fullname '{fullname}' not found.[/red]")
        raise typer.Exit(code=1)

    if json_out:
        console.json(data=app.model_dump())
        return

    _display_app_detail(app)
    if verbose:
        console.print(f"[dim]App store: {host.apps.app_store_directory}[/dim]")


def _display_app_table(apps: List[AppInfo], is_filtered: bool):
    """展示 App 概览表格"""
    title = "MOSS App Store"
    if is_filtered:
        title += " (Filtered)"

    # 准备表格数据
    table_data = []
    for app in sorted(apps, key=lambda x: x.address):
        table_data.append([
            f"[cyan]{app.group}[/cyan]",
            f"[cyan]{app.fullname}[/cyan]",
            app.description.split('\n')[0] if app.description else ""
        ])

    # 使用简洁表格显示
    print_simple_table(
        data=table_data,
        headers=["Group", "Fullname", "Description"],
        title=title,
        column_styles=["cyan", "cyan", ""],
        title_style="bold green",
    )

    console.print(f"\n[dim]Total: {len(apps)} apps discovered.[/dim]")
    console.print(f"[dim]Hint: Use [bold]moss apps show <fullname>[/bold] for more detail.[/dim]")


def _display_app_detail(app: AppInfo):
    """展示 App 的深度细节"""
    # 使用简洁面板显示基本信息
    content = (
        f"Group: [dim]{app.group}[/dim]\n"
        f"Name: [dim]{app.name}[/dim]\n"
        f"Description: [dim]{app.description}[/dim]\n"
        f"Directory: [dim]{app.work_directory}[/dim]\n"
        f"Address: [dim]{app.address}[/dim]"
    )
    print_simple_panel(content, title=app.fullname)

    # 启动配置 (Circus Params)
    console.print("\n[bold]Execution Config (Watcher):[/bold]")
    watcher = app.watcher.model_dump(exclude_defaults=False, exclude_none=False)
    watcher_yaml = yaml_pretty_dump(watcher)
    console.print(Syntax(watcher_yaml, "yaml", theme="monokai", background_color="default"))

    # 错误信息
    if app.error:
        console.print(f"\n[bold red]Last Error:[/bold red]")
        console.print(Panel(app.error, border_style="red"))
    if app.docstring:
        console.print(Panel(Markdown(app.docstring), title='docstring'))


@app_store_app.command(name="test")
def test_app(
        fullname: str = typer.Argument(..., help="The app fullname (group/name) to test."),
        args: str = typer.Argument("", help="Additional arguments passed to the app command."),
        verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose mode."),
        mode: str | None = typer.Option(None, "-m", "--mode", help="specific Mode"),
):
    """
    Start an app as a foreground subprocess for debugging/testing.
    This bypasses the AppStore runtime (Circus).
    """
    host = Host(mode=mode)
    print_host_mode_info(host)

    # 1. 获取 AppInfo
    app = host.apps.get_app_info(fullname)
    if not app:
        console.print(f"[red]Error: App '{fullname}' not found.[/red]")
        raise typer.Exit(1)

    # 2. 准备执行指令
    # 结合 AppWatcher 定义的 cmd 和 命令行传入的 args
    full_cmd = f"{app.watcher.cmd} {args}".strip()

    console.print(Panel(
        f"[bold green]Testing App:[/bold green] {app.fullname}\n"
        f"[bold blue]Directory:[/bold blue] {app.work_directory}\n"
        f"[bold blue]Address:[/bold blue] {app.address}\n"
        f"[bold yellow]Command:[/bold yellow] {full_cmd}",
        title="Debug Mode",
        border_style="bright_black"
    ))

    # 3. 执行子进程
    # 我们需要切换到 App 的工作目录执行
    try:
        # 使用 shlex.split 确保命令解析安全（处理空格等）
        cmd_args = shlex.split(full_cmd)

        # 继承当前环境并注入 Host 特有的 env (如果有)
        env = host.env.dump_moss_env(cell_address=app.address, for_child_process=True)
        # 这里可以根据需要注入 host.env_vars() 等信息

        console.print("[dim]—— Process Started (Ctrl+C to stop) ——[/dim]\n")

        subprocess.run(
            cmd_args,
            cwd=app.work_directory,
            env=env,
            check=False,  # 允许非零退出码，不抛出 Python 异常
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user.[/yellow]")
    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"\n[red]Failed to start test process: {e}[/red]")
        raise typer.Exit(1)
    finally:
        console.print("\n[dim]—— Test Session Ended ——[/dim]")


import inspect
import typer
from rich.table import Table
from rich.syntax import Syntax
from .utils import console

