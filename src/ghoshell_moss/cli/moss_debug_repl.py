import click
from ghoshell_moss.host import Host, Environment
from ghoshell_moss.host.tui_entries.toolset_tui import ToolsetTUI


@click.command()
@click.option(
    '--mode',
    default='default',
    help='设置 MOSS 的运行模式 (例如: default, dev, robot).'
)
@click.option(
    '--scope',
    default='global',
    help='设置当前的会话范围 (session scope).'
)
def moss_debug_repl_main(mode: str, scope: str):
    """
    启动 MOSS ToolSet TUI 调试终端。
    """
    click.echo(f"Starting MOSS Debug REPL in [{mode}] mode, scope: [{scope}]")

    # 初始化环境
    env = Environment.discover()
    env.set_mode(mode)
    env.set_session_scope(scope)

    # 启动 Host 与 TUI
    host = Host(env=env)
    tui = ToolsetTUI(host=host)
    tui.run()


if __name__ == '__main__':
    moss_debug_repl_main()
