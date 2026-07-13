import click
from ghoshell_moss.host import Host
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.host.tui_entries.moss_runtime_ui import MossRuntimeTUI


@click.command()
@click.option(
    '--mode',
    default='default',
    help='设置 MOSS 的运行模式 (例如: default, dev, robot).'
)
@click.option(
    '--scope',
    default='default',
    help='设置当前的会话范围 (session scope).'
)
def moss_debug_repl_main(mode: str, scope: str):
    """
    启动 MOSS ToolSet TUI 调试终端。
    """
    click.echo(f"Starting MOSS Debug REPL in [{mode}] mode, scope: [{scope}]")

    # §UU-1 seal 定案: 入口点显式构造 Environment(**cli_args) + seal, 注册 singleton.
    # Host 只消费 sealed env, 不承担参数收集责任.
    env = Environment(mode=mode, scope=scope)
    env.seal()

    host = Host(env=env)
    tui = MossRuntimeTUI(host=host)
    tui.run()


if __name__ == '__main__':
    moss_debug_repl_main()
