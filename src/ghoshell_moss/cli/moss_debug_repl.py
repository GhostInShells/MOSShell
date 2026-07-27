import click
from ghoshell_moss.host import Host
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.host.tui_entries.moss_runtime_ui import MossRuntimeTUI


@click.command()
@click.option(
    '--mode',
    default='default',
    help='MOSS runtime mode (e.g. default, dev, robot).',
)
@click.option(
    '--scope',
    default='default',
    help='Network scope for session isolation.',
)
@click.option(
    '--network',
    default='local',
    help='Network driver.',
)
def moss_debug_repl_main(mode: str, scope: str, network: str):
    """
    MOSS Shell runtime debugger — interactive TUI for testing CTML,
    inspecting channels, and debugging the MOSS runtime before a Ghost runs.
    """
    click.echo(f"Starting MOSS Shell debugger in [{mode}] mode, scope: [{scope}]")

    # §UU-1 seal 定案: 入口点显式构造 Environment(**cli_args) + seal, 注册 singleton.
    # Host 只消费 sealed env, 不承担参数收集责任.
    env = Environment(mode=mode, scope=scope)
    env.seal()

    host = Host(env=env)
    tui = MossRuntimeTUI(host=host)
    tui.run()


if __name__ == '__main__':
    moss_debug_repl_main()
