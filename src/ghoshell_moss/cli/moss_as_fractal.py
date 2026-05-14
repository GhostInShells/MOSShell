"""MOSS Fractal 分形协议 — CLI 入口.

将 MOSS Runtime 通过 Fractal 协议反向注册到父节点。

用法:
  moss-as-fractal --mode robot_arm --transport "tcp/192.168.1.100:20770"
"""

import asyncio
import logging
from datetime import datetime

import click

from ghoshell_moss.host import Host
from ghoshell_moss.core.blueprint.environment import Environment


class ClickHandler(logging.Handler):
    """将 log 记录通过 click.echo 输出到控制台。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            click.echo(msg)
        except Exception:
            self.handleError(record)


@click.command()
@click.option('--mode', default='default', help='MOSS 运行时模式')
@click.option('--session-scope', default='default', help='Session 作用域')
@click.option(
    '--transport', default=None,
    help='父节点 zenoh 端点，如 tcp/192.168.1.100:20770。为空使用 peer 多播自发现',
)
def main(mode: str, session_scope: str, transport: str | None):
    """启动 MOSS Runtime 并以 Fractal 分形协议反向注册到父节点。

    子节点（如机器人开发板）通过此命令主动连接到父节点（如 Mac），
    将本地的 shell.main_channel 提供给父节点使用。"""

    env = Environment.discover()
    if mode:
        env.set_mode(mode)
    if session_scope:
        env.set_session_scope(session_scope)

    host = Host(env=env)

    # 配置日志输出
    _logger = host.matrix().logger
    if isinstance(_logger, logging.Logger):
        _handler = ClickHandler()
        _handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
        ))
        _logger.addHandler(_handler)
        _logger.setLevel(logging.DEBUG)

    click.echo("")
    click.echo("=" * 52)
    click.echo(f"  Fractal Node Provider")
    click.echo(f"  Mode         : {mode}")
    click.echo(f"  Transport    : {transport or 'peer multicast'}")
    click.echo(f"  Session Scope: {session_scope}")
    click.echo(f"  Started at   : {datetime.now().strftime('%H:%M:%S')}")
    click.echo("=" * 52)
    click.echo("")

    try:
        # 使用 MossHost 提供的 provide_moss_as_fractal 方法
        # 它会自动从 IoC 容器获取 FractalNodeProvider
        asyncio.run(host.provide_moss_as_fractal(provider=None))
    except KeyboardInterrupt:
        click.echo("\nStopped.")
    except NotImplementedError as e:
        click.echo(f"\nError: {e}")
        click.echo("请确保 workspace 中已注册 ZenohSessionFractalNodeProvider。")
