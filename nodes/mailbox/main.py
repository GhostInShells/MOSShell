"""Mailbox Node — Ghost-MCP communication bridge.

Start:  moss nodes run <path-to-this-dir>    # via CLI
Debug:  python main.py                        # ad-hoc

The mailbox node enables bidirectional request-reply between an external
MCP agent (e.g. Claude Code) and a MOSS ghost via the Matrix signal bus.

  Agent -> send(message) -> Signal -> Ghost mindflow
  Ghost -> mailbox:reply(id, content) -> reply buffer -> agent.poll(task_id)

Thin shell over ``ghoshell_moss_contrib.nodes.mailbox`` — see that module
for the full implementation.
"""

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss_contrib.nodes.mailbox import MailboxBridge, serve_mailbox


async def main(matrix: Matrix) -> None:
    bridge = MailboxBridge()
    await serve_mailbox(matrix, bridge, port=20774)


if __name__ == "__main__":
    Matrix.discover().run(main)
