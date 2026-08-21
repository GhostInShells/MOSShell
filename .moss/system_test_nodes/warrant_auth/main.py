"""Warrant Auth — startup authorization via matrix.warrant (as a channel command).

Start:  moss nodes run .moss/system_test_nodes/warrant_auth/
Debug:  python main.py

The node provides a channel with an `authorize` command. Invoking it triggers
the warrant authorization: it asks a human (via QA watcher / `moss nodes
answer-node --namespace _warrant`) to approve. The result is returned by the
command, so the authorization can be re-triggered without restarting the node.
"""

from pydantic import BaseModel

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.warrant import Permission, AuthorizationResult
from ghoshell_moss.core.concepts.qa import Question, Answer


class StartupState(BaseModel):
    approved: bool = False


class StartupPermission(Permission[StartupState]):
    """Ask a human to approve this node's startup."""

    @classmethod
    def key(cls) -> str:
        return "warrant_auth.startup"

    @classmethod
    def type(cls) -> str:
        return "warrant_auth.startup"

    def default(self) -> StartupState:
        return StartupState(approved=False)

    def check(self, state: StartupState) -> Question | None:
        return Question(content="Allow node 'warrant_auth' to start?", kind="apply")

    def replied(self, answer: Answer) -> AuthorizationResult[StartupState]:
        approved = not answer.rejected
        return AuthorizationResult(
            allowed=approved,
            state=StartupState(approved=approved),
            reason=None if approved else "startup denied by watcher",
        )


async def main(matrix: Matrix):
    print("[warrant_auth] node started, providing channel", flush=True)

    warrant = matrix.warrant
    print(f"[warrant_auth] matrix.warrant running={warrant.is_running()}", flush=True)

    from ghoshell_moss.core.blueprint.channel_builder import new_channel
    channel = new_channel(name="warrant_auth", description="Warrant-authorized demo node")

    @channel.build.command()
    async def authorize() -> str:
        """Request startup authorization — a human must approve via answer-node.

        Returns the authorization result (allowed / reason).
        """
        print("[warrant_auth] authorize() called — requesting authorization", flush=True)
        result = await warrant.require(StartupPermission())
        print(
            f"[warrant_auth] require returned: allowed={result.allowed} reason={result.reason}",
            flush=True,
        )
        return f"allowed={result.allowed} reason={result.reason}"

    @channel.build.command()
    async def ping() -> str:
        """Reply pong — proves the node is alive."""
        return "pong"

    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
