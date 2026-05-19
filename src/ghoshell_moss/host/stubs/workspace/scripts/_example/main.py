"""Example script: send a test signal to the running ghost."""
from ghoshell_moss.core.blueprint.mindflow import Signal
from ghoshell_moss.core.blueprint.matrix import Matrix


async def main():
    """Send a simple text signal into the ghost's perception loop."""
    async with Matrix.discover() as matrix:
        signal = Signal.new(
            name="input",
            description="Hello from script cell!",
            priority=Signal.Priority.NOTICE,
        )
        matrix.session.add_signal(signal)
        print(f"Signal sent: {signal.id}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
