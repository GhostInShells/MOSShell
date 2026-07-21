"""MOSS node cell entry point.

Start:  moss nodes run <path-to-this-dir>    # via CLI (foreground, CLI is owner)
Debug:  python main.py                        # ad-hoc launch (from_proc identity)

Explore:
    moss codex get-interface ghoshell_moss.core.blueprint.cell:NodeManifest
    moss codex blueprint channel_builder
    moss codex blueprint matrix
    moss ctml read
"""

from ghoshell_moss.core.blueprint.matrix import Matrix


import asyncio
from ghoshell_moss.core.concepts.topic import TopicModel
from pydantic import Field


class HeartbeatTopic(TopicModel):
    """Dogfood heartbeat — producer publishes every second, consumer prints."""
    count: int = Field(default=0)
    message: str = Field(default="")

    @classmethod
    def topic_type(cls) -> str:
        return "dogfood/heartbeat"

    @classmethod
    def default_topic_name(cls) -> str:
        return "dogfood/heartbeat"


async def main(matrix: Matrix):
    publisher = matrix.session.topics.publisher(
        creator=matrix.this.address,
        topic_name="dogfood/heartbeat",
    )
    count = 0
    async with publisher:
        while True:
            count += 1
            topic = HeartbeatTopic(
                count=count,
                message=f"heartbeat #{count}",
            )
            publisher.pub(topic)
            matrix.logger.info(f"[producer] published heartbeat #{count}")
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    Matrix.discover().run(main)
