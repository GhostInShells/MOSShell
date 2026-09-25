"""QA Watcher — watches for approval questions and auto-approves them.

Start:  moss nodes run .moss/system_test_nodes/qa_watcher/
Debug:  python main.py

Run together with qa_asker in another terminal to verify cross-process QA.
"""

import asyncio

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.concepts.qa import QAManager


def _short_qid(qa) -> str:
    meta = qa.question.meta
    return meta.id[:8] if meta else "?"


async def main(matrix: Matrix):
    qa_mgr = matrix.container.fetch(QAManager)

    async with qa_mgr:
        watcher = qa_mgr.watch("dogfood/qa")

        def on_question(qa):
            matrix.logger.info(
                "[qa_watcher] received qa %s: %s", _short_qid(qa), qa.question.content
            )
            answer = qa.question.approve("auto-approved")
            qa.reply(answer)
            matrix.logger.info("[qa_watcher] replied to qa %s", _short_qid(qa))

        watcher.on_question(on_question)
        matrix.logger.info("[qa_watcher] watching dogfood/qa, Ctrl-C to stop")

        # Keep alive — watcher callbacks fire on zenoh subscriber thread
        try:
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            matrix.logger.info("[qa_watcher] shutting down")


if __name__ == "__main__":
    Matrix.discover().run(main)
