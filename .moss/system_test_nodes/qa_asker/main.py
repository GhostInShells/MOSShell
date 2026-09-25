"""QA Asker — issues approval questions periodically via ZenohQAManager from IoC.

Start:  moss nodes run .moss/system_test_nodes/qa_asker/
Debug:  python main.py

Run together with qa_watcher in another terminal to verify cross-process QA.
"""

import asyncio

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.concepts.qa import QAManager


def _short_qid(qa) -> str:
    meta = qa.question.meta
    return meta.id[:8] if meta else "?"


async def main(matrix: Matrix):
    qa_mgr = matrix.container.fetch(QAManager)
    count = 0

    async with qa_mgr:
        asker = qa_mgr.asker("dogfood/qa")
        matrix.logger.info("[qa_asker] started, issuing questions every 5s")

        while True:
            count += 1
            qa = asker.ask_approval(f"approve request #{count}?")
            matrix.logger.info(
                "[qa_asker] issued qa %s: %s", _short_qid(qa), qa.question.content
            )

            await qa.wait()
            if qa.done() and not qa.canceled():
                matrix.logger.info(
                    "[qa_asker] qa %s accepted: %s",
                    _short_qid(qa), qa.answer.content if qa.answer else "(no content)",
                )
            elif qa.canceled():
                matrix.logger.info(
                    "[qa_asker] qa %s canceled: %s",
                    _short_qid(qa), qa.answer.content if qa.answer else "(no content)",
                )

            await asyncio.sleep(5.0)


if __name__ == "__main__":
    Matrix.discover().run(main)
