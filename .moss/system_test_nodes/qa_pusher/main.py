"""QA Pusher — issues varied questions, awaits answers, prints summary.

Start:  moss nodes run .moss/system_test_nodes/qa_pusher/
Debug:  python main.py

Run together with moss-ghost or moss-shell to answer via TUI (C-q).
"""

import asyncio

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.concepts.qa import QAManager


def _short_qid(qa) -> str:
    meta = qa.question.meta
    return meta.id[:8] if meta else "?"


async def main(matrix: Matrix):
    qa_mgr = matrix.container.fetch(QAManager)
    results: list[dict] = []

    async with qa_mgr:
        asker = qa_mgr.asker("")
        print("[qa_pusher] started, issuing questions...\n")

        # 1. confirm
        qa1 = asker.ask_confirm(
            "Deploy v2.1.0 to production?",
            yes="Proceed with deployment",
            no="Hold off",
            markdown="## Deployment Summary\n- 15 files changed\n- 3 services affected\n- Rollback: `moss deploy --rollback`",
        )
        print(f"[qa_pusher] [1/4] confirm  #{_short_qid(qa1)}: {qa1.question.content}")
        await qa1.wait()
        results.append(_record(qa1, "confirm"))

        # 2. input with suggestions
        qa2 = asker.ask(
            "Name the new staging environment:",
            suggestions=["staging-east", "staging-west", "staging-canary"],
            markdown="DNS-safe name. Tab to complete from suggestions.",
        )
        print(f"[qa_pusher] [2/4] input    #{_short_qid(qa2)}: {qa2.question.content}")
        await qa2.wait()
        results.append(_record(qa2, "input"))

        # 3. choose
        qa3 = asker.ask_choose(
            "Select log level:",
            options={
                "debug": "Debug — verbose, includes trace",
                "info": "Info — standard production level",
                "warn": "Warn — errors and warnings only",
            },
            default="info",
            markdown="Applies to all cells in the environment.",
        )
        print(f"[qa_pusher] [3/4] choose   #{_short_qid(qa3)}: {qa3.question.content}")
        await qa3.wait()
        results.append(_record(qa3, "choose"))

        # 4. select (multi) with markdown
        qa4 = asker.ask_select(
            "Enable features for this environment:",
            options={
                "logging": "Structured JSON logging to stdout",
                "metrics": "Prometheus metrics on :9090",
                "tracing": "OpenTelemetry traces to collector",
                "alerts": "PagerDuty alert integration",
            },
            min_select=1,
            max_select=3,
            default=["logging"],
            markdown="## Feature Gates\nSelect 1-3 features. Logging is recommended.",
        )
        print(f"[qa_pusher] [4/4] select   #{_short_qid(qa4)}: {qa4.question.content}")
        await qa4.wait()
        results.append(_record(qa4, "select"))

        # summary
        print("\n── QA Pusher Results ──")
        for r in results:
            print(_format_result(r))
        print("── done ──")


def _record(qa, label: str) -> dict:
    answer = qa.answer
    return {
        "label": label,
        "qid": _short_qid(qa),
        "content": qa.question.content[:60],
        "canceled": qa.canceled(),
        "answer_content": answer.content if answer else "",
        "answer_choices": answer.choices if answer else [],
    }


def _format_result(r: dict) -> str:
    status = "CANCELED" if r["canceled"] else "OK"
    extra = ""
    if r["answer_choices"]:
        extra = f" choices={r['answer_choices']}"
    if r["answer_content"]:
        extra += f" content={r['answer_content']!r}"
    return f"  [{status}] {r['label']} #{r['qid']}{extra}"


if __name__ == "__main__":
    Matrix.discover().run(main)
