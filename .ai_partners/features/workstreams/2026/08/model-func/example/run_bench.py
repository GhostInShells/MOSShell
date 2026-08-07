"""Run utterance end detection benchmark.

Usage from repo root:
    PYTHONPATH=.ai_partners/features/workstreams/2026/08/model-func/example \
    .venv/bin/python .ai_partners/features/workstreams/2026/08/model-func/example/run_bench.py
"""

import asyncio
import sys
from pathlib import Path

_CWD = Path(__file__).resolve().parent
sys.path.insert(0, str(_CWD))


def _build_meta() -> "BenchmarkMeta":
    from ghoshell_moss.contracts.llms import BenchmarkMeta

    return BenchmarkMeta(
        title="utterance-end-detection",
        description="Flash model single-call utterance completeness detection",
        result_type="models:UtteranceEndScore",
        instruction=(
            "You are a conversation end detector. Rate how complete each utterance "
            "is on a 0-9 scale. 0 = clearly incomplete (mid-word, cut off). "
            "5 = ambiguous. 9 = clearly a complete thought. "
            "Return the score and a one-sentence reason."
        ),
        cases_file="cases.jsonl",
    )


def _resolve_model() -> "ResolvedModel":
    from ghoshell_moss.contracts.llms import LLMConfig
    return LLMConfig().resolve().get_model()


async def main() -> None:
    from ghoshell_moss.llms.funcs import PydanticAIFuncs

    meta = _build_meta()
    model = _resolve_model()
    print(f"model:  {model.service.name}/{model.model.model}")
    print(f"bench:  {meta.title}  ({meta.cases_file})")
    print(f"cases:  {meta.result_type}")
    print()

    funcs = PydanticAIFuncs()
    record = await funcs.run_benchmark(meta, model, cwd=_CWD)

    for i, r in enumerate(record.results):
        s = r.result.get("score", "?") if r.result else "?"
        reason = r.result.get("reason", "") if r.result else ""
        print(f"[{s}] {reason}  ({r.cast:.2f}s, {r.usage.get('output_tokens',0)} tok)")

    total = sum(r.cast for r in record.results)
    print(f"\n{len(record.results)} calls, total {total:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
