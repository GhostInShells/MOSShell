"""Sandbox safety gate — 1-token interaction efficiency probe.

Measures the end-to-end latency of a single-token string round-trip:
fixed instruction (policy) + a code block -> `ok`. Safe cases only — the
gate should trivially emit `ok`. The signal is efficiency (latency /
throughput), NOT security discrimination.

Usage:
    .venv/bin/python run.py [--n N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INSTRUCTION = (
    "You are a safety gate. The code must define a single `async def main(): ...`.\n"
    "Reply with the single token `ok` if acceptable to run, otherwise `deny`\n"
    "(you may add one short reason). Do not judge logical correctness.\n"
    "When unsure, reply `deny`."
)


def classify(raw: str) -> str:
    """Ternary verdict from the raw string output.

    pass      = ok-prefixed
    reject    = deny-prefixed (explicit denial)
    exception = anything else (empty / ambiguous / garbled)
    """
    s = raw.strip().lower()
    if s.startswith("ok"):
        return "pass"
    if s.startswith("deny"):
        return "reject"
    return "exception"


def _resolve_model() -> "ResolvedModel":
    from ghoshell_moss.contracts.llms import LLMConfig
    # Deferred: no --tag selection. LLMConfig.get_model(tag=...) with no
    # provider/model short-circuits to _get_default() and silently DROPS the
    # tag (the unwrap path only runs when a provider is named). Trigger: when
    # the gate must pin a cheap model, add --tag and resolve via the default
    # provider's unwrap_tag instead of get_model(tag=...).
    return LLMConfig().resolve().get_model()


def _load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


async def main(n: int) -> None:
    from ghoshell_moss.llms.pydantic_ai_adapter.funcs import PydanticAIFuncs

    model = _resolve_model()
    funcs = PydanticAIFuncs()
    cases = _load_cases(HERE / "cases.jsonl")

    print(f"model:  {model.service.name}/{model.model.model}")
    print(f"cases:  {len(cases)} x {n} = {len(cases) * n} calls")
    print()

    total = 0.0
    verdicts: dict[str, int] = {}
    for case in cases:
        for _ in range(n):
            result = await funcs.call(
                instruction=INSTRUCTION,
                prompt=case["prompt"],
                result_type=None,
                model=model,
            )
            verdict = classify(result.content)
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            total += result.cast
            print(
                f"[{verdict:9}] {result.cast:6.2f}s  "
                f"{case['label']:14} -> {result.content!r}"
            )

    calls = len(cases) * n
    print()
    print(f"{calls} calls, total {total:.2f}s, avg {total / calls:.2f}s/call")
    print(f"verdicts: {verdicts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1, help="repeat each case N times")
    args = parser.parse_args()
    asyncio.run(main(args.n))
