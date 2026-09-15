"""Utterance end detection — plain-text single-token 0-9 probe.

Measures whether a fast model can emit a BARE 0-9 completeness digit (no
prose) for a live-transcript utterance, and whether the digit lands in the
right band (low 0-3 / mid 4-6 / high 7-9).

Deliberately deviates from the structured benchmark convention: no
``result_type`` / ``models.py``. The constraint IS the signal — structured
output hides the ~100-token cost of tool-call decoding; a plain single
token costs n input tokens + 2 output tokens.

Usage:
    .venv/bin/python run.py [--n N] [--tag small_fast_model]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_instruction() -> str:
    return (HERE / "instruction.txt").read_text(encoding="utf-8")


_DIGIT_RE = re.compile(r"\d")


def parse(raw: str) -> tuple[int | None, bool]:
    """Extract the 0-9 score from the raw output.

    Returns ``(score, clean)``:
    - ``clean=True``: the whole output is exactly one digit 0-9 (modulo
      surrounding whitespace) — the constraint holds.
    - ``clean=False``: the output carries extra prose but still contains a
      digit — constraint violated, score salvaged for accuracy.
    - ``score=None``: no digit at all — unusable.
    """
    s = raw.strip()
    if re.fullmatch(r"[0-9]", s):
        return int(s), True
    m = _DIGIT_RE.search(s)
    if m is not None:
        return int(m.group(0)), False
    return None, False


def band(score: int) -> str:
    if score <= 3:
        return "low"
    if score <= 6:
        return "mid"
    return "high"


def _build_prompt(case: dict) -> str:
    ctx = case.get("context") or ""
    inp = case.get("input") or ""
    return f"<context>\n{ctx}\n</context>\n<input>\n{inp}\n</input>"


def _load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def _load_funcs():
    from ghoshell_moss.core.blueprint.project import Project
    from ghoshell_moss.contracts.llms import LLMFuncs

    project = Project.discover()
    project.bootstrap()
    return project.container.force_fetch(LLMFuncs)


async def main(n: int, tag: str | None) -> None:
    from ghoshell_moss.contracts.llms import CallSettings

    funcs = _load_funcs()
    instruction = _load_instruction()
    cases = _load_cases(HERE / "cases.jsonl")

    caller = funcs.caller(
        instruction=instruction,
        tag=tag,
        settings=CallSettings(max_output_tokens=2),
    )

    print(f"cases:  {len(cases)} x {n} = {len(cases) * n} calls")
    print()

    total = 0.0
    clean = 0
    correct = 0
    scored = 0
    tok_in = 0
    tok_cache_read = 0
    tok_cache_write = 0
    tok_out = 0
    rows: list[tuple[str, str, str, float, str, str]] = []
    for case in cases:
        for _ in range(n):
            result = await caller.run(_build_prompt(case))
            score, is_clean = parse(result.content or "")
            total += result.cast
            u = result.usage or {}
            i_tok = u.get("input_tokens", 0)
            r_tok = u.get("cache_read_tokens", 0)
            w_tok = u.get("cache_write_tokens", 0)
            o_tok = u.get("output_tokens", 0)
            tok_in += i_tok
            tok_cache_read += r_tok
            tok_cache_write += w_tok
            tok_out += o_tok
            if is_clean:
                clean += 1
            if score is not None:
                scored += 1
                got = band(score)
                if got == case["band"]:
                    correct += 1
                verdict = f"{score}({got})"
                status = "ok" if got == case["band"] else "MISS"
            else:
                verdict = "garbled"
                status = "MISS"
            tok = f"in={i_tok} cache_r={r_tok} cache_w={w_tok} out={o_tok}"
            rows.append((case["class"], case["label"], verdict, result.cast, status, tok))

    for cls, label, verdict, cast, status, tok in rows:
        print(f"[{status:4}] {cast:6.2f}s  {cls:26} {label:22} -> {verdict:10}  {tok}")

    calls = len(cases) * n
    print()
    print(f"{calls} calls, total {total:.2f}s, avg {total / calls:.2f}s/call")
    print(f"constraint: {clean}/{calls} clean single-digit ({clean / calls:.0%})")
    if scored:
        print(f"band accuracy: {correct}/{scored} ({correct / scored:.0%})")
    print(
        f"tokens: in={tok_in} cache_read={tok_cache_read} cache_write={tok_cache_write} "
        f"out={tok_out}  (cache_read/total_in = {tok_cache_read / tok_in:.0%})"
        if tok_in else "tokens: n/a"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1, help="repeat each case N times")
    parser.add_argument("--tag", type=str, default="small_fast_model", help="model tag")
    args = parser.parse_args()
    asyncio.run(main(args.n, args.tag))
