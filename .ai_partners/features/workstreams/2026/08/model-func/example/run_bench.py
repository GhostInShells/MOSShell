"""Run a benchmark from a bench.md + cases.jsonl directory.

Usage:
    PYTHONPATH=<bench-dir> .venv/bin/python example/run_bench.py <bench-dir> [flags]

Strategy A/B — the scoring hint (e.g. rubric.txt) is a placement variable:
    baseline           minimal instruction, no hint
    hint-in-instruction  --instruction <rubric>
    hint-in-thinking     --thinking <rubric>
    both                 --instruction <task> --thinking <rubric>
"""

import asyncio
import sys
from pathlib import Path

import yaml

from ghoshell_moss.contracts.llms import BenchmarkMeta


def _parse_bench_md(dir_: Path) -> BenchmarkMeta:
    """Parse bench.md YAML frontmatter into BenchmarkMeta."""
    text = (dir_ / "bench.md").read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError("bench.md must start with YAML frontmatter (---)")
    _, fm, body = text.split("---", 2)
    data = yaml.safe_load(fm) or {}
    data.setdefault("description", body.strip())
    return BenchmarkMeta(**data)


def _read_text_or_file(value: str, base: Path) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return value


def _resolve_model() -> "ResolvedModel":
    from ghoshell_moss.contracts.llms import LLMConfig
    return LLMConfig().resolve().get_model()


async def main(
        dir_: Path,
        *,
        thinking: str | None,
        instruction: str | None,
        effort: str | None,
) -> None:
    from ghoshell_moss.contracts.llms import Effort
    from ghoshell_moss.llms.pydantic_ai_adapter.funcs import PydanticAIFuncs

    sys.path.insert(0, str(dir_))  # make models.py importable
    meta = _parse_bench_md(dir_)
    if instruction is not None:
        meta = meta.model_copy(update={"instruction": _read_text_or_file(instruction, dir_)})
    thinking_text = _read_text_or_file(thinking, dir_) if thinking else None
    effort_val: Effort | None = effort if effort else None
    model = _resolve_model()
    print(f"model:   {model.service.name}/{model.model.model}")
    print(f"bench:   {meta.title} ({meta.version})")
    print(f"strategy: instruction={len(meta.instruction)}ch thinking={'Y' if thinking_text else 'N'} effort={effort_val}")
    print()

    funcs = PydanticAIFuncs()
    record = await funcs.run_benchmark(
        meta, model, cwd=dir_, effort=effort_val, thinking=thinking_text,
    )

    for i, r in enumerate(record.results):
        s = r.result.get("score", "?") if r.result else "?"
        print(f"[{s}]  ({r.cast:.2f}s, {r.usage.get('output_tokens',0)} tok)")
        _ = i

    total = sum(r.cast for r in record.results)
    print(f"\n{len(record.results)} calls, total {total:.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("bench_dir", nargs="?", default=".ai_partners/benchmarks/utterance-end-detection")
    parser.add_argument("--thinking", default=None, help="thinking block (text or file) — hint in 内观")
    parser.add_argument("--instruction", default=None, help="override instruction (text or file) — hint in system prompt")
    parser.add_argument("--effort", default=None, help="thinking effort (none..max)")
    args = parser.parse_args()

    asyncio.run(main(
        Path(args.bench_dir).resolve(),
        thinking=args.thinking,
        instruction=args.instruction,
        effort=args.effort,
    ))
