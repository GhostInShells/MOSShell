"""Run a benchmark from a bench.md + cases.jsonl directory.

Usage:
    PYTHONPATH=<bench-dir> .venv/bin/python example/run_bench.py <bench-dir>

If no directory is given, defaults to:
    .ai_partners/benchmarks/utterance-end-detection
"""

import asyncio
import sys
from pathlib import Path

import yaml


def _parse_bench_md(dir_: Path) -> "BenchmarkMeta":
    """Parse bench.md YAML frontmatter into BenchmarkMeta."""
    from ghoshell_moss.contracts.llms import BenchmarkMeta

    text = (dir_ / "bench.md").read_text(encoding="utf-8")
    # extract YAML frontmatter between --- delimiters
    if not text.startswith("---"):
        raise ValueError("bench.md must start with YAML frontmatter (---)")
    _, fm, body = text.split("---", 2)
    data = yaml.safe_load(fm) or {}
    data.setdefault("description", body.strip())
    return BenchmarkMeta(**data)


def _resolve_model() -> "ResolvedModel":
    from ghoshell_moss.contracts.llms import LLMConfig
    return LLMConfig().resolve().get_model()


async def main(dir_: Path) -> None:
    from ghoshell_moss.llms.pydantic_ai_adapter.funcs import PydanticAIFuncs

    sys.path.insert(0, str(dir_))  # make models.py importable
    meta = _parse_bench_md(dir_)
    model = _resolve_model()
    print(f"model:  {model.service.name}/{model.model.model}")
    print(f"bench:  {meta.title}")
    print(f"result: {meta.result_type}  ({meta.cases_file})")
    print()

    funcs = PydanticAIFuncs()
    record = await funcs.run_benchmark(meta, model, cwd=dir_)

    for i, r in enumerate(record.results):
        s = r.result.get("score", "?") if r.result else "?"
        reason = r.result.get("reason", "") if r.result else ""
        print(f"[{s}] {reason}  ({r.cast:.2f}s, {r.usage.get('output_tokens',0)} tok)")

    total = sum(r.cast for r in record.results)
    print(f"\n{len(record.results)} calls, total {total:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        bench_dir = Path(sys.argv[1]).resolve()
    else:
        bench_dir = Path(".ai_partners/benchmarks/utterance-end-detection").resolve()
    asyncio.run(main(bench_dir))
