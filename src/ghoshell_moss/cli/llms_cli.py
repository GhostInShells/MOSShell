"""moss llms — inspect and call LLM configs.

- ``list``: model meta + config file path + key env var presence. Never
  resolves env vars — no secret values on the command line.
- ``call`` / ``test``: one-shot pydantic-ai calls for availability
  verification. Registered only when the ``ghost`` extra (pydantic-ai) is
  installed — commands with no dependency are not shown.
"""

import asyncio
import importlib.util
import os
from pathlib import Path

import typer

from ghoshell_moss.contracts.llms import LLMConfig, ResolvedModel, ModelRef

from .utils import (
    print_simple_table,
    print_error,
    print_info,
    print_success,
    echo,
)

llms_app = typer.Typer(
    help="Inspect and call LLM configs.",
    no_args_is_help=True,
)


def _load_config() -> LLMConfig:
    """Read LLMConfig — workspace config if present, else the default. Read-only."""
    try:
        from ghoshell_moss.core.blueprint.project import Project
        store = Project.discover().configs
        try:
            return store.get(LLMConfig)
        except Exception:
            return LLMConfig()
    except Exception:
        return LLMConfig()


def _config_source_path() -> str | None:
    try:
        from ghoshell_moss.core.blueprint.project import Project
        path = Path(Project.discover().configs.get_config_path(LLMConfig.conf_name()))
        return path.as_posix()
    except Exception:
        return None


def _collect_env_refs(conf: LLMConfig) -> list[str]:
    """Collect ``$ENV_VAR`` references from the config (presence check only)."""
    refs: set[str] = set()
    for service in conf.services:
        for field in (service.api_key, service.base_url):
            if isinstance(field, str) and field.startswith("$"):
                refs.add(field[1:])
    return sorted(refs)


@llms_app.command(
    name="list",
    short_help="List configured LLM providers/models (never resolves env vars).",
)
def list_models_cmd(
        provider: str = typer.Option(
            "", "--provider", help="Filter by provider/service name.",
        ),
) -> None:
    """List configured LLM models — meta + file path + env presence only.

    绝不 resolve 环境变量: api_key/base_url 的 $ENV_VAR 原样展示, 不打印解析值.
    """
    conf = _load_config()
    models = conf.list_models(provider)

    rows = []
    for resolved in models:
        ref = ModelRef.from_resolved(resolved)
        rows.append([
            ref.service,
            ref.protocol,
            ref.model,
            ref.description or "-",
            ",".join(sorted(ref.tags)) or "-",
            ",".join(ref.content_types) or "*",
            str(ref.max_output_tokens),
        ])
    print_simple_table(
        rows,
        headers=["service", "protocol", "model", "description", "tags", "content_types", "max_out"],
        title="LLM Models",
    )

    env_rows = [
        [ref, "set" if os.environ.get(ref) else "missing"]
        for ref in _collect_env_refs(conf)
    ]
    if env_rows:
        print_simple_table(
            env_rows, headers=["env var", "status"],
            title="Key Env Vars (presence only)",
        )

    source = _config_source_path()
    if source:
        print_info(f"Config source: {source}")
    else:
        print_info("Config source: default (env only, no workspace config file)")


def _resolve_for_call(
        conf: LLMConfig,
        *,
        provider: str,
        model: str,
        tag: str | None,
        no_fallback: bool = False,
) -> ResolvedModel:
    resolved = conf.get_model(
        provider=provider, model=model, tag=tag, no_fallback=no_fallback,
    )
    for name, field in [
        ("api_key", resolved.service.api_key),
        ("base_url", resolved.service.base_url),
    ]:
        if isinstance(field, str) and field.startswith("$"):
            print_error(
                f"env var {field[1:]} is not set (service {resolved.service.name!r} {name}) "
                f"— set it, or pick another provider/model."
            )
            raise typer.Exit(code=1)
    return resolved


def _call(
        resolved: ResolvedModel,
        prompt: str,
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
) -> str:
    from ghoshell_moss.llms.client import build_agent
    agent = build_agent(
        resolved,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return agent.run_sync(prompt).output


def _call_structured(
        resolved: ResolvedModel,
        *,
        prompt: str,
        response_model: str,
        instruction: str | None,
        json_output: bool,
        verbose: bool,
        repeat: int,
) -> None:
    """Structured call via model-func engine (PydanticAIFuncs).

    ``response_model`` is module:attr pointing to a BaseModel subclass.
    ``instruction`` is auto-read from file if it is an existing path.
    """
    import json as _json

    from ghoshell_common.helpers import import_from_path
    from ghoshell_moss.llms.funcs import PydanticAIFuncs

    result_type = import_from_path(response_model)
    inst = _read_instruction(instruction)
    funcs = PydanticAIFuncs()

    async def _run() -> list[dict]:
        results = []
        for _ in range(repeat):
            r = await funcs.call(
                instruction=inst,
                prompt=prompt,
                result_type=result_type,
                model=resolved,
            )
            results.append(r.model_dump(exclude_none=True))
        return results

    items = asyncio.run(_run())
    if repeat == 1 and not json_output:
        _print_single_result(items[0], verbose)
    elif repeat == 1 and json_output:
        echo(_json.dumps(items[0], indent=2, ensure_ascii=False))
    elif json_output:
        echo(_json.dumps(items, indent=2, ensure_ascii=False))
    elif verbose:
        for i, item in enumerate(items):
            print_info(f"[{i + 1}/{repeat}]")
            _print_single_result(item, verbose)
    else:
        for item in items:
            echo(item.get("content", "") or str(item["result"]))


def _print_single_result(item: dict, verbose: bool) -> None:
    """Print a single LLMFuncResult dict — structured then optional verbose."""
    result = item.get("result")
    content = item.get("content")
    if result:
        echo(str(result))
    elif content:
        echo(content)
    if verbose:
        print_simple_table(
            [
                [
                    _fmt_usage(item.get("usage")),
                    f"{item.get('cast', 0):.2f}s",
                    str(item.get("retries", 0)),
                ]
            ],
            headers=["usage", "elapsed", "retries"],
        )


def _fmt_usage(usage: dict | None) -> str:
    if not usage:
        return "-"
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    return f"in={inp} out={out}"


_GHOST_EXTRA_AVAILABLE: bool | None = None


def _ghost_extra_available() -> bool:
    """[ghost] extra 是否安装 — find_spec 轻量检查, 不 import (避免拖进 pydantic-ai + anthropic 全套).

    进程内缓存 (模块私有 flag), 重入无副作用, 只查一次.
    """
    global _GHOST_EXTRA_AVAILABLE
    if _GHOST_EXTRA_AVAILABLE is None:
        _GHOST_EXTRA_AVAILABLE = (
            importlib.util.find_spec("pydantic_ai") is not None
            and importlib.util.find_spec("anthropic") is not None
        )
    return _GHOST_EXTRA_AVAILABLE


# ── call / test — require the `ghost` extra (pydantic-ai). No dependency → hidden. ──
if _ghost_extra_available():

    def _read_instruction(value: str | None) -> str:
        """If value is a file path that exists, read it; otherwise return as-is."""
        if not value:
            return ""
        candidate = Path(value)
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        except OSError:
            pass
        return value

    @llms_app.command(
        name="call",
        short_help="Send a one-shot prompt to a configured model.",
    )
    def call_cmd(
            prompt: str = typer.Argument(..., help="Prompt to send."),
            provider: str = typer.Option("", "--provider", help="Provider/service name."),
            model: str = typer.Option("", "--model", help="Exact model name."),
            tag: str = typer.Option(None, "--tag", help="Model tag (small_fast_model/flash/pro)."),
            temperature: float = typer.Option(None, "--temperature", help="Sampling temperature."),
            max_output_tokens: int = typer.Option(None, "--max-output-tokens", help="Max output tokens."),
            no_fallback: bool = typer.Option(
                False, "--no-fallback",
                help="Strict mode — raise if provider/model not found (no silent fallback to default).",
            ),
            instruction: str = typer.Option(
                None, "--instruction", "-i",
                help="Instruction (system prompt) — string or file path. File is auto-read.",
            ),
            json_output: bool = typer.Option(
                False, "--json", "-j",
                help="Output result as JSON (structured response).",
            ),
            verbose: bool = typer.Option(
                False, "--verbose", "-v",
                help="Show usage, timing, and retries alongside output.",
            ),
            response_model: str = typer.Option(
                None, "--response-model", "-r",
                help="module:attr of a BaseModel for structured output (e.g. mypkg.models:Score).",
            ),
            repeat: int = typer.Option(
                1, "-n",
                help="Number of in-process repetitions.",
            ),
    ) -> None:
        """One-shot LLM call. Prompt + optional instruction and structured output.

        Without ``-r``: plain-text call (built-in). With ``-r``: structured
        call via model-func engine — instruction + prompt -> BaseModel result.
        ``-n`` > 1 repeats in-process. ``-i`` auto-reads a file if the value
        is an existing path.
        """
        conf = _load_config()
        resolved = _resolve_for_call(
            conf.resolve(),
            provider=provider, model=model, tag=tag, no_fallback=no_fallback,
        )
        try:
            if response_model:
                _call_structured(
                    resolved,
                    prompt=prompt,
                    response_model=response_model,
                    instruction=instruction,
                    json_output=json_output,
                    verbose=verbose,
                    repeat=repeat,
                )
            else:
                output = _call(
                    resolved, prompt,
                    temperature=temperature, max_output_tokens=max_output_tokens,
                )
                echo(output)
        except Exception as e:
            print_error(f"call failed: {e}")
            raise typer.Exit(code=1)

    @llms_app.command(
        name="test",
        short_help="Verify end-to-end llms availability with a tiny call.",
    )
    def test_cmd(
            provider: str = typer.Option("", "--provider", help="Provider/service name."),
            model: str = typer.Option("", "--model", help="Exact model name."),
            tag: str = typer.Option(None, "--tag", help="Model tag (small_fast_model/flash/pro)."),
            no_fallback: bool = typer.Option(
                False, "--no-fallback",
                help="Strict mode — raise if provider/model not found (no silent fallback to default).",
            ),
    ) -> None:
        """Integrated availability check — resolve, call, report. api_key never printed."""
        conf = _load_config()
        resolved = _resolve_for_call(
            conf.resolve(),
            provider=provider, model=model, tag=tag, no_fallback=no_fallback,
        )
        try:
            output = _call(resolved, "Reply with exactly: pong")
        except Exception as e:
            print_error(f"llms test FAILED: {e}")
            raise typer.Exit(code=1)
        print_success(
            f"llms OK — {resolved.service.name}/{resolved.model.model} replied: {output!r}"
        )
