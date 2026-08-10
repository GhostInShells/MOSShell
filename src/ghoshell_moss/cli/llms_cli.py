"""moss llms — inspect and call LLM configs.

- ``list``: model meta + config file path + key env var presence. Never
  resolves env vars — no secret values on the command line.
- ``call`` / ``test``: one-shot pydantic-ai calls for availability
  verification. Registered only when the ``ghost`` extra (pydantic-ai) is
  installed — commands with no dependency are not shown.
"""

import asyncio
import os
from pathlib import Path

import typer

from ghoshell_moss.contracts.llms import Effort, LLMConfig, LLMFuncs, ModelRef, ResolvedModel
from ghoshell_moss.depends import available

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


def _project_container():
    """Bootstrap 项目并返回容器 — 轻路径, 无 matrix/cell/网络.

    Project.bootstrap() 幂等: 载入 env + container.bootstrap (触发
    ConfigInstanceRegisterBootstrapper 注册 config 实例). 环境坏时也大概率能拉起.
    """
    from ghoshell_moss.core.blueprint.project import Project
    project = Project.discover()
    project.bootstrap()
    return project.container


def _load_config() -> LLMConfig:
    """Read LLMConfig from the project container — else the default. Read-only."""
    try:
        return _project_container().force_fetch(LLMConfig)
    except Exception:
        return LLMConfig()


def _load_funcs() -> LLMFuncs:
    """Read the LLMFuncs engine from the project container (decision A provider)."""
    return _project_container().force_fetch(LLMFuncs)


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
        effort: Effort | None = None,
) -> str:
    from ghoshell_moss.llms.client import build_agent
    agent = build_agent(
        resolved,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        effort=effort,
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
        effort: Effort | None = None,
        export_anchor: str | None = None,
        input_anchor: str | None = None,
        thinking: str | None = None,
) -> None:
    """Structured call via model-func engine (PydanticAIFuncs).

    ``response_model`` is module:attr pointing to a BaseModel subclass.
    ``instruction`` is auto-read from file if it is an existing path.
    ``export_anchor`` — anchor target filename (no .anchor.yml suffix, may
    embed a path); '' = auto-generate a uid-based name. Produced anchor file
    paths are printed after the results.
    ``input_anchor`` — anchor file to consume: its turns are injected as
    message_history (introspection). Read via ``Anchor.from_file`` — the
    data structure self-explains the protocol, the engine sees only the
    Anchor constraint.
    ``thinking`` — manual thinking block (string or file, auto-read),
    injected as a ModelResponse(ThinkingPart) — introspection (内观).
    """
    import json as _json

    from ghoshell_common.helpers import import_from_path
    from ghoshell_moss.anchor import Anchor
    from ghoshell_moss.llms.funcs import PydanticAIFuncs

    result_type = import_from_path(response_model)
    inst = _read_instruction(instruction)
    anchor = Anchor.from_file(input_anchor) if input_anchor else None
    thinking_text = _read_instruction(thinking) if thinking else None
    funcs = PydanticAIFuncs()

    async def _run() -> tuple[list[dict], list[str]]:
        results = []
        anchor_paths: list[str] = []
        for _ in range(repeat):
            r = await funcs.call(
                instruction=inst,
                prompt=prompt,
                result_type=result_type,
                model=resolved,
                effort=effort,
                export_anchor=export_anchor,
                input_anchor=anchor,
                thinking=thinking_text,
            )
            if r.anchor is not None:
                if export_anchor:
                    anchor_paths.append(f"{export_anchor}.anchor.yml")
                else:
                    anchor_paths.append(f"{r.anchor.meta.name}.anchor.yml")
            item = r.model_dump(exclude_none=True)
            item.pop("anchor", None)
            results.append(item)
        return results, anchor_paths

    items, anchor_paths = asyncio.run(_run())
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
    for p in anchor_paths:
        print_success(f"anchor: {p}")


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


# ── call / test — require the `ghost` extra (pydantic-ai). No dependency → hidden. ──
# available() 走 depends 的 find_spec 门, 不 import (避免拖进 pydantic-ai + anthropic 全套).
if available("pydantic_ai", "anthropic"):

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
            effort: Effort = typer.Option(
                None, "--effort",
                help="Thinking effort: none/minimal/low/medium/high/xhigh/max.",
            ),
            export_anchor: str = typer.Option(
                None, "--export-anchor",
                help=(
                    "Anchor target filename (no .anchor.yml suffix, may embed a path). "
                    "'' = auto-generate a uid-based name. Structured calls only."
                ),
            ),
            input_anchor: str = typer.Option(
                None, "--input-anchor",
                help=(
                    "Anchor file to consume — its turns are injected as "
                    "message_history before this call (introspection). "
                    "CallAnchor refs only. Structured calls only."
                ),
            ),
            thinking: str = typer.Option(
                None, "--thinking",
                help=(
                    "Manual thinking block (string or file, auto-read) — injected "
                    "as a ModelResponse(ThinkingPart), introspection (内观). "
                    "A/B vs putting the position in the prompt (外观). Structured calls only."
                ),
            ),
    ) -> None:
        """One-shot LLM call. Prompt + optional instruction and structured output.

        Without ``-r``: plain-text call (built-in). With ``-r``: structured
        call via model-func engine — instruction + prompt -> BaseModel result.
        ``-n`` > 1 repeats in-process. ``-i`` auto-reads a file if the value
        is an existing path. ``--effort`` maps per protocol (anthropic_effort /
        openai_reasoning_effort). ``--export-anchor`` freezes each call as a
        cognitive anchor file — name it for a stable address (re-run overwrites,
        versions live in git), or pass ``''`` for an auto uid-based name.
        ``--input-anchor`` consumes an anchor file: its turn chain becomes the
        message history of this call — the new anchor chains onto the old.
        ``--thinking`` injects a thinking block as the model's own prior
        reasoning (introspection) — compare against the same position fed as
        a user prompt (external) to A/B 内观 vs 外观.
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
                    effort=effort,
                    export_anchor=export_anchor,
                    input_anchor=input_anchor,
                    thinking=thinking,
                )
            else:
                output = _call(
                    resolved, prompt,
                    temperature=temperature, max_output_tokens=max_output_tokens,
                    effort=effort,
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

    @llms_app.command(
        name="count",
        short_help="Count tokens in a string (tiktoken estimate for non-OpenAI protocols).",
    )
    def count_cmd(
            text: str = typer.Argument("", help="Text to count tokens for."),
            file: Path = typer.Option(
                None, "--file", "-f",
                help="Read text from a file instead of the argument.",
            ),
            provider: str = typer.Option(
                "", "--provider", help="Provider/service name (affects tokenizer).",
            ),
            model: str = typer.Option(
                "", "--model", help="Exact model name (affects tokenizer).",
            ),
            tag: str = typer.Option(None, "--tag", help="Model tag."),
            no_fallback: bool = typer.Option(
                False, "--no-fallback",
                help="Strict mode — raise if provider/model not found.",
            ),
            tokens: bool = typer.Option(
                False, "--tokens", "-t",
                help="Also print the tokenized ids.",
            ),
    ) -> None:
        """Count tokens for a string. OpenAI protocols count exactly; others estimate."""
        source = file.read_text(encoding="utf-8") if file is not None else text
        if not source:
            print_error("empty input — provide text or --file")
            raise typer.Exit(code=1)
        conf = _load_config()
        # count 是纯本地计算, 不需要 api_key — 只解析模型选分词器, 不做 env 检查.
        resolved = conf.resolve().get_model(
            provider=provider, model=model, tag=tag, no_fallback=no_fallback,
        )
        funcs = _load_funcs()
        result = funcs.count_tokens(source, model=resolved, include_tokens=tokens)

        parts = [f"{result.count} tokens", f"encoding={result.encoding}"]
        if result.service:
            parts.append(f"service={result.service}")
        if result.model:
            parts.append(f"model={result.model}")
        print_success(" | ".join(parts))
        if result.estimate:
            print_info("estimate — tiktoken is the OpenAI tokenizer (non-OpenAI protocol)")
        if tokens and result.tokens:
            echo(" ".join(str(t) for t in result.tokens))
