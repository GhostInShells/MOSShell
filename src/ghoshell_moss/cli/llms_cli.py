"""moss llms — inspect and call LLM configs.

- ``list``: model meta + config file path + key env var presence. Never
  resolves env vars — no secret values on the command line.
- ``call`` / ``test``: one-shot pydantic-ai calls for availability
  verification. Registered only when the ``ghost`` extra (pydantic-ai) is
  installed — commands with no dependency are not shown.
"""

import os
from pathlib import Path

import typer

from ghoshell_moss.contracts.llms import LLMConfig, ResolvedModel

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
        m = resolved.model
        rows.append([
            resolved.service.name,
            resolved.client_protocol,
            m.model,
            m.description or "-",
            ",".join(sorted(m.tags)) or "-",
            ",".join(m.content_types) or "*",
            str(m.max_output_tokens),
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


# ── call / test — require the `ghost` extra (pydantic-ai). No dependency → hidden. ──
try:
    from ghoshell_moss.depends import depend_ghost
    depend_ghost()
except ImportError:
    pass
else:

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
    ) -> None:
        """One-shot LLM call for debugging. Resolves config internally — never prints secrets."""
        conf = _load_config()
        resolved = _resolve_for_call(
            conf.resolve(),
            provider=provider, model=model, tag=tag, no_fallback=no_fallback,
        )
        try:
            output = _call(resolved, prompt, temperature=temperature, max_output_tokens=max_output_tokens)
        except Exception as e:
            print_error(f"call failed: {e}")
            raise typer.Exit(code=1)
        echo(output)

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
