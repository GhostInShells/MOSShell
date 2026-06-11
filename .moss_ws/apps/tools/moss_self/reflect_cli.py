#!/usr/bin/env python
"""Build-time reflection: recursively traverse moss CLI Typer command tree.

Run standalone:
    python reflect_cli.py

Outputs a markdown command tree for AI consumption.
"""

from typer.main import get_command
from typer.models import DefaultPlaceholder


def _resolve(v, default=None):
    if isinstance(v, DefaultPlaceholder):
        return default
    return v


def _is_hidden(info) -> bool:
    return _resolve(info.hidden, False)


def _short_help(info) -> str:
    return _resolve(info.short_help, "") or ""


def _get_command_help(typer_app, cmd_name: str) -> str:
    try:
        click_group = get_command(typer_app)
        sub_cmd = click_group.commands.get(cmd_name) if hasattr(click_group, 'commands') else None
        if not sub_cmd:
            return ""
        return (sub_cmd.short_help or sub_cmd.help or "").split("\n")[0].strip()
    except Exception:
        return ""


def _get_command_params(typer_app, cmd_name: str) -> list[str]:
    try:
        click_group = get_command(typer_app)
        sub_cmd = click_group.commands.get(cmd_name) if hasattr(click_group, 'commands') else None
        if not sub_cmd:
            return []
        result = []
        for p in sub_cmd.params:
            opts = "|".join(p.opts) if p.opts else p.name
            parts = [opts]
            if p.type and hasattr(p.type, 'name'):
                parts.append(f"<{p.type.name.lower()}>")
            if p.help:
                parts.append(f"— {p.help}")
            if p.default is not None and p.default is not False:
                parts.append(f"(default: {p.default})")
            if p.required:
                parts.append("[required]")
            result.append(" ".join(parts))
        return result
    except Exception:
        return []


def reflect(app, depth: int = 3) -> str:
    lines = [
        "## moss command tree (reflected)",
        "",
        "You can execute any of these commands via the `exec` command.",
        "Example: <apps.tools_moss_self:exec>codex get-interface ghoshell_moss.channels.typer_channel</apps.tools_moss_self:exec>",
        "",
    ]

    groups = [g for g in app.registered_groups if not _is_hidden(g)]
    root_cmds = [c for c in app.registered_commands if not _is_hidden(c)]

    for grp in groups:
        sh = _short_help(grp) or "—"
        lines.append(f"### {grp.name} — {sh}")
        if depth >= 2:
            sub = grp.typer_instance
            sub_cmds = [c for c in sub.registered_commands if not _is_hidden(c)]
            max_name = max((len(c.name) for c in sub_cmds), default=0)
            for cmd in sub_cmds:
                help_line = _get_command_help(sub, cmd.name)
                lines.append(f"  {cmd.name.ljust(max_name + 2)}{help_line}")
            if depth >= 3:
                for cmd in sub_cmds:
                    params = _get_command_params(sub, cmd.name)
                    if params:
                        lines.append(f"  {cmd.name} parameters:")
                        for p in params:
                            lines.append(f"    {p}")
        lines.append("")

    if root_cmds:
        lines.append("### (root commands)")
        for cmd in root_cmds:
            help_line = _get_command_help(app, cmd.name)
            lines.append(f"  {cmd.name}  {help_line}")
            if depth >= 3:
                params = _get_command_params(app, cmd.name)
                if params:
                    lines.append(f"  {cmd.name} parameters:")
                    for p in params:
                        lines.append(f"    {p}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    from ghoshell_moss.cli.main import app
    print(reflect(app, depth=3))
