#!/usr/bin/env python3
"""`ideas` — the command line over the same files the node draws.

Every verb is an edit to `*.idea.md`; nothing else is stored. This script is the
implementation that anybody can drive — a shell, another agent, the node's `run`
command — so the node never owns the only way in.

    ideas --root <dir> list
    ideas spark "双工开得太满，两个人不在同一个时间点上说话" --group voice
    ideas state duplex-gap active
    ideas link duplex-gap attention-cue --kind depend
    ideas rename duplex-gap duplex-gap-between-clicks

Root resolution: `--root` > `MOSS_IDEAS_ROOT` > the node's own `ideas/`.
"""

from __future__ import annotations

from pathlib import Path

import click

from store import STATES, Store, resolve_root, slugify

NODE_DIR = Path(__file__).resolve().parent


def _store(ctx) -> Store:
    store = Store(resolve_root(ctx.obj.get("root", ""), node_dir=NODE_DIR))
    store.reload()
    return store


def _guard(fn, *args, **kwargs) -> str:
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise click.ClickException(f"not found: {exc.args[0]}") from None
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--root", default="", help="the ideas directory (default: $MOSS_IDEAS_ROOT or ./ideas)")
@click.pass_context
def cli(ctx, root: str) -> None:
    """ideas — plain files, one idea per `*.idea.md`, grouped by directory.

    A group is a first-level subdirectory of the root; its README.md is the group's
    own context, and each `*.idea.md` is one idea: frontmatter carries state /
    priority / tags / links, the body carries the thinking.

    States: spark → active → waiting → done / retired. Kinds: related | depend.
    """
    ctx.obj = {"root": root}


@cli.command()
@click.pass_context
def groups(ctx) -> None:
    """List the groups (first-level directories of the root)."""
    store = _store(ctx)
    if not store.groups:
        click.echo(f"no group under {store.root}")
        return
    for g in store.groups.values():
        click.echo(f"{g.name} · {len(g.ideas)} idea(s){' · README' if g.readme else ''}")


@cli.command("list")
@click.option("--group", "-g", default="", help="only this group")
@click.option("--state", "-s", default="", help="only this state")
@click.option("--tag", "-t", default="", help="only ideas carrying this tag")
@click.option("--limit", "-n", default=40, show_default=True)
@click.pass_context
def list_(ctx, group: str, state: str, tag: str, limit: int) -> None:
    """Read the ideas as they stand, newest file change first."""
    store = _store(ctx)
    rows = store.list_lines(group=group, state=state, tag=tag, limit=limit)
    click.echo("\n".join(rows) if rows else "nothing matches")
    if store.broken:
        click.echo("— broken files (not skipped, shown):", err=True)
        for line in store.broken:
            click.echo(f"  {line}", err=True)


@cli.command()
@click.argument("name")
@click.pass_context
def show(ctx, name: str) -> None:
    """Print one idea's file — frontmatter and body, exactly as stored."""
    store = _store(ctx)
    click.echo(_guard(store.read_doc, name), nl=False)


@cli.command()
@click.argument("text")
@click.option("--group", "-g", default="", help="which group; default = the first one")
@click.option("--name", "-n", default="", help="filename stem; default = slug of the text")
@click.option("--priority", "-p", default=1.0, show_default=True)
@click.pass_context
def spark(ctx, text: str, group: str, name: str, priority: float) -> None:
    """Capture one idea. Write the sentence; nothing else is asked."""
    store = _store(ctx)
    idea = _guard(store.spark, text, group=group, name=name, priority=priority)
    click.echo(f"{idea.group}/{idea.name}")


@cli.command()
@click.argument("name")
@click.argument("text")
@click.pass_context
def note(ctx, name: str, text: str) -> None:
    """Append a paragraph to an idea's body."""
    store = _store(ctx)
    _guard(store.note, name, text)
    click.echo(f"{name} +note")


@cli.command()
@click.argument("name")
@click.argument("state", type=click.Choice(list(STATES)))
@click.pass_context
def state(ctx, name: str, state: str) -> None:
    """Move an idea's state."""
    store = _store(ctx)
    _guard(store.set_state, name, state)
    click.echo(f"{name} → {state}")


@cli.command()
@click.argument("name")
@click.argument("priority", type=float)
@click.pass_context
def priority(ctx, name: str, priority: float) -> None:
    """Change how much an idea weighs (it is the bubble's size)."""
    store = _store(ctx)
    _guard(store.set_priority, name, priority)
    click.echo(f"{name} p{priority:g}")


@cli.command()
@click.argument("name")
@click.argument("word")
@click.pass_context
def tag(ctx, name: str, word: str) -> None:
    """Tag an idea."""
    store = _store(ctx)
    _guard(store.tag, name, word)
    click.echo(f"{name} [{word}]")


@cli.command()
@click.argument("name")
@click.argument("word")
@click.pass_context
def untag(ctx, name: str, word: str) -> None:
    """Remove a tag."""
    store = _store(ctx)
    _guard(store.untag, name, word)
    click.echo(f"{name} -[{word}]")


@cli.command()
@click.argument("src")
@click.argument("dst")
@click.option("--kind", "-k", default="related", type=click.Choice(["related", "depend"]))
@click.pass_context
def link(ctx, src: str, dst: str, kind: str) -> None:
    """Connect two ideas (a --depend--> b means a depends on b)."""
    store = _store(ctx)
    _guard(store.link, src, dst, kind)
    click.echo(f"{src} --{kind}--> {dst}")


@cli.command()
@click.argument("src")
@click.argument("dst")
@click.pass_context
def unlink(ctx, src: str, dst: str) -> None:
    """Drop a connection."""
    store = _store(ctx)
    _guard(store.unlink, src, dst)
    click.echo(f"{src} -/-> {dst}")


@cli.command()
@click.argument("old")
@click.argument("new")
@click.pass_context
def rename(ctx, old: str, new: str) -> None:
    """Rename an idea's file and rewrite every link that pointed at it.

    The filename is the identity, so this is the one verb that must sweep the root.
    """
    store = _store(ctx)
    click.echo(_guard(store.rename, old, new))


@cli.command()
@click.pass_context
def root(ctx) -> None:
    """Print the root this run governs."""
    store = _store(ctx)
    click.echo(str(store.root))


@cli.command()
@click.argument("text")
@click.pass_context
def slug(ctx, text: str) -> None:
    """Preview the filename a text would get (no write)."""
    click.echo(slugify(text))


if __name__ == "__main__":
    cli()
