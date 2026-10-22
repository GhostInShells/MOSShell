#!/usr/bin/env python3
"""`ideas` — the command line over the same graph the node draws.

One `graph.json` per group holds the whole picture; every verb here edits it. This
script is the implementation anybody can drive — a shell, another agent, the node's
`run` command — so the node never owns the only way in.

    ideas tree                        # the whole graph, in one read
    ideas add "storage" --from architecture --concept design
    ideas link "storage -> md, storage -> JSON"
    ideas set storage --flag          # flag it: this point has a problem
    ideas diff --since 12             # what moved since r12

Root resolution: `--root` > `MOSS_IDEAS_ROOT` > the node's own `ideas/`.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import click

from store import (
    GraphStore,
    junctions,
    leaves,
    resolve_root,
    shortest_path,
    spine,
)

NODE_DIR = Path(__file__).resolve().parent


def _store(ctx) -> GraphStore:
    store = GraphStore(resolve_root(ctx.obj.get("root", ""), node_dir=NODE_DIR))
    store.reload()
    return store


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise click.ClickException(f"not found: {exc.args[0]}") from None
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None


def _group(store: GraphStore, name: str):
    try:
        return store.require_group(name)
    except KeyError:
        raise click.ClickException(
            f"no group {name!r} — one of: {', '.join(store.groups) or '(none)'}") from None


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--root", default="", help="the ideas directory (default: $MOSS_IDEAS_ROOT or ./ideas)")
@click.pass_context
def cli(ctx, root: str) -> None:
    """ideas — one `graph.json` per group; nodes, edges and the concept scale together.

    A node is `{id, label, body, concept, weight, flag}`; an edge is
    `{id, from, to, kind}` where kind is detail | depend | related. Weight decays with
    generation (child = parent × 0.8, floor 0.2), so depth is size.
    """
    ctx.obj = {"root": root}


# ------------------------------------------------------------------------ reading


@cli.command()
@click.pass_context
def groups(ctx) -> None:
    """List the groups (first-level directories), with node and edge counts."""
    store = _store(ctx)
    visible = [g for g in store.groups.values() if g.status != "archived"]
    if not visible:
        click.echo(f"no group under {store.root}")
        return
    for g in visible:
        rev, name = store.last_save(g)
        tail = f"{g.name} [{g.status}] · {len(g.nodes)} nodes · {len(g.edges)} edges · r{g.rev}"
        if name:
            tail += f' · {store.ops_since_save(g)} ops since "{name}"'
        click.echo(tail)


@cli.command()
@click.option("--group", "-g", default="")
@click.pass_context
def tree(ctx, group: str) -> None:
    """The whole graph as a tree, one line per node — the cold read."""
    store = _store(ctx)
    targets = [_group(store, group)] if group else list(store.groups.values())
    for g in targets:
        if len(targets) > 1:
            click.echo(f"## {g.name}  (r{g.rev})")
        click.echo("\n".join(store.tree_lines(g)))
        if store.broken:
            click.echo("broken: " + "; ".join(store.broken), err=True)


@cli.command("list")
@click.option("--group", "-g", default="")
@click.option("--concept", "-c", default="")
@click.option("--flagged", is_flag=True, help="only the ones marked red")
@click.pass_context
def list_(ctx, group: str, concept: str, flagged: bool) -> None:
    """Flat list: id, label, concept, weight, and whether it carries a flag."""
    store = _store(ctx)
    rows = []
    for g in store.groups.values():
        if group and g.name != group:
            continue
        for n in g._sorted_nodes():
            if concept and n.concept != concept:
                continue
            if flagged and not n.flag:
                continue
            mark = " !" if n.flag else ""
            rows.append(f"{g.name}/{n.id} · {n.label or '(no label)'} · {n.concept} · w{n.weight:g}{mark}")
    click.echo("\n".join(rows) if rows else "nothing matches")


@cli.command()
@click.argument("key")
@click.option("--group", "-g", default="")
@click.pass_context
def show(ctx, key: str, group: str) -> None:
    """One node in full: label, concept, weight, body, and its edges."""
    store = _store(ctx)
    g, node = _guard(store.find, key, group) or (None, None)
    if node is None:
        raise click.ClickException(f"not found: {key}")
    assert g is not None
    click.echo(f"## {node.label or node.id}   [{g.name}/{node.id}]")
    click.echo(f"concept {node.concept} · weight {node.weight:g} · rev r{g.rev}"
               + (" · flagged" if node.flag else ""))
    if node.body:
        click.echo("")
        click.echo(node.body)
    out = g.children(node.id)
    into = g.parents(node.id)
    if into:
        click.echo("")
        click.echo("↑ " + ", ".join(f"{g.nodes[e.src].name}({e.kind})" for e in into))
    if out:
        click.echo("↓ " + ", ".join(f"{g.nodes[e.dst].name}({e.kind})" for e in out))


@cli.command()
@click.option("--group", "-g", default="")
@click.option("--since", "-s", default=None, type=int, help="only ops after this rev")
@click.option("--since-save/--all", "from_save", default=True,
              help="start from the last save checkpoint (default) or from the beginning")
@click.pass_context
def diff(ctx, group: str, since: int | None, from_save: bool) -> None:
    """What changed since a revision — the account, not the truth.

    Default anchor is the last checkpoint: "what moved since I last looked seriously".
    """
    store = _store(ctx)
    targets = [_group(store, group)] if group else list(store.groups.values())
    for g in targets:
        if since is None:
            since = store.last_save(g)[0] if from_save else 0
        rev, name = store.last_save(g)
        anchor = f'since "{name}" (r{rev})' if name else "from the beginning"
        click.echo(f"## {g.name} (r{g.rev}, {anchor} → now)")
        click.echo("\n".join(store.diff_lines(g, since)))


@cli.command()
@click.option("--group", "-g", default="")
@click.option("--top", "-n", default=5, show_default=True)
@click.pass_context
def critical(ctx, group: str, top: int) -> None:
    """Where the spine is: the longest chain, then the branch points."""
    store = _store(ctx)
    targets = [_group(store, group)] if group else list(store.groups.values())
    for g in targets:
        click.echo(f"## {g.name}")
        chain = spine(g)
        if chain:
            click.echo("spine: " + " → ".join(g.nodes[n].name + (" !" if g.nodes[n].flag else "")
                                            for n in chain if n in g.nodes) + f"  ({len(chain) - 1} hops)")
        for nid, fan in junctions(g)[:top]:
            node = g.nodes[nid]
            click.echo(f"branch {node.name} · {fan} downstream" + (" !" if node.flag else ""))


@cli.command()
@click.argument("src")
@click.argument("dst")
@click.option("--group", "-g", default="")
@click.pass_context
def path(ctx, src: str, dst: str, group: str) -> None:
    """Shortest path between two nodes, ignoring edge direction."""
    store = _store(ctx)
    g = _group(store, group)
    hop = shortest_path(g, src, dst)
    if hop is None:
        raise click.ClickException(f"no path between {src!r} and {dst!r}")
    click.echo(" → ".join(g.nodes[n].name for n in hop) + f"  ({len(hop) - 1} hops)")


@cli.command("leaves")
@click.option("--group", "-g", default="")
@click.pass_context
def leaves_(ctx, group: str) -> None:
    """Leaf nodes — the pruning candidates, with how much hangs under each."""
    store = _store(ctx)
    targets = [_group(store, group)] if group else list(store.groups.values())
    for g in targets:
        for nid in leaves(g):
            node = g.nodes[nid]
            click.echo(f"{node.name} · w{node.weight:g}" + (" !" if node.flag else ""))


# ------------------------------------------------------------------------ writing


@cli.command()
@click.argument("label")
@click.option("--group", "-g", default="")
@click.option("--body", "-b", default="")
@click.option("--concept", "-c", default="")
@click.option("--from", "parent", "-f", default="", help="grow it out of this node")
@click.option("--weight", "-w", default=None, type=float)
@click.pass_context
def add(ctx, label: str, group: str, body: str, concept: str, parent: str, weight: float | None) -> None:
    """Add a node; with `--from`, it grows out of that node and inherits its edge."""
    store = _store(ctx)
    g = _group(store, group)
    node = _guard(store.add_node, g, label, body=body, concept=concept, weight=weight, parent=parent)
    click.echo(f"{g.name}/{node.id} · {node.label} · w{node.weight:g}")


@cli.command("set")
@click.argument("key")
@click.option("--group", "-g", default="")
@click.option("--label", default=None)
@click.option("--body", default=None)
@click.option("--concept", default=None)
@click.option("--weight", default=None, type=float)
click.option("--flag/--no-flag", "flag", default=None, help="flag it / clear the flag")
@click.pass_context
def set_(ctx, key: str, group: str, label, body, concept, weight, flag) -> None:
    """Change one node's fields. `--flag` is the human's red mark."""
    store = _store(ctx)
    g, node = store.find(key, group)
    if node is None or g is None:
        raise click.ClickException(f"not found: {key}")
    _guard(store.set_node, g, node.id, label=label, body=body, concept=concept,
           weight=weight, flag=flag)
    fresh = g.find(node.id)
    click.echo(f"{fresh.id} · {fresh.label} · {fresh.concept} · w{fresh.weight:g}"
               + (" !" if fresh.flag else ""))


def _parse_edges(spec: str) -> list[tuple[str, str, str]]:
    """`a -> b, c => d` — several edges in one line, or nobody keeps the graph current."""
    out: list[tuple[str, str, str]] = []
    for chunk in re.split(r"[,\n]+", spec or ""):
        part = chunk.strip()
        if not part:
            continue
        for sep, kind in (("=>", "depend"), ("->", "detail"), ("~", "related")):
            if sep in part:
                src, _, dst = part.partition(sep)
                src, dst = src.strip(), dst.strip()
                if not src or not dst:
                    raise click.BadParameter(f"bad edge: {part!r}")
                out.append((src, dst, kind))
                break
        else:
            raise click.BadParameter(f"bad edge: {part!r} (use `a -> b`, `a => b` or `a ~ b`)")
    return out


@cli.command()
@click.argument("spec")
@click.option("--group", "-g", default="")
@click.pass_context
def link(ctx, spec: str, group: str) -> None:
    """Connect nodes: `a -> b` (detail), `a => b` (depend, a depends on b), `a ~ b` (related)."""
    store = _store(ctx)
    g = _group(store, group)
    for src, dst, kind in _parse_edges(spec):
        edge = _guard(store.link, g, src, dst, kind)
        click.echo(f"{edge.id}  {g.nodes[edge.src].name} --{kind}--> {g.nodes[edge.dst].name}")


@cli.command()
@click.argument("spec")
@click.option("--group", "-g", default="")
@click.pass_context
def unlink(ctx, spec: str, group: str) -> None:
    """Drop edges: `a -> b, c ~ d` (the arrow only separates the two ends)."""
    store = _store(ctx)
    g = _group(store, group)
    for src, dst, _kind in _parse_edges(spec):
        dropped = _guard(store.unlink, g, src, dst)
        click.echo(f"✂ {src} → {dst} ({dropped})")


@cli.command()
@click.argument("key")
@click.option("--group", "-g", default="")
@click.pass_context
def rm(ctx, key: str, group: str) -> None:
    """Drop a node and every edge touching it. The op log keeps the trace."""
    store = _store(ctx)
    g, node = store.find(key, group)
    if node is None or g is None:
        raise click.ClickException(f"not found: {key}")
    gone = _guard(store.remove_node, g, node.id)
    click.echo(f"− {gone['label']} ({gone['node']}), {gone['edges']} edges with it")


@cli.command()
@click.argument("name")
@click.argument("color")
@click.option("--group", "-g", default="")
@click.pass_context
def concept(ctx, name: str, color: str, group: str) -> None:
    """Define or override a concept's colour in this group's scale."""
    store = _store(ctx)
    g = _group(store, group)
    _guard(store.set_concept, g, name, color)
    click.echo(f"{name} = {color}")


@cli.command("op")
@click.argument("payload", required=False)
@click.pass_context
def op_(ctx, payload: str | None) -> None:
    """Apply ops as JSON — one object, an array, or {"ops": [...]}.

    This is the same door the node's `op` command and the page use; every other verb is
    sugar over it. The schema is in `op_help`.

        ideas op '{"op":"set","id":"n10","concept":"risk"}'
        ideas op '[{"op":"add","group":"design","label":"storage"}, {...}]'
    """
    raw = payload if payload is not None else click.get_text_stream("stdin").read()
    ops: list[dict] = []
    try:
        data = json.loads(raw)
    except ValueError:
        # JSONL: one op per line, `#` opens a comment. A body that starts with `{` is read
        # as a literal by CTML, so a streamed body opens with a comment line — not a
        # quirk, just the cheapest way to let the parser through.
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                one = json.loads(line)
            except ValueError as exc:
                raise click.ClickException(f"bad json line {line[:40]!r}: {exc}") from None
            ops.extend(one if isinstance(one, list) else [one])
        data = None
    if data is not None:
        if isinstance(data, dict) and "ops" in data:
            data = data["ops"]
        ops = data if isinstance(data, list) else [data]
    store = _store(ctx)
    for line in store.apply_ops(ops):
        click.echo(line)


@cli.command()
@click.argument("name")
@click.option("--group", "-g", default="")
@click.pass_context
def save(ctx, name: str, group: str) -> None:
    """Name the current rev — a checkpoint you can diff against later.

    A save writes no graph data; it only marks the account and the sign.
    """
    store = _store(ctx)
    g = _group(store, group)
    info = store.save(g, name)
    click.echo(f"★ r{info['rev']} 「{info['name']}」")


@cli.command("op-help")
def op_help() -> None:
    """The op schema, in one place — this is what a skill would document."""
    click.echo(json.dumps({
        "ops": ["add", "set", "link", "unlink", "rm", "concept", "status"],
        "common": {"group": "which group; empty = the first"},
        "add": {"label": "1–8 chars", "body": "", "concept": "word", "weight": "0.2..1",
                "parent": "grow out of this node (id or label)"},
        "set": {"id": "node id or label", "label": "", "body": "", "concept": "",
                "weight": 0.5, "flag": True},
        "link": {"src": "id or label", "dst": "id or label", "kind": "detail|depend|related"},
        "unlink": {"src": "", "dst": ""},
        "rm": {"id": "node id or label"},
        "concept": {"name": "word", "color": "#rrggbb"},
        "status": {"group": "group name", "value": "active|waiting|archived"},
    }, ensure_ascii=False, indent=2))


@cli.command()
@click.argument("lines", nargs=-1)
@click.option("--group", "-g", default="", help="apply to every line that names no group")
@click.pass_context
def batch(ctx, lines, group: str) -> None:
    """Run many verbs at once — one per line, from arguments or stdin.

    The batch door exists because governance is a pass over the whole graph, not a single
    edit: seventeen `set --concept` lines should be one call, not seventeen. Every
    line still goes through the same verbs, so nothing here is a second mechanism.

        ideas batch "set motivation --concept core" "set execution --concept core"
        ideas batch < plan.txt        # one verb per line, `#` comments
    """
    from click.testing import CliRunner
    text = "\n".join(lines) if lines else click.get_text_stream("stdin").read()
    runner = CliRunner()
    done = failed = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        argv = shlex.split(line)
        if group and not any(a in ("-g", "--group") for a in argv):
            argv += ["--group", group]
        result = runner.invoke(cli, ["--root", ctx.obj.get("root", ""), *argv])
        out = (result.output or "").strip()
        if result.exit_code == 0:
            done += 1
            if out:
                click.echo(out)
        else:
            failed += 1
            click.echo(f"! {line}\n  {out or result.exception}", err=True)
    click.echo(f"— {done} ok, {failed} failed", err=bool(failed))


@cli.command()
@click.argument("name")
@click.argument("status", type=click.Choice(["active", "waiting", "archived"]))
@click.pass_context
def status(ctx, name: str, status: str) -> None:
    """Set a group's sign: active | waiting | archived. The graph itself never moves."""
    store = _store(ctx)
    g = _group(store, name)
    _guard(store.set_status, g, status)
    click.echo(f"{g.name} → {status}")


@cli.command()
@click.pass_context
def root(ctx) -> None:
    """Print the root this run governs."""
    click.echo(str(_store(ctx).root))


if __name__ == "__main__":
    cli()
