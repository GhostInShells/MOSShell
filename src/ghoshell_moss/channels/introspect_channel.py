"""Introspect — 运行时反射,读取正在运行的 MOSS 自身源码 | 集成 | alpha

读取的是 **运行中的活对象**,不是磁盘上的源码文本:import 解析到
``sys.modules[name]``(同一个在内存里活着的模块),反射它的 ``__dict__`` /
``__globals__`` / 实际解析到的 import / 被 monkey-patch 过的属性 / 实时状态。
权威是 runtime 对象,``inspect.getsource`` 只是它的表示 —— 从不做"路径猜测,
从 N 份相同的源码里挑一份"这种静态读数。

``scope`` 是**构建时声明**的边界(工厂参数):允许反射哪些包前缀。
绑定即授权 —— channel 进入 ghost 的树,就授权它看到 scope 内的全部源码。
默认 scope=``ghoshell_moss``(MOSS runtime 自身),宿主 app 在 build 时扩宽。

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.introspect_channel import build_introspect_channel

    main = new_shell_main_channel()
    main.import_channels(build_introspect_channel())  # 默认 scope: ghoshell_moss
    main.import_channels(build_introspect_channel(scope=("ghoshell_moss", "my_app")))
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.channel_builder import (
    MutableChannel,
    ChannelFactory,
    new_channel,
)
from ghoshell_moss.core.concepts.channel import Channel

__all__ = ["new_introspect_channel", "build_introspect_channel"]

_RESULT_CHAR_CAP = 20_000
_SELF_PREFIX_DEFAULT = "ghoshell_moss"

Scope = str | Iterable[str] | Callable[[str], bool] | None
"""scope 声明:包前缀(str 或 iterable),或 `(import_path)->bool` 谓词,或 None(默认 self)."""


def _in_scope(prefix: str, import_path: str) -> bool:
    """判断 import_path 是否落在 prefix 下. 先剥掉 `:attr`,兼容冒号/点两种写法."""
    module = import_path.split(":", 1)[0]
    return module == prefix or module.startswith(prefix + ".")


def _normalize_scope(scope: Scope) -> Callable[[str], bool]:
    if scope is None:
        return lambda p: _in_scope(_SELF_PREFIX_DEFAULT, p)
    if callable(scope):
        return scope
    prefixes = (scope,) if isinstance(scope, str) else tuple(scope)
    return lambda p: any(_in_scope(pref, p) for pref in prefixes)


def _describe_scope(scope: Scope) -> str:
    """instruction 里声明边界用的可读描述."""
    if scope is None:
        return _SELF_PREFIX_DEFAULT
    if callable(scope):
        return "<custom predicate>"
    if isinstance(scope, str):
        return scope
    return ", ".join(scope)


def _resolve_import_path(import_path: str) -> object:
    """按 CLI 惯例解析 import 路径:module.path 或 module.path:attr(点作冒号回退)."""
    from ghoshell_common.helpers import import_from_path
    try:
        return import_from_path(import_path)
    except ImportError:
        if "." in import_path:
            parts = import_path.rsplit(".", 1)
            return import_from_path(f"{parts[0]}:{parts[1]}")
        raise


def _cap(text: str) -> str:
    """封顶保留**头部**(源码/契约在前,依赖块在后,截尾不丢主体)."""
    if len(text) <= _RESULT_CHAR_CAP:
        return text
    dropped = len(text) - _RESULT_CHAR_CAP
    return text[:_RESULT_CHAR_CAP] + f"\n...[{dropped} chars truncated]\n"


def _parse_lines(spec: str) -> tuple[int, int | None] | None:
    """Parse a 1-indexed line spec: 'start-end' / 'start-' / '-end' / 'n'.

    end None = to end of file. Empty spec = whole source. Raises ValueError on bad ints.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    if "-" in spec:
        a, _, b = spec.partition("-")
        start = int(a) if a.strip() else 1
        end = int(b) if b.strip() else None
        return (start, end)
    n = int(spec)
    return (n, n)


def _deny(import_path: str) -> str:
    return (
        f"Refused: '{import_path}' is outside the declared scope.\n"
        "This channel only reflects the runtime's own code (bind-scope). "
        "Pass an in-scope import path, or widen the scope when building the channel."
    )


def _architecture_map() -> str:
    """策展架构地图 —— 从 ghoshell_moss.architecture 的 import 清单反射出分区表."""
    import re
    from ghoshell_moss.core.codex import from_module
    import ghoshell_moss.architecture as arch

    source = inspect.getsource(arch)
    chunks = re.split(r"^# =+\n", source, flags=re.MULTILINE)
    sections: list[tuple[str, str, list[tuple[str, object]]]] = []
    pending_title = ""
    pending_path = ""

    def _manifest(value: object) -> object:
        try:
            return from_module(value)
        except Exception:
            return None

    for chunk in chunks:
        lines = [ln for ln in chunk.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        has_imports = any(ln.startswith("import ") and " as " in ln for ln in lines)
        if not has_imports:
            for line in lines:
                if line.startswith("# ") and not pending_title:
                    pending_title = line[2:].strip()
                elif line.startswith("# ") and pending_title and not pending_path:
                    pending_path = line[2:].strip()
            continue
        title, pkg_path = pending_title, pending_path
        pending_title, pending_path = "", ""
        entries = []
        for line in lines:
            if line.startswith("import ") and " as " in line:
                parts = line[len("import "):].split(" as ")
                if len(parts) == 2:
                    value = getattr(arch, parts[1].strip(), None)
                    if value is not None and inspect.ismodule(value):
                        m = _manifest(value)
                        if m is not None:
                            entries.append((parts[1].strip(), m))
        if entries:
            sections.append((title, pkg_path, entries))
    if sections:
        total = sum(len(e) for _, _, e in sections)
        out = [f"MOSS architecture map — {total} entries in {len(sections)} sections."]
        for title, pkg_path, entries in sections:
            out.append(f"\n## {title}")
            out.append(f"  {pkg_path}")
            for alias, m in entries:
                kind = "pkg" if _is_package_manifest(m) else "mod"
                desc = (getattr(m, "short_doc", "") or "").split("\n")[0].strip()
                out.append(f"| {kind} {alias} | {getattr(m, 'module_path', alias)} | {desc} |")
        return "\n".join(out)
    return source


def _is_package_manifest(m: object) -> bool:
    try:
        return bool(getattr(m, "is_package", False))
    except Exception:
        return False


def new_introspect_channel(
    scope: Scope = None,
    *,
    name: str = "introspect",
    description: str | None = None,
) -> MutableChannel:
    """创建 Introspect 运行时反射 channel — 读取正在运行的 MOSS 自身源码.

    :param scope: 构建时声明的边界. 见 :data:`Scope`. 默认 self(``ghoshell_moss``).
    :param name: CTML 标签名,默认 ``introspect``.
    :param description: 覆盖默认描述.
    """
    allow = _normalize_scope(scope)
    scope_desc = _describe_scope(scope)
    desc = description or (
        "Introspect — read the live MOSS runtime's own Python code. "
        "Reflection resolves to in-memory runtime objects (import -> sys.modules), "
        "not source-file snapshots. Bound to scope at build time."
    )

    chan = new_channel(name=name, description=desc)

    @chan.build.instruction
    def instruction() -> str:
        return (
            "## introspect channel\n"
            "Reflect the LIVE MOSS runtime you are running on — not a snapshot of "
            "source files on disk. `import` resolves to the in-memory module object; "
            "reflection shows its real `__dict__`, resolved imports, live state, and "
            "current channel/host/matrix topology. The runtime reflects ITSELF through "
            "these commands — use them to teach yourself how your body is wired.\n"
            f"Scope boundary (declared at build time): only these package prefixes may "
            f"be reflected — {scope_desc}. Requests outside scope are refused; widen the "
            f"scope when building the channel to authorize more.\n"
            "Use progressively: `get-interface <path>` for the contract, then "
            "`get-source <path>` / `get-interface <path:attr>` to go deeper. "
            "`get-source` returns the total line count — page a large file with "
            "`lines='100-200'` instead of pulling the whole thing. "
            "Reflect the interface first, not the whole tree — avoid flooding context."
        )

    @chan.build.command(name="get-interface", always_observe=True)
    async def get_interface(import_path: str, deps: bool = False) -> str:
        """Reflect the live runtime object's interface (module or module:attr).

        Reads the in-memory object, not a disk snapshot. `deps` adds reflected
        dependency interfaces. Pass a full import path, e.g.
        'ghoshell_moss.core.concepts.channel' or '...:Channel'.
        """
        if not allow(import_path):
            return _deny(import_path)
        from ghoshell_common.helpers import generate_import_path
        from ghoshell_moss.core.codex import reflect_any_by_import_path
        try:
            value = _resolve_import_path(import_path)
        except Exception as e:
            return f"cannot resolve {import_path!r}: {e}"
        canonical = generate_import_path(value)
        try:
            result = reflect_any_by_import_path(canonical, deps=deps)
        except Exception as e:
            return f"reflect {canonical!r} failed: {e}"
        return f"# {canonical}\n" + _cap(result)

    @chan.build.command(name="get-source", always_observe=True)
    async def get_source(import_path: str, lines: str = "") -> str:
        """Read the live object's source, resolved from its loaded provenance.

        `import` yields the in-memory object; `inspect.getsource` reads from the
        object's own runtime origin (`module.__file__`), not a guessed path among
        identical copies. Pass a full import path.

        `lines` selects a 1-indexed range ('100-200' / '100-' / '-50' / '40') to
        page a large file instead of flooding context. The return records the
        file, the range, and the total line count so you can navigate.
        """
        if not allow(import_path):
            return _deny(import_path)
        from ghoshell_common.helpers import generate_import_path
        try:
            value = _resolve_import_path(import_path)
        except Exception as e:
            return f"cannot resolve {import_path!r}: {e}"
        canonical = generate_import_path(value)
        try:
            src = inspect.getsource(value)
        except TypeError:
            return f"no source for {canonical!r} (builtin/C/namespace)"
        except OSError as e:
            return f"cannot read source: {e}"
        try:
            source_file = inspect.getfile(value)
        except TypeError:
            source_file = "<no file>"

        body_lines = src.split("\n")
        if body_lines and body_lines[-1] == "":
            body_lines.pop()
        total = len(body_lines)

        try:
            spec = _parse_lines(lines)
        except ValueError:
            return f"invalid line range {lines!r} — expected 'start-end' like '100-200' or '40'"

        if spec is None:
            range_desc = f"1-{total}"
            body = "\n".join(body_lines)
        else:
            start, end = spec
            start = max(1, start)
            end = total if end is None else min(end, total)
            if start > end:
                return f"invalid line range {lines!r} (file has {total} lines)"
            range_desc = f"{start}-{end}"
            body = "\n".join(body_lines[start - 1:end])

        header = (
            f"# {canonical}\n"
            f"# file: {source_file}\n"
            f"# lines {range_desc} / {total}\n"
        )
        return header + (f"(empty source)\n" if total == 0 else _cap(body))

    @chan.build.command(name="where", always_observe=True)
    async def where(import_path: str) -> str:
        """Resolve an import path to its live definition: canonical path + loaded file.

        Reports the object's runtime origin (`inspect.getfile`), the file it was
        actually loaded from — not a filesystem search. Pass a full import path.
        """
        if not allow(import_path):
            return _deny(import_path)
        from ghoshell_common.helpers import generate_import_path
        try:
            value = _resolve_import_path(import_path)
        except Exception as e:
            return f"cannot resolve {import_path!r}: {e}"
        canonical = generate_import_path(value)
        try:
            source_file = inspect.getfile(value)
        except TypeError:
            source_file = "<no file: builtin/C/namespace>"
        return f"Canonical: {canonical}\nLoaded from: {source_file}"

    @chan.build.command(name="list", always_observe=True)
    async def list_members(module_path: str) -> str:
        """List a module's live members, or a package's submodules (runtime view).

        For a package: its submodules via the live `__path__`. For a module:
        classes/functions/variables defined in that module. Pass an import path.
        """
        if not allow(module_path):
            return _deny(module_path)
        from ghoshell_common.helpers import import_from_path
        try:
            value = import_from_path(module_path)
        except Exception as e:
            return f"cannot import {module_path!r}: {e}"

        # package -> list submodules via live __path__
        if inspect.ismodule(value) and hasattr(value, "__path__"):
            import pkgutil
            submods = sorted(
                info.name
                for info in pkgutil.iter_modules(value.__path__)
                if info.name != "__init__"
            )
            if not submods:
                return f"Package: {module_path}\n(no submodules)"
            return f"Package: {module_path}\n  " + "\n  ".join(submods)

        # module -> live members defined here
        if not inspect.ismodule(value):
            return f"{module_path!r} is not a module"
        modname = getattr(value, "__name__", module_path)
        members = inspect.getmembers(value)
        classes = sorted(n for n, o in members if inspect.isclass(o) and getattr(o, "__module__", None) == modname)
        funcs = sorted(n for n, o in members if inspect.isfunction(o) and getattr(o, "__module__", None) == modname)
        vars_ = sorted(
            n for n, o in members
            if not n.startswith("_")
            and not inspect.isclass(o) and not inspect.isfunction(o) and not inspect.ismodule(o)
            and not callable(o)
        )
        lines = [f"Members of {modname}:"]
        for n in classes:
            lines.append(f"  class {n}")
        for n in funcs:
            lines.append(f"  def {n}")
        for n in vars_:
            lines.append(f"  {n}")
        return "\n".join(lines) if len(lines) > 1 else f"Members of {modname}:\n  (none)"

    @chan.build.command(name="architecture", always_observe=True)
    async def architecture() -> str:
        """Show the curated MOSS architecture map — where each abstraction lives.

        Hand-curated import list (ghoshell_moss.architecture). Use for navigation
        instead of grepping: find the module that holds a concept, then reflect it.
        """
        return _architecture_map()

    return chan


def build_introspect_channel(
    scope: Scope = None,
    *,
    name: str = "introspect",
    description: str | None = None,
) -> ChannelFactory:
    """IoC 集成工厂:从容器解析并返回 introspect channel 的 ChannelFactory.

    introspect 是纯反射 channel,无 IoC 依赖;工厂保留容器参数以符合
    :data:`ChannelFactory` 契约,scope / name / description 由调用方构建时声明.
    """
    def factory(container: IoCContainer) -> Channel:
        return new_introspect_channel(scope=scope, name=name, description=description)
    return factory
