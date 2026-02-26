import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from typing import Optional

from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer

from ghoshell_moss.core import Channel, ChannelRuntime, PyChannel


class ModuleChannel(Channel):
    """Wrap an importable Python module as a :class:`~ghoshell_moss.core.Channel`.

    The goal of ``ModuleChannel`` is to turn a Python module into a runnable MOSS
    channel automatically.

    The module can be provided either as:

    - an importable dotted path, e.g. ``"a.b.c"``
    - a concrete python script path, e.g. ``"a/b/c.py"``

    It supports two modes:

    1) **Explicit channel mode** (preferred)
       If a channel file is provided or discoverable, ``ModuleChannel`` loads a
       pre-defined channel instance from that file and boots it.

       - If ``channel_file`` is provided, it is used.
       - Otherwise, ``ModuleChannel`` tries to locate a sibling ``__channel__.py``
         in the same directory as the target module (or the package directory).

       The channel file must define an attribute named by ``expect_channel_val``
       (default: ``__channel__``) and that attribute must be an instance of
       :class:`~ghoshell_moss.core.Channel`.

    2) **Auto-wrapping mode** (fallback)
       When no explicit channel file is found, ``ModuleChannel`` imports
       ``module_name`` and automatically registers exported callables as commands
       on a new :class:`~ghoshell_moss.core.PyChannel`.

    Export rules in auto-wrapping mode:

    - If ``include`` is provided, only names in that list are considered.
    - Else, if the module defines ``__all__`` (list/tuple of strings), only those
      names are considered.
    - Else, iterate over ``vars(module)`` and register public routines
      (``inspect.isroutine``) that are defined in the module itself
      (``obj.__module__ == module.__name__``).
    - Commands are registered using the *attribute name* as the command name, so
      re-exported aliases are supported.

    Notes / limitations:

    - Some stdlib / C-extension callables have positional-only parameters
      (e.g. ``math.sqrt(x, /)``). MOSS commands are invoked using keyword
      arguments, so positional-only callables would fail with
      "takes no keyword arguments". ``ModuleChannel`` detects positional-only
      signatures and generates a thin Python wrapper so they can be called with
      kwargs.
    - Variadic signatures (``*args`` / ``**kwargs``) are not rewritten.
      If such callables cannot be reflected reliably, they may be skipped.
    """

    def __init__(
        self,
        name: str,
        description: str,
        module_name: str,
        include: list[str] | None = None,
        expect_channel_val: str = "__channel__",
        channel_file: str | None = None,
        reload_on_bootstrap: bool = False,
    ):
        """Create a module-backed channel.

        Args:
            name: The channel name exposed to MOSS.
            description: Human/LLM-facing channel description.
            module_name: Import path of the target module (e.g. ``"a.b.c"``),
                or a python script path (e.g. ``"a/b/c.py"``).
            include: Optional allow-list of exported names to register as commands
                in auto-wrapping mode. When provided, ``__all__`` is ignored.
            expect_channel_val: The attribute name in channel files that stores
                a prebuilt :class:`~ghoshell_moss.core.Channel` instance.
                Defaults to ``"__channel__"``.
            channel_file: Optional path to a specific ``.py`` file that
                defines the channel value (``expect_channel_val``).
                When provided, it takes precedence over sibling ``__channel__.py``
                auto-discovery.
            reload_on_bootstrap: When ``module_name`` is a python script path
                (e.g. ``"a/b/c.py"``), this flag controls whether the script is
                reloaded on every :meth:`bootstrap`.

                - ``False`` (default): reuse the same imported module instance.
                  Module-level state persists across bootstraps.
                - ``True``: force re-import/reload on every bootstrap.
                  Module-level state resets, but import-time side effects will
                  run repeatedly.
        """
        self._name = name
        self._description = description
        self._uid = uuid()
        self._module_name = module_name
        self._include = include
        self._expect_channel_val = expect_channel_val
        self._channel_py_file = channel_file
        self._reload_on_bootstrap = reload_on_bootstrap

    def name(self) -> str:
        return self._name

    def id(self) -> str:
        return self._uid

    def description(self) -> str:
        return self._description

    def bootstrap(self, container: Optional[IoCContainer] = None) -> ChannelRuntime:
        channel_file = self._resolve_channel_file()
        if channel_file is not None:
            channel_val = self._load_channel_from_file(channel_file)
            return channel_val.bootstrap(container)

        module = self._import_target_module()

        channel = PyChannel(name=self._name, description=self._description)

        exported_names = self._get_exported_names(module)
        if exported_names is not None:
            for name in exported_names:
                if name.startswith("_"):
                    continue
                if not hasattr(module, name):
                    continue
                self._try_register_attr(channel, module, name, getattr(module, name))
        else:
            for attr_name, attr_value in vars(module).items():
                if attr_name.startswith("_"):
                    continue
                # By default only export routines defined in this module.
                if not self._is_own_routine(module, attr_value):
                    continue
                self._try_register_attr(channel, module, attr_name, attr_value)

        return channel.bootstrap(container)

    def _resolve_channel_file(self) -> Path | None:
        if self._channel_py_file is not None:
            path = Path(self._channel_py_file).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.exists() or not path.is_file() or path.suffix != ".py":
                raise FileNotFoundError(f"channel_file not found or invalid: {path}")
            return path

        module_file = self._resolve_module_file()
        if module_file is not None:
            directory = module_file.parent
        else:
            spec = importlib.util.find_spec(self._module_name)
            if spec is None:
                raise ModuleNotFoundError(self._module_name)

            # For packages, prefer the package directory.
            if spec.submodule_search_locations:
                directory = Path(list(spec.submodule_search_locations)[0])
            else:
                if spec.origin is None:
                    return None
                directory = Path(spec.origin).parent

        channel_file = directory / "__channel__.py"
        if channel_file.exists() and channel_file.is_file():
            return channel_file
        return None

    def _load_channel_from_file(self, path: Path) -> Channel:
        module = self._import_module_from_path(path, reload=self._reload_on_bootstrap)

        channel_val = getattr(module, self._expect_channel_val, None)
        if not isinstance(channel_val, Channel):
            raise TypeError(f"channel file {path} must define `{self._expect_channel_val}` as a Channel instance")
        return channel_val

    def _import_target_module(self) -> types.ModuleType:
        module_file = self._resolve_module_file()
        if module_file is not None:
            return self._import_module_from_path(module_file, reload=self._reload_on_bootstrap)
        return importlib.import_module(self._module_name)

    def _resolve_module_file(self) -> Path | None:
        """Return module file path if `module_name` is a `.py` path."""

        candidate = Path(self._module_name).expanduser()
        if candidate.suffix != ".py":
            return None
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"module file not found: {candidate}")
        return candidate

    @staticmethod
    def _import_module_from_path(path: Path, *, reload: bool = False) -> types.ModuleType:
        module_name, sys_path_entry = ModuleChannel._derive_module_name_and_sys_path(path)
        with ModuleChannel._temp_sys_path(sys_path_entry):
            if ModuleChannel._is_dotted_package_module_name(module_name):
                module = importlib.import_module(module_name)
                if reload:
                    return importlib.reload(module)
                return module

            stable_name = ModuleChannel._stable_file_module_name(path)
            if not reload and stable_name in sys.modules:
                return sys.modules[stable_name]

            unique_name = stable_name if not reload else f"{stable_name}_{uuid()}"
            spec = importlib.util.spec_from_file_location(unique_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"failed to load python file: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[unique_name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                if reload:
                    sys.modules.pop(unique_name, None)

            if not reload:
                sys.modules[stable_name] = module
            return module

    @staticmethod
    def _derive_module_name_and_sys_path(path: Path) -> tuple[str, Path]:
        """Best-effort derive (module_name, sys.path entry) for a python file."""

        package_parts: list[str] = []
        directory = path.parent
        while (directory / "__init__.py").exists():
            package_parts.insert(0, directory.name)
            directory = directory.parent

        sys_path_entry = directory if package_parts else path.parent

        if package_parts:
            if path.name == "__init__.py":
                return ".".join(package_parts), sys_path_entry
            return ".".join([*package_parts, path.stem]), sys_path_entry

        # For non-package scripts, use a private module name prefix.
        return "_ghoshell_moss_file", sys_path_entry

    @staticmethod
    def _is_dotted_package_module_name(name: str) -> bool:
        # Names with the private prefix are intentionally non-importable.
        # Anything else is treated as a real module name derived from a package.
        return bool(name) and not name.startswith("_ghoshell_moss_")

    @staticmethod
    def _stable_file_module_name(path: Path) -> str:
        resolved = str(path.expanduser().resolve())
        digest = hashlib.sha1(resolved.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
        return f"_ghoshell_moss_file_{digest}"

    @staticmethod
    @contextlib.contextmanager
    def _temp_sys_path(entry: Path):
        entry_str = str(entry)
        inserted = False
        if entry_str not in sys.path:
            sys.path.insert(0, entry_str)
            inserted = True
        try:
            yield
        finally:
            if inserted:
                try:
                    sys.path.remove(entry_str)
                except ValueError:
                    pass

    def _get_exported_names(self, module) -> list[str] | None:
        """Return explicit export list if provided, else None."""

        if self._include is not None:
            return list(self._include)

        module_all = getattr(module, "__all__", None)
        if isinstance(module_all, (list, tuple)) and all(isinstance(x, str) for x in module_all):
            return list(module_all)
        return None

    @staticmethod
    def _is_own_routine(module, value: object) -> bool:
        if not inspect.isroutine(value):
            return False
        value_module = getattr(value, "__module__", None)
        return value_module == module.__name__

    @staticmethod
    def _try_register_attr(channel: PyChannel, module, attr_name: str, attr_value: object) -> None:
        if not inspect.isroutine(attr_value):
            return

        func_to_register = attr_value
        try:
            sig = inspect.signature(attr_value)
            if any(p.kind == inspect.Parameter.POSITIONAL_ONLY for p in sig.parameters.values()):
                wrapped = ModuleChannel._wrap_positional_only(attr_value, attr_name, sig)
                if wrapped is not None:
                    func_to_register = wrapped
        except (TypeError, ValueError):
            # Some callables (builtins / C-extensions) may not provide a signature.
            pass

        try:
            # Use attribute name as the command name to support aliases.
            channel.build.command(name=attr_name)(func_to_register)
        except (TypeError, ValueError):
            # Some callables (builtins / C-extensions) may not have a valid signature.
            return

    @staticmethod
    def _wrap_positional_only(
        func: object,
        name: str,
        sig: inspect.Signature,
    ) -> types.FunctionType | None:
        """Wrap callables with positional-only params.

        PyCommand executes functions by calling them with keyword arguments only.
        For C-implemented builtins (e.g. stdlib `math.sqrt(x, /)`), calling with
        kwargs will raise `TypeError: ... takes no keyword arguments`.

        This wrapper relaxes positional-only markers by generating a Python
        function that accepts keyword arguments and forwards them to the origin
        callable positionally.
        """

        if not callable(func):
            return None

        # If the origin has *args/**kwargs, we can't make it compatible with the
        # current PyCommand calling convention without losing semantics.
        for p in sig.parameters.values():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                return None

        params: list[str] = []
        call_args: list[str] = []
        call_kwargs: list[str] = []
        inserted_star = False
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.KEYWORD_ONLY and not inserted_star:
                params.append("*")
                inserted_star = True

            default_str = ""
            if p.default is not inspect.Parameter.empty:
                try:
                    default_str = "=" + repr(p.default)
                except Exception:
                    return None

            params.append(f"{p.name}{default_str}")

            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                call_args.append(p.name)
            elif p.kind == inspect.Parameter.KEYWORD_ONLY:
                call_kwargs.append(f"{p.name}={p.name}")

        wrapper_name = f"_moss_wrapped_{name}"
        call_parts = []
        if call_args:
            call_parts.append(", ".join(call_args))
        if call_kwargs:
            call_parts.append(", ".join(call_kwargs))
        call_expr = ", ".join(call_parts)
        src = f"def {wrapper_name}({', '.join(params)}):\n    return _origin({call_expr})\n"

        scope: dict[str, object] = {"_origin": func}
        exec(src, scope)  # noqa: S102
        wrapped = scope[wrapper_name]
        # Make it look nicer in metas/interfaces.
        wrapped.__name__ = name
        return wrapped
