"""
=============================================================================
[MOSS Codex: Runtime Module Reflection & Discovery]

Development Goal:
    1. Provide a lightweight, lazy-evaluated module discovery mechanism.
    2. Decouple module scanning from actual code execution (using spec finders).
    3. Expose dynamic iterators with closure-based predicate filtering for
       AI capability discovery, abandoning static hardcoded type enumerations.

Author / AI Persona:
    Gemini (Acting as your Architectural Co-pilot / Human Engineer Assistant)
=============================================================================
"""

import inspect
import importlib
import importlib.util
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Tuple, TypeVar

__all__ = [
    'ModuleManifest', 'MemberPredicate',
    'CodexReflectionError',
    'scan_module', 'scan_package',
    'is_subclass_of', 'is_class', 'is_routine', 'is_native_to',
]

# Type alias for member filtering closures
# Takes (member_name, member_object) and returns bool
MemberPredicate = Callable[[str, Any], bool]

T = TypeVar('T')


class CodexReflectionError(Exception):
    """Base exception for Codex runtime reflection failures."""
    pass


@dataclass
class ModuleManifest:
    """
    A lightweight reference to a module in the runtime environment.
    It holds only the path strings. Actual module loading and member
    inspection are done lazily via methods.
    """
    module_path: str
    file_path: Optional[str] = None
    is_package: bool = False

    @property
    def module(self) -> Any:
        """
        Lazily loads and returns the actual Python ModuleType object.
        Code execution (import) ONLY happens when this property is accessed.
        """
        try:
            return importlib.import_module(self.module_path)
        except Exception as e:
            raise CodexReflectionError(f"Failed to dynamically load module '{self.module_path}': {str(e)}")

    @property
    def docstring(self) -> str:
        """Dynamically fetches the module docstring."""
        try:
            return inspect.getdoc(self.module) or ''
        except CodexReflectionError:
            return ''

    @property
    def short_doc(self) -> str:
        return self.docstring.split('\n')[0]

    def iter_members(
            self,
            predicate: Optional[MemberPredicate] = None,
            respect_all: bool = True,
    ) -> Iterator[Tuple[str, Any]]:
        """
        Lazily yields members of the module.

        Args:
            predicate: A closure `lambda name, obj: bool` to filter members dynamically.
            respect_all: If True, restricts yielding to __all__ if defined,
                         or ignores names starting with '_' by default.
        """
        mod = self.module
        public_names = getattr(mod, '__all__', None) if respect_all else None

        for name, obj in inspect.getmembers(mod):
            # Visibility filtering
            if respect_all:
                if public_names is not None and name not in public_names:
                    continue
                if public_names is None and name.startswith('_'):
                    continue

            # Dynamic capability filtering via closure
            if predicate is None or predicate(name, obj):
                yield name, obj


# ============================================================================
# Discovery & Scanning APIs (codex.pkg)
# ============================================================================

def scan_module(module_path: str) -> ModuleManifest:
    """
    Scans a single module path and returns its Manifest WITHOUT executing its code.
    This relies on importlib spec finding, making it incredibly fast and safe.
    """
    try:
        # find_spec locates the module without loading it into sys.modules
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            raise CodexReflectionError(f"Module spec not found for: '{module_path}'")

        # If submodule_search_locations is not None, it's a package
        is_package = spec.submodule_search_locations is not None

        return ModuleManifest(
            module_path=module_path,
            file_path=spec.origin,
            is_package=is_package
        )
    except Exception as e:
        raise CodexReflectionError(f"Error scanning module '{module_path}': {e}")


def scan_package(package_path: str, max_depth: int = 1) -> Iterator[ModuleManifest]:
    """
    Recursively scans a package and yields ModuleManifests up to max_depth.

    Depth 0: Yields only the root package.
    Depth 1: Yields the root package + direct submodules/subpackages.
    """
    try:
        root_manifest = scan_module(package_path)
        yield root_manifest
    except CodexReflectionError:
        return  # Skip if root cannot be scanned

    if not root_manifest.is_package or max_depth <= 0:
        return

    try:
        spec = importlib.util.find_spec(package_path)
        if spec and spec.submodule_search_locations:
            # Iterate through the physical directories of the package
            for module_info in pkgutil.iter_modules(spec.submodule_search_locations):
                submodule_path = f"{package_path}.{module_info.name}"

                if module_info.ispkg:
                    # Recursive yield from sub-packages
                    yield from scan_package(submodule_path, max_depth=max_depth - 1)
                else:
                    # Yield single module
                    try:
                        yield scan_module(submodule_path)
                    except CodexReflectionError:
                        continue
    except CodexReflectionError:
        # Silently ignore unreadable package directories during deep scans
        pass


# ============================================================================
# Common Capability Predicates (codex.ref)
# ============================================================================
# Instead of Enum types, we provide high-order functions for AI to use dynamically.

def is_class() -> MemberPredicate:
    """Predicate: Matches any class."""
    return lambda name, obj: inspect.isclass(obj)


def is_routine() -> MemberPredicate:
    """Predicate: Matches functions, methods, and builtins."""
    return lambda name, obj: inspect.isroutine(obj)


def is_subclass_of(base_class: type, exclude_base: bool = True) -> MemberPredicate:
    """
    Predicate: Matches classes that inherit from a specific base class.
    Extremely useful for finding AI Actions or Plugins in the runtime.
    """

    def _predicate(name: str, obj: Any) -> bool:
        if not inspect.isclass(obj):
            return False
        if exclude_base and obj is base_class:
            return False
        return issubclass(obj, base_class)

    return _predicate


def is_native_to(module_path: str) -> MemberPredicate:
    """
    Predicate: Matches objects actually defined in the module,
    filtering out things imported from elsewhere.
    """

    def _predicate(name: str, obj: Any) -> bool:
        return getattr(obj, '__module__', None) == module_path

    return _predicate
