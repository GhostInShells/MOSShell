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

import concurrent.futures
import inspect
import importlib
import importlib.util
import pkgutil
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Iterator, Optional, Tuple, TypeVar

__all__ = [
    'ModuleManifest', 'MemberPredicate',
    'CodexReflectionError', 'ScanError',
    'scan_module', 'from_module', 'scan_package',
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
class ScanError:
    """Captures a non-fatal error encountered during package scanning.

    When ``strict=False`` (the default), scanners collect these rather than
    raising, so callers can inspect what went wrong after the scan completes.
    """
    module_path: str
    exception: Exception
    stage: str  # "scan" | "import" | "iterate"


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
    timeout: Optional[float] = None

    @property
    def module_name(self) -> str:
        return self.module_path.split('.')[-1]

    @property
    def module(self) -> Any:
        """
        Lazily loads and returns the actual Python ModuleType object.
        Code execution (import) ONLY happens when this property is accessed.
        """
        if self.timeout is not None:
            return _import_with_timeout(self.module_path, self.timeout)
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

def _import_with_timeout(module_path: str, timeout: float) -> Any:
    """Import a module in a thread, raising CodexReflectionError on timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(importlib.import_module, module_path)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise CodexReflectionError(
                f"Import of '{module_path}' timed out after {timeout}s"
            )
        except Exception as e:
            raise CodexReflectionError(
                f"Failed to dynamically load module '{module_path}': {str(e)}"
            )


def scan_module(module_path: str, *, timeout: float | None = None) -> ModuleManifest:
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
            is_package=is_package,
            timeout=timeout,
        )
    except Exception as e:
        raise CodexReflectionError(f"Error scanning module '{module_path}': {e}")


def from_module(module: ModuleType) -> ModuleManifest:
    """Build a ModuleManifest from an already-imported module.

    Unlike scan_module() which uses importlib spec-finding (no execution),
    this wraps a live module object — useful when the module is already
    in sys.modules and you want the structured manifest without re-scanning.
    """
    file_path: str | None = getattr(module, '__file__', None)
    is_package = hasattr(module, '__path__')
    return ModuleManifest(
        module_path=module.__name__,
        file_path=file_path,
        is_package=is_package,
    )


def scan_package(
        package_path: str,
        max_depth: int = 1,
        *,
        parse: Callable[[ModuleManifest], bool] | None = None,
        strict: bool = False,
        errors: list[ScanError] | None = None,
        timeout: float | None = None,
) -> Iterator[ModuleManifest]:
    """
    Recursively scans a package and yields ModuleManifests up to max_depth.

    Depth 0: Yields only the root package.
    Depth 1: Yields the root package + direct submodules/subpackages.

    Args:
        strict: When True, exceptions propagate instead of being silently skipped.
        errors: When provided, suppressed errors are appended to this list
                (even when strict=False).
        timeout: Maximum seconds for any single module import.
    """
    if parse is None:
        parse = lambda x: True

    def _collect(module_path: str, exc: Exception, stage: str) -> None:
        if errors is not None:
            errors.append(ScanError(module_path=module_path, exception=exc, stage=stage))

    try:
        root_manifest = scan_module(package_path, timeout=timeout)
        if parse(root_manifest):
            yield root_manifest
    except CodexReflectionError as e:
        _collect(package_path, e, "scan")
        if strict:
            raise
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
                    yield from scan_package(
                        submodule_path, max_depth=max_depth - 1,
                        parse=parse, strict=strict, errors=errors, timeout=timeout,
                    )
                else:
                    # Yield single module
                    try:
                        got = scan_module(submodule_path, timeout=timeout)
                        if parse(got):
                            yield got
                    except CodexReflectionError as e:
                        _collect(submodule_path, e, "scan")
                        if strict:
                            raise
                        continue
    except CodexReflectionError as e:
        _collect(package_path, e, "scan")
        if strict:
            raise


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
