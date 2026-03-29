import os
import warnings
from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path

from typing_extensions import Self

from ghoshell_container import IoCContainer


__all__ = [
    'DEFAULT_WORKSPACE_DIR', 'MOSS_ENV_FILE', 'WORKSPACE_ENV_KEY', 'ENVIRONMENT_IMPORT_PATH_ENV_KEY',

]

# Core constants for MOSS environment discovery
DEFAULT_WORKSPACE_DIR = '.moss'
MOSS_ENV_FILE = ".moss_env"
WORKSPACE_ENV_KEY = 'MOSS_WORKSPACE'
ENVIRONMENT_IMPORT_PATH_ENV_KEY = "MOSS_ENVIRONMENT"

# Type aliases for clarity
FoundWorkspace = Path | None
FoundEnvFile = Path | None


class Environment(ABC):
    """
    Environment discovery capability. Defines an implementation that provides
    all resources based on environment discovery for the MOSS architecture.

    The Environment must manage its own isolation levels (e.g., process-level
    or thread-level). By default, it should act as a process-level singleton.
    """

    @classmethod
    @abstractmethod
    def new(
            cls,
            *,
            found_import_path: str | None,
            found_env_file: Path | None,
            found_workspace: Path | None,
    ) -> Self:
        """
        Instantiate the Environment, passing in context about how it was discovered.
        """
        pass

    @abstractmethod
    def workspace(self) -> Path:
        """
        Returns the absolute path to the current workspace.
        """
        pass

    @abstractmethod
    def discover_env(self) -> dict[str, str]:
        """
        Returns the environment variables required for environment discovery.
        This is useful for passing identical environment context when spawning sub-processes.
        """
        pass

    @abstractmethod
    def get_container(self) -> IoCContainer:
        """
        Provides dependencies from the Environment via the IoC container.
        It should only provide foundational services sharing the same isolation level
        as the Environment (e.g., logging, workspace path, process management).
        The IoC container itself should have the current Environment object registered.
        """
        pass


_environment: Environment | None = None
"""Supports patching to define a global singleton environment."""


def set_environment(env: Environment) -> None:
    """
    Global patch mechanism. Registers an environment instance globally so it can be
    retrieved without an import path.
    """
    global _environment
    _environment = env


def get_environment(import_path: str | None = None) -> Environment:
    f"""
    The globally agreed-upon mechanism for retrieving the Environment instance.
    Any custom instance retrieval mechanism should be built on top of this.

    This ensures that MOSS components and tools in the same environment can locate 
    the Environment instance using identical discovery logic.

    Discovery priority for the Environment class:
    1. Explicit `import_path` provided as an argument.
    2. Import path found in the `{ENVIRONMENT_IMPORT_PATH_ENV_KEY}` env variable.
    3. The `{MOSS_ENV_FILE}` file exists in the current directory containing the import path.
    4. Recursive search upwards for the `{MOSS_ENV_FILE}` file.
    5. The current directory contains a `{DEFAULT_WORKSPACE_DIR}` directory, which contains `{MOSS_ENV_FILE}`.

    :param import_path: The import path for the environment instance, following the 
                        [module_import_path:attribute] syntax.
    :returns: The instantiated Environment object.
    """
    found_workspace: FoundWorkspace = None
    found_env_file: FoundEnvFile = None

    if import_path is None:
        import_path, found_workspace, found_env_file = _find_environment_constants()

    if import_path is None:
        # If no valid Environment class definition is found, attempt to return a default instance.
        # Check if a patched global environment exists first.
        global _environment
        if _environment is not None:
            return _environment
        return default_environment()

    # Clean the import path to prevent whitespace-related import errors
    import_path = import_path.strip()
    parts = import_path.split(':', 1)

    if len(parts) != 2:
        raise ValueError(
            f"Invalid import_path format: '{import_path}'. "
            f"It must strictly follow the 'module_import_path:attribute' syntax."
        )

    module_path, attr_name = parts

    # 1. Attempt to import the module
    try:
        imported = import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{module_path}' specified in MOSS environment path '{import_path}'. "
            f"Underlying error: {e}"
        ) from e

    # 2. Attempt to retrieve the attribute/class
    env_cls = getattr(imported, attr_name, None)
    if env_cls is None:
        raise AttributeError(
            f"Found Environment import_path '{import_path}', but the module '{module_path}' "
            f"does not contain the attribute '{attr_name}'."
        )

    # 3. Validate inheritance
    if not isinstance(env_cls, type) or not issubclass(env_cls, Environment):
        raise TypeError(
            f"The object '{attr_name}' found at '{import_path}' is of type {type(env_cls)}, "
            f"which is not a valid subclass of the Environment ABC."
        )

    # Instantiate and return using the factory method
    return env_cls.new(
        found_import_path=import_path,
        found_env_file=found_env_file,
        found_workspace=found_workspace
    )


def default_environment() -> Environment:
    """
    Provides a fallback, zero-configuration Environment instance if no explicit
    configuration is found.
    """
    raise NotImplementedError("Default environment fallback is not yet implemented.")


def find_defined_workspace(root: Path | None = None) -> Path | None:
    """
    Locates the defined workspace path based on environment variables or directory conventions.
    """
    if WORKSPACE_ENV_KEY in os.environ:
        workspace_str = os.environ[WORKSPACE_ENV_KEY].strip()
        workspace = Path(workspace_str).resolve()
        if workspace.exists() and workspace.is_dir():
            return workspace
        warnings.warn(f"Invalid MOSS workspace provided via environment variable: {workspace_str}")

    # Use resolved path to safely handle symbolic links
    root = root.resolve() if root else Path.cwd().resolve()
    expect_root_workspace = root / DEFAULT_WORKSPACE_DIR

    if expect_root_workspace.exists() and expect_root_workspace.is_dir():
        return expect_root_workspace

    return None


def _find_environment_constants() -> tuple[str | None, FoundWorkspace, FoundEnvFile]:
    """
    Searches for the environment class import path based on predefined conventions.
    Returns a tuple of (import_path, found_workspace_path, found_env_file_path).
    """
    # 1. Check environment variables (Highest priority)
    if ENVIRONMENT_IMPORT_PATH_ENV_KEY in os.environ:
        env_val = os.environ[ENVIRONMENT_IMPORT_PATH_ENV_KEY].strip()
        if env_val:
            return env_val, None, None

    # Resolve cwd to prevent issues if the user is operating within a symlinked directory
    cwd = Path.cwd().resolve()

    # 2. Check the current working directory for the environment file
    expect_cwd_env_file = cwd / MOSS_ENV_FILE
    if expect_cwd_env_file.exists() and expect_cwd_env_file.is_file():
        value = expect_cwd_env_file.read_text(encoding="utf-8").strip()
        if value:
            return value, None, expect_cwd_env_file

    # 3. Traverse upwards to find a valid environment file
    for parent in cwd.parents:
        expect_parent_env_file = parent / MOSS_ENV_FILE
        if expect_parent_env_file.exists() and expect_parent_env_file.is_file():
            value = expect_parent_env_file.read_text(encoding="utf-8").strip()
            if value:
                return value, None, expect_parent_env_file

    # 4. Fallback to checking within a conventionally discovered workspace directory
    # Note: Workspace discovery has lower priority than direct Environment file discovery
    workspace = find_defined_workspace(cwd)
    if workspace:
        expect_workspace_env_file = workspace / MOSS_ENV_FILE
        if expect_workspace_env_file.exists() and expect_workspace_env_file.is_file():
            value = expect_workspace_env_file.read_text(encoding="utf-8").strip()
            if value:
                return value, workspace, expect_workspace_env_file

    # Give up if all discovery methods fail
    return None, None, None
