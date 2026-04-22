from ghoshell_moss.host.abcd.host_interface import MossMode
from ghoshell_moss.core.codex.discover import scan_package
from ghoshell_moss.host.abcd.environment import MODE_STUB_PACKAGE
from importlib import import_module
from pathlib import Path
from .manifests import PackageManifest
import inspect
import shutil

__all__ = [
    'ROOT_MODES_PACKAGE',
    'MODE_PACKAGE',
    "DEFAULT_MODE_FILENAME",
    'find_mode_from_package',
    'list_modes_from_root_package',
    'new_mode',
]

ROOT_MODES_PACKAGE = 'MOSS.modes'
MODE_PACKAGE = 'MOSS.modes.{name}'
DEFAULT_MODE_FILENAME = "MODE.md"


def new_mode(
        name: str,
        apps: list[str],
        bring_up_apps: list[str],
        description: str = "",
        target_root_package: str = ROOT_MODES_PACKAGE,
        stub_package: str = MODE_STUB_PACKAGE,
) -> None:
    # 1. 确定目标路径
    root_module = import_module(target_root_package)
    target_root_dir = Path(root_module.__file__).parent.resolve()
    target_mode_dir = target_root_dir / name

    if target_mode_dir.exists():
        raise NameError(f"Mode directory {name} already exists")

    # 2. 确定 Stub 来源路径
    stub_module = import_module(stub_package)
    stub_dir = Path(stub_module.__file__).parent.resolve()

    # 3. 复制 Stub 目录下的一切 (CLAUDE.md, .instructions, 等)
    # 忽略 __pycache__ 和 __init__.py (如果需要自动生成新的)
    shutil.copytree(
        stub_dir,
        target_mode_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "__init__.py")
    )

    # 4. 覆盖/生成核心的 MODE.md
    # 我们基于 Stub 里的模板（如果有）或者直接写入新的
    mode_file = target_mode_dir / DEFAULT_MODE_FILENAME

    # 构造新的模式实例
    mode = MossMode(
        name=name,
        description=description,
        instruction='',
        apps=apps,
        bringup=bring_up_apps,
        file=str(mode_file),
    )

    # 写入 Markdown
    mode_file.write_text(mode.to_markdown())

    # 5. 自动补全 __init__.py 使其成为可导入的包
    (target_mode_dir / "__init__.py").touch()


def list_modes_from_root_package(package_import_path: str = ROOT_MODES_PACKAGE) -> list[MossMode]:
    """
    通过复用 scan_package 逻辑发现所有模式。
    """
    modes = []
    # 我们只关心根包下的一级子包 (max_depth=1)
    # scan_package 第一个产出通常是 ROOT 本身，我们需要跳过它或过滤掉
    for module_manifest in scan_package(package_import_path, max_depth=1):
        # 排除掉根包本身，只处理子包（即具体的 Mode 包）
        if module_manifest.module_path == package_import_path:
            continue

        # 只要是子包，就尝试解析为 Mode
        mode = find_mode_from_package(module_manifest.module_path)
        if mode:
            modes.append(mode)
    return modes


def _ensure_manifest_to_mode(package_path: str, mode: MossMode) -> MossMode:
    """
    如果 Mode 还没有关联 Manifest，尝试为其绑定一个 PackageManifest。
    """
    if mode.__manifest__ is None:
        # 使用当前发现该 Mode 的包路径来初始化资源扫描
        if mode.import_path:
            package_path = mode.import_path
        mode.with_manifest(PackageManifest(package_path))
    return mode


def find_mode_from_package(package_import_path: str) -> MossMode | None:
    try:
        module = import_module(package_import_path)
    except ImportError:
        return None

    mode: MossMode | None = None

    # 1. 尝试从 module 属性中直接获取实例
    for attr in ("mode", "__mode__"):
        instance = getattr(module, attr, None)
        if isinstance(instance, MossMode):
            mode = instance
            break

    # 2. 如果没有实例，尝试从 MODE.md 发现
    if mode is None and hasattr(module, "__file__") and module.__file__:
        mode_dir = Path(module.__file__).parent.resolve()
        expect_file = mode_dir.joinpath(DEFAULT_MODE_FILENAME)
        if expect_file.exists() and expect_file.is_file():
            mode = MossMode.from_markdown(expect_file)

    # 3. 如果还是没有，根据约定自动生成（Convention over Configuration）
    if mode is None:
        description = inspect.getdoc(module) or f"Auto-generated mode for {package_import_path}"
        docstring = ''
        mode = MossMode(
            name=package_import_path.split(".")[-1],
            instruction=docstring,
            description=description,
            import_path=package_import_path,
        )
        if hasattr(module, "__file__") and module.__file__:
            mode.file = str(Path(module.__file__).parent.resolve())

    # 最后确保 Manifest 被挂载
    return _ensure_manifest_to_mode(package_import_path, mode)
