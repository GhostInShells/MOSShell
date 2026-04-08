from typing import Iterable, Any
from ghoshell_container import Provider
from ghoshell_common.helpers import generate_import_path
from ghoshell_moss.core.codex.discover import scan_package, is_native_to
from dataclasses import dataclass
import inspect

ModuleFile = str
ModulePath = str

MANIFEST_CONTRACTS_PATH = 'MOSS.manifests.contracts'

__all__ = [
    'ModuleFile', 'ModulePath',
    'MANIFEST_CONTRACTS_PATH',
    'ContractInfo',
    'read_contract_info',
    'match_contract_infos',
    'find_contract_infos_from_package',
    'search_contract_infos_from_package',
]


# 管理从环境中发现能力的逻辑.
@dataclass(frozen=True)
class ContractInfo:
    """
    contract info of the provider.
    """
    found: str
    'the python module import path where found the contract provider, pattern foo.bar:attr'

    file: str
    'the python file absolute path where found the contract provider'

    provider: Provider

    @property
    def name(self) -> str:
        """python import path of the contract"""
        return generate_import_path(self.provider.contract())

    @property
    def aliases(self) -> list[str]:
        result = []
        for alias in self.provider.aliases():
            result.append(generate_import_path(alias))
        return result

    @property
    def docstring(self) -> str:
        """docstring  of the contract"""
        return inspect.getdoc(self.provider.contract())

    @property
    def provider_type(self) -> str:
        return generate_import_path(type(self.provider))

    @property
    def description(self) -> str:
        return self.docstring.split('\n')[0]

    @property
    def singleton(self) -> bool:
        return self.provider.singleton()

    @property
    def source(self) -> str:
        return inspect.getsource(self.provider.contract())


def search_contract_infos_from_package(
        package_import_path: str = MANIFEST_CONTRACTS_PATH,
) -> Iterable[ContractInfo]:
    """
    search contract infos from a python package.
    """
    providers = set()
    for found_file, found_path, provider in find_contract_infos_from_package(package_import_path):
        if provider in providers:
            continue
        providers.add(provider)
        contract_info = read_contract_info(module_file=found_file, provider_import_path=found_path, provider=provider)
        if contract_info:
            yield contract_info


def find_contract_infos_from_package(package_import_path: str) -> Iterable[tuple[ModuleFile, ModulePath, Provider]]:
    """
    实现方案：
    1. 递归扫描 package (depth=2 或更多，视你 manifests 目录层级而定)
    2. 只对 module 内“原生定义”的对象进行检测（防止重复扫描 import 进来的对象）
    3. 过滤出所有 isinstance(obj, Provider) 的实例
    """
    # 扫描包下的所有模块
    for manifest in scan_package(package_import_path, max_depth=2):
        if manifest.is_package:
            continue

        # 谓词过滤：
        # a) 必须是该模块内定义的（is_native_to），避免重扫从 core 导入的 Provider
        # b) 必须是 Provider 的实例

        try:
            for name, obj in manifest.iter_members(respect_all=True):
                # 检查是否是原生定义的 Provider 实例
                if is_provider(obj):
                    # 拼接 provider 的完整导入路径，例如 MOSS.manifests.contracts.zenoh:zenoh_provider
                    provider_import_path = f"{manifest.module_path}:{name}"
                    yield manifest.file_path, provider_import_path, obj
        except Exception:
            # 记录日志或跳过损坏的模块，确保 CLI 的鲁棒性
            continue


def is_provider(value: Any) -> bool:
    return isinstance(value, Provider)


def match_contract_infos(contracts: list[ContractInfo], search: str) -> Iterable[ContractInfo]:
    """
    支持模糊匹配。
    1. 先尝试完全匹配 Contract Name (Identity)
    2. 再尝试匹配 Provider 所在模块名
    3. 最后进行简单的关键词包含搜索
    """
    search_lower = search.lower()
    for info in contracts:
        # 匹配契约全称 (如 ghoshell_moss.contracts.logger.Logger)
        if search_lower in info.name.lower():
            yield info
        # 匹配发现路径 (如 MOSS.manifests.contracts.workspace)
        elif search_lower in info.found.lower():
            yield info


def read_contract_info(module_file: str, provider_import_path: str, provider: Provider) -> ContractInfo | None:
    """
    read contract info from an IoC provider.
    """
    contract = provider.contract()
    if not inspect.isclass(contract):
        return None
    return ContractInfo(
        found=provider_import_path,
        file=module_file,
        provider=provider,
    )
