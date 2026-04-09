from typing import Any, Iterable
from ghoshell_moss.core.codex.discover import scan_package
from ghoshell_moss.core.concepts.topic import TopicModel, TopicSchema
from ghoshell_moss.moss.abcd.manifests import TopicInfo

__all__ = ['find_topic_infos_from_package', 'MANIFEST_TOPICS_PATH', 'TopicInfo', 'search_topic_infos_from_package']

MANIFEST_TOPICS_PATH = 'MOSS.manifests.topics'

TopicName = str
ModuleFile = str
ModulePath = str


def find_topic_infos_from_package(
        package_import_path: str,
) -> Iterable[tuple[ModuleFile, ModulePath, type[TopicModel] | TopicSchema]]:
    """
    扫描逻辑：寻找原生定义的 TopicModel 子类。
    """
    # 限制递归深度为 2
    for manifest in scan_package(package_import_path, max_depth=2):
        if manifest.is_package:
            continue

        # 我们寻找类，且必须是本模块定义的
        for name, obj in manifest.iter_members(predicate=is_topic_info_object):
            model_path = f"{manifest.module_path}:{name}"
            yield manifest.file_path, model_path, obj


def search_topic_infos_from_package(
        package_import_path: str = MANIFEST_TOPICS_PATH,
) -> dict[TopicName, TopicInfo]:
    """
    将扫描到的类转化为 TopicInfo 对象，并以 topic_name 为 key 聚合
    """
    topics: dict[TopicName, TopicInfo] = {}

    for file, path, model in find_topic_infos_from_package(package_import_path):
        # 转化为 Info 结构
        info = TopicInfo.from_topic_type(
            found=path.split(':')[0],  # 模块路径
            file=file,
            model=model
        )

        # 如果有重复的 topic_name，这里可以做日志记录或者简单的覆盖
        topics[info.name] = info

    return topics


def is_topic_info_object(name: str, obj: Any) -> bool:
    """
    detect some value is topic info type
    """
    if isinstance(obj, type):
        return issubclass(obj, TopicModel)
    return isinstance(obj, TopicSchema)


def match_topic_infos(topic_infos: dict[TopicName, TopicInfo], search: str) -> Iterable[TopicInfo]:
    """
    匹配逻辑：搜索 TopicName 或 TopicType
    """
    search_lower = search.lower()
    for info in topic_infos.values():
        if search_lower in info.name.lower() or search_lower in info.type.lower():
            yield info
