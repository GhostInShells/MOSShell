from typing import Any, Iterable
from typing_extensions import Self
from dataclasses import dataclass
from ghoshell_common.helpers import generate_import_path, import_from_path
from ghoshell_moss.core.codex.discover import scan_package, is_class
from ghoshell_moss.core.concepts.topic import TopicModel, TopicSchema
import inspect

MANIFEST_TOPICS_PATH = 'MOSS.manifests.topics'

TopicName = str
ModuleFile = str
ModulePath = str


@dataclass
class TopicInfo:
    """
    Topic info.
    """
    found: str  # 发现 topic 的 module name, 如 MOSS.manifests.topics
    file: str  # 发现 topic 的 module filename
    model: str  # topic 如果是通过 TopicModel 定义的, 此处是它的 import path.
    schema: TopicSchema  # topic schema.

    @classmethod
    def from_topic_type(
            cls,
            found: str,
            file: str,
            model: type[TopicModel] | TopicSchema,
            topic_name: str | None = None,
    ) -> Self:
        if isinstance(model, type) and issubclass(model, TopicModel):
            model_path = generate_import_path(model)
            schema = model.topic_schema(topic_name or None)
        elif isinstance(model, TopicSchema):
            model_path = ''
            schema = model
        else:
            raise TypeError(f"'{type(model)}' is not a topic model")

        return TopicInfo(found=found, file=file, schema=schema, model=model_path)

    @property
    def model_source(self) -> str:
        """source of topic model"""
        if self.model:
            model_type = import_from_path(self.model)
            return inspect.getsource(model_type)
        return ''

    @property
    def description(self) -> str:
        """topic description"""
        return self.schema.description

    @property
    def name(self) -> str:
        """topic name"""
        return self.schema.topic_name

    @property
    def type(self) -> str:
        """topic type"""
        return self.schema.topic_type

    @property
    def json_schema(self) -> dict[str, Any]:
        """topic JSON Schema"""
        return self.schema.json_schema


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
