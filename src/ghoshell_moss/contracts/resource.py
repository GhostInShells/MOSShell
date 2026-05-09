"""
资源管理抽象 (Resource Management Abstraction) — 验证版.

从 contracts/resource.py 复制, 加上讨论的改动:
  - host: 实例级标识, 同一 scheme 下的不同数据集
  - path: host 内的资源路径
  - locator: 计算属性, scheme://host/path 完整句柄
  - Query: 带 session_id 的查询对象
  - QueryResult: 结构化返回, 替代 ClarifyError 在 agent 层的抛异常

验证通过后覆盖回 contracts/resource.py, 全局替换 import 路径.

核心概念:
  scheme://host/path    全局资源句柄, 字符串即可引用, 跨工具传递.
  ResourceMeta          给 AI 阅读的结构化元信息 (Pydantic).
  ResourceItem          meta + 懒加载的实际数据.
  ResourceStorage       单个 (scheme, host) 的资源后端.
  ResourceRegistry      跨 scheme+host 的路由层 (VFS).

特性:
  - 自解释: usage() + help() 让 AI 自服务发现.
  - 查询语义由后端定义: SQL/keyword/regex/自然语言 均可, usage() 声明.
  - 可扩展: 新 scheme = 新 Storage 实现, 不改接口.
  - 可传递: scheme://host/path 作为全局句柄在工具间传递引用.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence, Union, Type
from pydantic import BaseModel, Field
from ghoshell_container import IoCContainer, Bootstrapper

__all__ = [
    "ResourceMeta", "ResourceItem", "ResourceStorage", "ResourceRegistry",
    "Query", "Recollection", 'unpack_query',
    "ClarifyError",
    "RESOURCE_META", "RESOURCE_TYPE", "R",
    "ResourceStorageFactory",
    "ResourceStorageFactoryBootstrapper",
]

RESOURCE_TYPE = TypeVar("RESOURCE_TYPE")


class ResourceMeta(BaseModel, ABC):
    """
    可寻址的资源元信息.
    全局地址: scheme://host/path
    locator 是计算属性, 不序列化到 JSON.
    """

    host: str = Field(
        description="实例标识, 区分同一 scheme 下的不同数据集",
    )
    path: str = Field(
        description="host 内的资源路径",
    )
    description: str = Field(
        description="描述信息 (AI 可读)",
    )

    @classmethod
    @abstractmethod
    def scheme(cls) -> str:
        """
        scheme 名称. scheme://host/path 构成全局资源地址.
        """
        pass

    @classmethod
    @abstractmethod
    def scheme_description(cls) -> str:
        """
        scheme 的介绍, 说明这类资源是什么.
        """
        pass

    @property
    def locator(self) -> str:
        """
        完整资源句柄: scheme://host/path.
        计算属性, 不参与序列化.
        """
        return f"{self.scheme()}://{self.host}/{self.path}"

    def as_content(self) -> str:
        """
        返回给 AI 阅读的 JSON 字符串, 注入 locator 字段.
        """
        data = self.model_dump()
        data["locator"] = self.locator
        import json
        return json.dumps(data, ensure_ascii=False, default=str)


RESOURCE_META = TypeVar("RESOURCE_META", bound=ResourceMeta)


class ResourceItem(Generic[RESOURCE_META, RESOURCE_TYPE], ABC):
    """
    资源项. meta 总是立即可用, get() 懒加载实际数据.

    meta_type() 声明 Item 携带的 Meta 类型.
    scheme() 从 meta_type() 派生, 不需要额外实现.
    """

    @classmethod
    @abstractmethod
    def meta_type(cls) -> type[RESOURCE_META]:
        """
        返回此 Item 所携带的 ResourceMeta 子类.
        """
        pass

    @classmethod
    def scheme(cls) -> str:
        """
        scheme 名称. 从 meta_type() 派生.
        """
        return cls.meta_type().scheme()

    @property
    @abstractmethod
    def meta(self) -> RESOURCE_META:
        """
        返回资源元信息. 始终立即可用, 不触发 I/O.
        """
        pass

    @abstractmethod
    async def get(self) -> RESOURCE_TYPE:
        """
        返回实际数据对象. 可能触发 I/O (文件读取/API 调用等).
        """
        pass


class ClarifyError(Exception):
    """
    需要调用者澄清时抛出.
    question 向上层传递, 上层 (AI 或人类) 补全信息后重试.
    """

    def __init__(self, question: str, *args) -> None:
        self.question = question
        super().__init__(question, *args)


# -- Query / QueryResult: Agent 层的交互模型 --------------------------------

_SessionId = Union[str | None]
_Text = str

Query = tuple[_Text, _SessionId | None] | str


def unpack_query(q: Query) -> tuple[_Text, _SessionId]:
    if isinstance(q, str):
        return q, None
    elif isinstance(q, tuple):
        text, session_id = q
        return text, session_id
    raise TypeError(f"Query {type(q)} not supported")


class Recollection(BaseModel):
    """
    结构化查询结果.
    done=True 时 locators 包含匹配结果 (完整 scheme://host/path 句柄).
    done=False 时 prompt/choices 提供澄清问题, 调用者带 session_id 继续.
    """

    done: bool = Field(
        default=True,
        description="交互是否完成. False 表示需要调用者澄清",
    )
    locators: list[str] = Field(
        default_factory=list,
        description="匹配的完整资源句柄列表 (scheme://host/path)",
    )
    reasoning: str = Field(
        default="",
        description="为什么匹配这些资源",
    )
    prompt: str | None = Field(
        default=None,
        description="当 done=False 时, 向调用者提出的澄清问题",
    )
    choices: list[str] | None = Field(
        default=None,
        description="当 done=False 时, 可选的候选项",
    )
    session_id: str | None = Field(
        default=None,
        description="当前会话 ID, 用于后续 new_query 继续交互",
    )


# -- Storage & Registry ---------------------------------------------------


class ResourceStorage(Generic[RESOURCE_META, RESOURCE_TYPE], ABC):
    """
    单一 (scheme, host) 的资源存储后端.
    注册到 ResourceRegistry 后, AI 通过 locator (scheme://host/path) 访问.

    scheme() 是类级别的, host 是实例级别的.
    get/delete 接受 path (storage 已知自己的 scheme 和 host).
    """

    @abstractmethod
    def scheme(self) -> str:
        """
        资源的 scheme. 通常委托给 ResourceMeta.scheme().
        """
        pass

    @abstractmethod
    def scheme_description(self) -> str:
        """
        scheme 的介绍.
        """
        pass

    @property
    @abstractmethod
    def host(self) -> str:
        """
        实例级标识. 同一 scheme 下区分不同数据集.
        """
        pass

    @abstractmethod
    def usage(self) -> str:
        """
        解释自身的用法.
        关键内容: query 支持什么语法 (SQL/keyword/regex/自然语言),
        限制, 使用示例, 返回的 ResourceMeta 字段含义.
        AI 调用者应该先读 usage() 再使用.
        """
        pass

    @abstractmethod
    async def help(self, question: str | None = None) -> str:
        """
        类似 CLI 的 --help 或 man.
        question=None 时返回概览, 有值时回答具体问题.
        """
        pass

    @abstractmethod
    async def list_metas(
            self,
            query: str | None = None,
            limit: int = -1,
    ) -> Sequence[RESOURCE_META]:
        """
        浏览或搜索资源.
        :param query: 查询条件. None 表示浏览全量.
                      查询语法由各 Storage 自行定义, 通过 usage() 声明.
        :param limit: 返回数量上限. -1 为约定的最大值.
        :return: 匹配的资源元信息列表.
        """
        pass

    async def recall(self, query: Query) -> Recollection:
        """
        使用 ai 或别手段响应偏自然语言的查询逻辑.
        """
        raise NotImplementedError(f"resource storage {type(self)} does not implement recall")

    @abstractmethod
    async def get(
            self, path: str
    ) -> ResourceItem[RESOURCE_META, RESOURCE_TYPE] | None:
        """
        按 path 获取资源项. Storage 已知 scheme 和 host.
        :return: None 表示不存在.
        """
        pass

    @abstractmethod
    async def put(
            self, item: ResourceItem[RESOURCE_META, RESOURCE_TYPE]
    ) -> str:
        """
        保存或更新资源. 若是新资源, Storage 负责分配 path.
        :return: 完整 locator (scheme://host/path).
        """
        pass

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """
        删除指定资源.
        :param path: host 内的资源路径.
        :return: True 表示删除成功, False 表示资源不存在.
        """
        pass


class ResourceStorageFactory(ABC):
    """
    registry 的无副作用自解释配置项.

    可以用于做环境发现自解释.
    """

    @abstractmethod
    def scheme(self) -> str:
        """
        资源的 scheme. 通常委托给 ResourceMeta.scheme().
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        自身的介绍.
        """
        pass

    @property
    @abstractmethod
    def host(self) -> str:
        """
        实例级标识. 同一 scheme 下区分不同数据集.
        """
        pass

    @abstractmethod
    def factory(self, container: IoCContainer) -> ResourceStorage:
        """
        根据环境注册的工厂方法完成 resource storage 的实例化.
        """
        pass


R = TypeVar("R", bound=ResourceItem)


class ResourceRegistry(ABC):
    """
    跨 scheme+host 的资源路由层 (VFS).
    以 (scheme, host) 二元组为主键索引.

    最小实现 = 内存 dict.
    完整实现 = web hub.

    get(locator) 解析 scheme://host/path, 路由到对应 storage.
    get_by_item_type() 通过 item_cls.scheme() 路由, 保留 Python 侧的静态类型.
    """

    @abstractmethod
    def register(self, storage: ResourceStorage) -> None:
        """
        注册一个 ResourceStorage 实现.
        以 (storage.scheme(), storage.host) 为键.
        """
        pass

    @abstractmethod
    def unregister(self, scheme: str, host: str) -> bool:
        """
        移除指定 (scheme, host) 的存储.
        :return: True 表示移除成功, False 表示未注册.
        """
        pass

    @abstractmethod
    def schemes(self) -> Sequence[str]:
        """
        列出所有已注册的 scheme (去重).
        """
        pass

    @abstractmethod
    def hosts(self, scheme: str) -> Sequence[str]:
        """
        列出指定 scheme 下的所有 host.
        """
        pass

    @abstractmethod
    async def get(
            self, locator: str
    ) -> ResourceItem | None:
        """
        按完整 locator 获取资源.
        locator 格式: scheme://host/path
        :return: None 表示 scheme/host 未注册或 path 不存在.
        """
        pass

    @abstractmethod
    async def get_by_item_type(
            self, item_cls: type[R], locator: str
    ) -> R | None:
        """
        按 ResourceItem 子类获取资源. 保留具体类型.
        :return: None 表示未注册或 locator 不存在.
        """
        pass

    @abstractmethod
    async def list_metas(
            self,
            scheme: str,
            host: str | None = None,
            query: str | None = None,
            limit: int = 50,
    ) -> Sequence[ResourceMeta]:
        """
        浏览或搜索资源.
        :param scheme: 目标 scheme.
        :param host: 目标 host. None 表示该 scheme 下所有 host.
        :param query: 查询条件. None 表示浏览全量.
        :param limit: 返回数量上限.
        :return: 匹配的资源元信息列表.
        """
        pass

    @abstractmethod
    async def help(
            self,
            scheme: str,
            host: str | None = None,
            question: str | None = None,
    ) -> str:
        """
        获取指定 (scheme, host) 的帮助信息.
        """
        pass

    @abstractmethod
    async def usage(self, scheme: str, host: str | None = None) -> str:
        """
        获取指定 (scheme, host) 的用法说明.
        """
        pass


class ResourceStorageFactoryBootstrapper(Bootstrapper):
    """
    resource bootstrapper to register resource to registry when container bootstrapping.
    """

    def __init__(self, storage_factory: ResourceStorageFactory) -> None:
        self._storage_factory = storage_factory

    def bootstrap(self, container: IoCContainer) -> None:
        registry = container.force_fetch(ResourceRegistry)
        self_resource_storage = self._storage_factory.factory(container)
        registry.register(self_resource_storage)
