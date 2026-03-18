from abc import ABC, abstractmethod
from typing import TypedDict, Optional, Any, Callable, Coroutine, Type, TypeVar, Generic
from pydantic import BaseModel, Field, ValidationError
from ghoshell_common.helpers import uuid
from typing import get_args, get_origin
from enum import Enum

"""
# 说明

当一个项目运行时, 它需要一系列的控制函数可以操作它.
传统的项目会把所有控制函数都严格定义出来. 这样做的缺点是缺乏拓展性.

因此有一种约定优先于实现的弱类型的做法: 

1. 定义了一个 API 类, 用 Pydantic 来代替复杂的 Protocol 做协议化. 
2. 定义了一个 API Manager 抽象, 可以返回所有可访问的 API.
3. 每个 API 都有自己的 入参/出参 的JSON Schema
4. 用强类型数据调用 API Manager, 返回一个强类型的 Result.
5. 底层用 JSONRPC 协议做弱类型通讯. 

我们假设一个 Ghost 运行的时候, 仍然可以对外提供它当前的所有可操作 API, 这些 API 是自解释的, 动态可变的.
这些 API 是提供给 UI 界面和开发者的, 不是提供给 AI 自身的.
UI 界面可以根据 API 的约定, 提前实现界面元素.

"""

__all__ = [
    'APIManager',
    'API',
    'APISchema',

    'JRRequest',
    'JRError',
    'JRFailure',
    'JRRequestError',
    'JRErrorCode',

    'JSONRpcFunc',
]


class JRRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str = Field(
        description="method of the request",
    )
    params: Any = Field(
        description="params of the request",
    )
    id: int | str = Field(
        default_factory=uuid,
        description="unique id of the request",
    )

    def success(self, result: Any) -> "JRSuccess":
        return JRSuccess(
            id=self.id,
            result=result,
        )

    def fail(self, code: int, message: str) -> "JRFailure":
        return JRFailure(
            id=self.id,
            error=JRError(
                code=code,
                message=message,
            )
        )

    @staticmethod
    def invalid(code: int, message: str) -> "JRRequestError":
        return JRRequestError(
            error=JRError(
                code=code,
                message=message,
            )
        )


class JRError(BaseModel):
    code: int = Field(
        description="error code",
    )
    message: str = Field(
        description="error message",
    )
    data: Optional[Any] = Field(
        default=None,
        description="data of the error",
    )


class JRRequestError(BaseModel):
    id: None = None
    error: JRError


class JRSuccess(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int = Field(
        description="request id"
    )
    result: Optional[Any] = Field(
        description="result of the request",
    )


class JRErrorCode(int, Enum):
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMETER = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR = -32000


class JRFailure(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int = Field(
        description="request id"
    )
    error: JRError


JSONRpcFunc = Callable[[JRRequest], Coroutine[Any, Any, JRSuccess | JRFailure | JRRequestError]]

API_PARAMS = TypeVar("API_PARAMS", bound=BaseModel)
API_RESULT = TypeVar("API_RESULT", bound=BaseModel)


class APISchema(TypedDict):
    method: str
    params_schema: dict
    result_schema: dict


class API(Generic[API_PARAMS, API_RESULT], ABC):

    @classmethod
    @abstractmethod
    def method(cls) -> str:
        pass

    @classmethod
    async def call(
            cls,
            func: JSONRpcFunc,
            params: API_PARAMS,
    ) -> tuple[API_RESULT | None, JRRequestError | JRError | None]:
        req = JRRequest(
            method=cls.method(),
            params=params.model_dump(exclude_none=True),
        )
        result, err = await func(req)
        if err is not None:
            return None, err

        try:
            api_result = cls.result_type()(**result.result)
        except ValidationError as e:
            api_result = None
            err = JRError(
                code=JRErrorCode.SERVER_ERROR.value,
                message=str(e),
            )
        return api_result, err

    @classmethod
    def schema(cls) -> APISchema:
        return APISchema(
            method=cls.method(),
            params_schema=cls.params_type().model_json_schema(),
            result_schema=cls.result_type().model_json_schema(),
        )

    @classmethod
    def params_type(cls) -> Type[API_PARAMS]:
        if "__orig_bases__" in cls.__dict__:
            orig_bases = getattr(cls, "__orig_bases__")
            for parent in orig_bases:
                if get_origin(parent) is not API:
                    continue
                args = get_args(parent)
                if not args or not len(args) == 2:
                    break
                return args[0]
        raise AttributeError("can not get params type")

    @classmethod
    def result_type(cls) -> Type[API_RESULT]:
        if "__orig_bases__" in cls.__dict__:
            orig_bases = getattr(cls, "__orig_bases__")
            for parent in orig_bases:
                if get_origin(parent) is not API:
                    continue
                args = get_args(parent)
                if not args or not len(args) == 2:
                    break
                return args[1]
        raise AttributeError("can not get params type")


_METHOD = str


class APIManager(ABC):
    """
    一种自解释的 API 封装实现.


    """

    @abstractmethod
    def apis(self) -> dict[_METHOD, API]:
        """
        返回所有的 API.
        """
        pass

    @abstractmethod
    def exists(self, method: str) -> bool:
        """
        判断 method 是否存在.
        """
        pass

    @abstractmethod
    async def call(self, req: JRRequest) -> tuple[JRSuccess | None, JRError | JRRequestError | None]:
        """
        Golang 风格的 call 处理.

        result, err = call(request)
        if err is not None:
            ...
        else:
            ...
        """
        pass
