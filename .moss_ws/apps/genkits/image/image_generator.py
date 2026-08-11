"""ImageGenerator — 文生图领域接口 + 豆包/千问实现 + IoC Provider。

设计：
  ImageGenerator (ABC)         — 领域接口
  GenerationResult             — 值对象
  BaseImageGenerator           — 抽象基类，提供下载 + 状态管理
  DoubaoImageGenerator         — 豆包 Seedream 实现
  QwenImageGenerator           — 千问实现
  DoubaoImageGeneratorProvider — IoC Provider
  QwenImageGeneratorProvider   — IoC Provider
"""

import asyncio
import os
import tempfile
from abc import ABC, abstractmethod
from enum import Enum
import aiohttp
from pydantic import BaseModel

from ghoshell_container import Provider, IoCContainer
from ghoshell_moss.resources.local_image import (
    LocalImageStorage,
    LocalImageItem,
    LocalImageInfo,
)
from ghoshell_moss.message import unique_id

# ── 值对象 ──────────────────────────────────────────────────────────


class GenerationStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationResult(BaseModel):
    """单次生成的结果值对象。"""

    tag: str
    prompt: str
    status: GenerationStatus
    locator: str | None = None
    error: str | None = None


# ── 领域接口 ────────────────────────────────────────────────────────


class ImageGenerator(ABC):
    """文生图领域接口。"""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        tag: str = "",
        model: str = "",
    ) -> GenerationResult:
        ...

    @abstractmethod
    def get_state(self) -> dict[str, GenerationResult]:
        """返回当前内存中的任务状态映射 (task_id → result)。"""
        ...


# ── 共用父类 ──────────────────────────────────────────────────────


class BaseImageGenerator(ImageGenerator, ABC):
    """下载 + 状态管理，供具体 ImageGenerator 实现复用。"""

    @abstractmethod
    async def generate(self, prompt: str, tag: str = "", model: str = "") -> GenerationResult:
        ...

    def __init__(self, api_key: str, storage: LocalImageStorage):
        self._api_key = api_key
        self._storage = storage
        self._state: dict[str, GenerationResult] = {}

    async def _download_and_store(
        self, url: str, prompt: str, model: str
    ) -> str | None:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                content = await resp.read()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            info = LocalImageInfo(
                host="workspace-assets",
                description=prompt,
                tags=[model] if model else [],
            )
            item = LocalImageItem(info, tmp_path)
            return await self._storage.put(item)
        finally:
            os.unlink(tmp_path)

    def get_state(self) -> dict[str, GenerationResult]:
        return dict(self._state)


# ── 豆包实现 ────────────────────────────────────────────────────────


class DoubaoImageGenerator(BaseImageGenerator):
    """豆包 Seedream 文生图。"""

    ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
    DEFAULT_MODEL = "doubao-seedream-5-0-260128"
    DEFAULT_SIZE = "2048x2048"
    TIMEOUT = 120
    MAX_RETRIES = 2
    RETRY_DELAY = 3

    def __init__(self, api_key: str, storage: LocalImageStorage):
        BaseImageGenerator.__init__(self, api_key, storage)

    async def generate(
        self, prompt: str, tag: str = "", model: str = ""
    ) -> GenerationResult:
        task_id = unique_id()
        self._state[task_id] = GenerationResult(
            tag=tag, prompt=prompt, status=GenerationStatus.PENDING
        )

        model = model or self.DEFAULT_MODEL
        body = {
            "model": model,
            "prompt": prompt,
            "size": self.DEFAULT_SIZE,
            "response_format": "url",
            "output_format": "png",
            "watermark": False,
        }
        headers = self._headers()

        try:
            data = await self._call_api(body, headers)
            self._check_error(data)
            urls = [item["url"] for item in data.get("data", []) if "url" in item]
        except Exception as e:
            result = GenerationResult(
                tag=tag, prompt=prompt,
                status=GenerationStatus.FAILED, error=f"{type(e).__name__}: {e}",
            )
            self._state[task_id] = result
            return result

        if not urls:
            result = GenerationResult(
                tag=tag, prompt=prompt,
                status=GenerationStatus.FAILED, error="no image returned",
            )
            self._state[task_id] = result
            return result

        locators: list[str] = []
        for url in urls:
            locator = await self._download_and_store(url, prompt, model)
            if locator:
                locators.append(locator)

        if not locators:
            result = GenerationResult(
                tag=tag, prompt=prompt,
                status=GenerationStatus.FAILED, error="no valid image URL",
            )
            self._state[task_id] = result
            return result

        result = GenerationResult(
            tag=tag, prompt=prompt, status=GenerationStatus.COMPLETED, locator=locators[0],
        )
        self._state[task_id] = result
        return result

    # -- helpers -------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    async def _call_api(self, body: dict, headers: dict) -> dict:
        last_error = None
        for attempt in range(self.MAX_RETRIES + 1):
            connector = aiohttp.TCPConnector(
                limit=1, force_close=True, enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(
                total=self.TIMEOUT, connect=15, sock_read=self.TIMEOUT,
            )
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout,
            ) as session:
                try:
                    async with session.post(
                        self.ENDPOINT, json=body, headers=headers,
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_error = e
                    if attempt < self.MAX_RETRIES:
                        await asyncio.sleep(self.RETRY_DELAY)
        raise RuntimeError(
            f"Doubao API failed after {self.MAX_RETRIES + 1} attempts: {last_error}"
        )

    @staticmethod
    def _check_error(data: dict) -> None:
        if "error" in data:
            err = data["error"]
            raise RuntimeError(
                f"Volcengine API error: {err.get('code')} - {err.get('message')}"
            )


# ── 千问实现 ────────────────────────────────────────────────────────


class QwenImageGenerator(BaseImageGenerator):
    """千问图像生成。"""

    ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    DEFAULT_MODEL = "qwen-image-2.0-pro"
    DEFAULT_SIZE = "2048x2048"
    TIMEOUT = 60
    MAX_RETRIES = 0
    RETRY_DELAY = 0

    def __init__(self, api_key: str, storage: LocalImageStorage):
        BaseImageGenerator.__init__(self, api_key, storage)

    async def generate(
        self, prompt: str, tag: str = "", model: str = ""
    ) -> GenerationResult:
        task_id = unique_id()
        self._state[task_id] = GenerationResult(
            tag=tag, prompt=prompt, status=GenerationStatus.PENDING
        )

        model = model or self.DEFAULT_MODEL
        body = {
            "model": model,
            "input": {
                "messages": [
                    {"role": "user", "content": [{"text": prompt}]},
                ],
            },
            "parameters": {
                "size": self.DEFAULT_SIZE.replace("x", "*"),
                "n": 1,
                "watermark": False,
            },
        }
        headers = self._headers()

        try:
            data = await self._call_api(body, headers)
            self._check_error(data)
            urls = self._parse_image_urls(data)
        except Exception as e:
            result = GenerationResult(
                tag=tag, prompt=prompt,
                status=GenerationStatus.FAILED, error=f"{type(e).__name__}: {e}",
            )
            self._state[task_id] = result
            return result

        if not urls:
            result = GenerationResult(
                tag=tag, prompt=prompt,
                status=GenerationStatus.FAILED, error="no image returned",
            )
            self._state[task_id] = result
            return result

        locators: list[str] = []
        for url in urls:
            locator = await self._download_and_store(url, prompt, model)
            if locator:
                locators.append(locator)

        if not locators:
            result = GenerationResult(
                tag=tag, prompt=prompt,
                status=GenerationStatus.FAILED, error="no valid image URL",
            )
            self._state[task_id] = result
            return result

        result = GenerationResult(
            tag=tag, prompt=prompt, status=GenerationStatus.COMPLETED, locator=locators[0],
        )
        self._state[task_id] = result
        return result

    # -- helpers -------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    async def _call_api(self, body: dict, headers: dict) -> dict:
        last_error = None
        for attempt in range(self.MAX_RETRIES + 1):
            connector = aiohttp.TCPConnector(
                limit=1, force_close=True, enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(
                total=self.TIMEOUT, connect=15, sock_read=self.TIMEOUT,
            )
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout,
            ) as session:
                try:
                    async with session.post(
                        self.ENDPOINT, json=body, headers=headers,
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_error = e
                    if attempt < self.MAX_RETRIES:
                        await asyncio.sleep(self.RETRY_DELAY)
        raise RuntimeError(
            f"Qwen API failed after {self.MAX_RETRIES + 1} attempts: {last_error}"
        )

    @staticmethod
    def _parse_image_urls(data: dict) -> list[str]:
        urls: list[str] = []
        for choice in data.get("output", {}).get("choices", []):
            for item in choice.get("message", {}).get("content", []):
                url = item.get("image") or item.get("image_url", "")
                if url:
                    urls.append(url)
        return urls

    @staticmethod
    def _check_error(data: dict) -> None:
        if "code" in data:
            raise RuntimeError(
                f"Qwen API error: {data.get('code')} - {data.get('message')}"
            )


# ── IoC Providers ───────────────────────────────────────────────────


class DoubaoImageGeneratorProvider(Provider[ImageGenerator]):
    """IoC Provider：豆包 Seedream。"""

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> ImageGenerator:
        from ghoshell_moss.contracts.workspace import Workspace

        workspace = con.force_fetch(Workspace)
        data_dir = workspace.assets().abspath() / "pil-images"
        storage = LocalImageStorage(data_dir, host="workspace-assets")
        return DoubaoImageGenerator(
            api_key=os.getenv("VOLCENGINE_API_KEY", ""), storage=storage
        )


class QwenImageGeneratorProvider(Provider[ImageGenerator]):
    """IoC Provider：千问图像。"""

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> ImageGenerator:
        from ghoshell_moss.contracts.workspace import Workspace

        workspace = con.force_fetch(Workspace)
        data_dir = workspace.assets().abspath() / "pil-images"
        storage = LocalImageStorage(data_dir, host="workspace-assets")
        return QwenImageGenerator(
            api_key=os.getenv("QWEN_API_KEY", ""), storage=storage
        )
