from abc import ABC, abstractmethod
from typing import Optional, Protocol, Union
from pathlib import Path
import os


class Storage(Protocol):

    @abstractmethod
    def abspath(self) -> Path:
        """
        abspath of this storage
        """
        pass

    @abstractmethod
    def sub_storage(self, relative_path: str | Path) -> "Storage":
        """
        :param relative_path: 必须是当前目录的子目录.
        :return:
        """
        pass

    @abstractmethod
    def get(self, file_path: str | Path) -> bytes:
        """
        获取一个 Storage 路径下一个文件的内容.
        :param file_path: storage 下的一个相对路径.
        """
        pass

    @abstractmethod
    def remove(self, file_path: str | Path) -> None:
        """
        删除一个当前目录管理下的文件.
        """
        pass

    @abstractmethod
    def exists(self, file_path: str | Path) -> bool:
        """
        if the object exists
        :param file_path: file_path or directory path
        """
        pass

    @abstractmethod
    def put(self, file_path: str | Path, content: bytes) -> None:
        """
        保存一个文件的内容到 file_path .
        :param file_path: storage 下的一个相对路径.
        :param content: 文件的内容.
        """
        pass


class Workspace(ABC):
    """
    simple workspace manager.
    """

    @abstractmethod
    def root(self) -> Storage:
        """
        workspace 根 storage.
        """
        pass

    def root_path(self) -> Path:
        return self.root().abspath()

    @abstractmethod
    def cwd(self) -> Path:
        """
        system current working directory.
        """
        pass

    def configs(self) -> Storage:
        """
        配置文件存储路径.
        """
        return self.root().sub_storage("configs")

    def runtime(self) -> Storage:
        """
        运行时数据存储路径.
        """
        return self.root().sub_storage("runtime")

    def assets(self) -> Storage:
        """
        数据资产存储路径.
        """
        return self.root().sub_storage("assets")

    def source(self) -> Storage:
        """
        源码位置, 默认应该加入 python path.
        """
        return self.root().sub_storage("src")


class LocalStorage:
    """
    local storage by gemini 3.
    """

    def __init__(self, root_path: Union[str, Path]):
        # 转换为绝对路径以确保校验准确
        self._root = Path(root_path).resolve()
        # 确保根目录存在
        self._root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, relative_path: Union[str, Path]) -> Path:
        """
        核心校验函数：拼接路径并检查是否越界。
        """
        # 拼接并获取真实物理路径（处理 .. 等符号）
        full_path = (self._root / relative_path).resolve()

        # 校验：如果生成的路径不是以 root 开头，说明发生了路径泄漏（如 ../../etc/passwd）
        if not str(full_path).startswith(str(self._root)):
            raise PermissionError(f"Path escape detected: {relative_path} is outside of {self._root}")

        return full_path

    def abspath(self) -> Path:
        return self._root

    def sub_storage(self, relative_path: Union[str, Path]) -> "LocalStorage":
        safe_sub_path = self._safe_path(relative_path)
        return LocalStorage(safe_sub_path)

    def get(self, file_path: Union[str, Path]) -> bytes:
        target = self._safe_path(file_path)
        return target.read_bytes()

    def put(self, file_path: Union[str, Path], content: bytes) -> None:
        target = self._safe_path(file_path)
        # 自动创建中间目录
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def remove(self, file_path: Union[str, Path]) -> None:
        target = self._safe_path(file_path)
        if target.is_file():
            target.unlink()
        elif target.is_dir():
            import shutil
            shutil.rmtree(target)

    def exists(self, file_path: Union[str, Path]) -> bool:
        # 这里同样需要 safe_path，防止通过 exists 探测外部文件
        try:
            target = self._safe_path(file_path)
            return target.exists()
        except PermissionError:
            return False


class LocalWorkspace(Workspace):

    def __init__(self, root_path: Union[str, Path], cwd: Optional[Path] = None):
        storage = LocalStorage(root_path)
        self._root = storage
        cwd = cwd or Path(os.getcwd()).resolve()
        self._cwd = cwd

    def root(self) -> Storage:
        return self._root

    def cwd(self) -> Path:
        return self._cwd
