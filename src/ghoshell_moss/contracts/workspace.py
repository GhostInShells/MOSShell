from abc import ABC, abstractmethod
from typing import Optional, Protocol, Union
from pathlib import Path
import os
import time
import re

__all__ = ["Workspace", "Storage", "LocalStorage", "Lock", "LocalWorkspace", "FileLocker"]


class Lock(Protocol):
    """
    Workspace 环境进程锁接口。
    help with gemini 3
    """

    @abstractmethod
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        尝试获取锁。
        :param timeout:
            - None: 阻塞直到成功 (Blocking)
            - 0: 立即返回，拿不到就 False (Non-blocking / Fast-fail)
            - >0: 最多等待指定的秒数
        :return: 是否成功获取锁
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """释放锁。如果锁不是由当前对象持有，应视情况抛出异常或静默处理。"""
        pass

    @abstractmethod
    def is_locked(self, /, by_self: bool = False) -> bool:
        """
        检查锁当前是否被占用。
        注意：即使返回 False，也不保证接下来的 acquire 一定成功（存在竞争）。
        但如果返回 True 且 PID 存活，则说明资源确实被占用。
        """
        pass

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Could not acquire lock")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


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
        :param relative_path: 必须是当前目录的子目录.不存在会自动创建.
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

    @abstractmethod
    def lock(self, key: str) -> Lock:
        """
        创建一个进程锁.
        :param key: pattern r'^[a-zA-Z0-9_-]+$'
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


class LocalStorage:
    """
    local storage by gemini 3.
    """

    def __init__(self, root_path: Union[str, Path]):
        # 转换为绝对路径以确保校验准确
        self._root = Path(root_path).resolve().absolute()
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


class FileLocker(Lock):
    """
    基于文件系统的进程锁实现。
    by gemini 3
    """

    def __init__(self, lock_path: Path):
        self.path = lock_path
        self._has_lock = False

    @staticmethod
    def _is_pid_running(pid: int) -> bool:
        """检查进程是否仍在运行"""
        if pid <= 0:
            return False
        try:
            # 信号 0 不会发送信号，但会执行错误检查
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _read_pid(self) -> Optional[int]:
        try:
            # 使用二进制读取并 strip，避免编码或换行符问题
            if not self.path.exists():
                return None
            content = self.path.read_text().strip()
            return int(content) if content else None
        except (FileNotFoundError, ValueError, OSError, PermissionError):
            # 批量跑单测时，PermissionError 很常见
            return None

    def is_locked(self, /, by_self: bool = False) -> bool:
        """检查锁是否被存活的进程持有"""
        pid = self._read_pid()
        if pid is None:
            return False
        if not self._is_pid_running(pid):
            return False
        return not by_self or pid == os.getpid()

    def acquire(self, timeout: Optional[float] = 0) -> bool:
        # --- 新增：防止重入死锁 ---
        if self._has_lock and self.is_locked(by_self=True):
            return True
        # -----------------------

        start_time = time.time()
        while True:
            try:
                # O_SYNC 确保同步写入
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    with os.fdopen(fd, 'w') as f:
                        f.write(str(os.getpid()))
                        f.flush()
                        os.fsync(f.fileno())  # 强制刷到硬盘
                    self._has_lock = True
                    return True
                except Exception:
                    if os.path.exists(self.path):
                        os.unlink(self.path)
                    raise
            except FileExistsError:
                # 检查是否是僵尸锁
                pid = self._read_pid()

                # 如果读取不到 PID（可能正在写入中），我们视其为被占用
                if pid is not None and not self._is_pid_running(pid):
                    try:
                        os.unlink(self.path)
                        continue  # 清理成功，立即重试创建
                    except FileNotFoundError:
                        continue

                        # 检查超时
                if timeout == 0:
                    return False
                if timeout is not None and (time.time() - start_time) >= timeout:
                    return False

                time.sleep(0.05)  # 稍微缩短重试间隔
        return False

    def release(self) -> None:
        try:
            # 只有确实是自己拿的锁才去删
            if self.path.exists():
                if self._read_pid() == os.getpid():
                    self.path.unlink(missing_ok=True)
        finally:
            self._has_lock = False

    def __enter__(self):
        if not self.acquire(timeout=None):  # 默认阻塞
            raise RuntimeError(f"Failed to acquire lock on {self.path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


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

    def lock(self, key: str) -> Lock:
        """
        实现进程锁。
        锁文件存放在 runtime/locks 目录下。
        by gemini 3
        """
        # 1. 校验 Key 的合法性，防止路径穿越或非法字符
        if not re.match(r'^[a-zA-Z0-9_-]+$', key):
            raise ValueError(f"Invalid lock key: '{key}'. Must match pattern ^[a-zA-Z0-9_-]+$")

        # 2. 获取锁文件存放的 storage 实例 (runtime/locks)
        # sub_storage 会自动创建目录
        lock_storage = self.runtime().sub_storage("locks")

        # 3. 构造完整的锁文件路径
        lock_file_path = lock_storage.abspath() / f"{key}.lock"

        # 4. 返回 FileLocker 实例
        return FileLocker(lock_file_path)
