from abc import ABC, abstractmethod
from typing import Protocol, Union
import re

import fcntl
import os
import time
from pathlib import Path
from typing import Optional

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
    基于 fcntl.flock 的增强型进程锁。
    由 Gemini 3 重写：内核级原子性，支持非阻塞/阻塞/超时。
    """

    def __init__(self, lock_path: Path):
        self.path = lock_path
        self._fd: Optional[int] = None

    def _is_pid_running(self, pid: int) -> bool:
        if pid <= 0: return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def is_locked(self, /, by_self: bool = False) -> bool:
        """
        检查锁是否被占用。
        """
        # 如果我自己持有着文件描述符，那肯定锁着
        if self._fd is not None:
            return True if by_self else True

        if not self.path.exists():
            return False

        try:
            # 尝试以只读方式打开并尝试加锁（非阻塞）
            with open(self.path, 'r') as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # 能加锁成功，说明之前没被别人锁住
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return False
                except BlockingIOError:
                    # 加锁失败，说明被别人占着
                    return True
        except (FileNotFoundError, PermissionError):
            return False

    def acquire(self, timeout: Optional[float] = 0) -> bool:
        """
        核心逻辑：
        1. 即使 flock 会随进程消失，我们依然写入 PID，方便人工排查。
        2. 使用 O_RDWR 保持文件句柄常驻以持有内核锁。
        """
        # 防止重入
        if self._fd is not None:
            return True

        start_time = time.time()

        # 确保目录存在
        self.path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                # 以读写模式打开（不使用 O_TRUNC 以免破坏读取逻辑）
                fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)

                # 尝试内核加锁 (LOCK_EX: 排他锁, LOCK_NB: 非阻塞)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    # 锁被占用
                    os.close(fd)

                    if timeout == 0: return False
                    if timeout is not None and (time.time() - start_time) >= timeout:
                        return False

                    time.sleep(0.05)
                    continue

                # 成功拿到了内核锁！
                # 写入当前 PID 以供调试（覆盖原有内容）
                os.ftruncate(fd, 0)
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(fd, str(os.getpid()).encode())

                self._fd = fd
                return True

            except Exception:
                # 发生意外（如权限问题），确保关闭 FD
                if 'fd' in locals(): os.close(fd)
                raise

    def release(self) -> None:
        """
        释放内核锁并关闭文件描述符。
        注意：不主动 unlink 文件，保留文件作为“占位符”是 Unix 锁的常见做法，
        可以减少创建文件时的竞态条件。
        """
        if self._fd is not None:
            try:
                # 释放内核锁
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            finally:
                self._fd = None

    def __enter__(self):
        # 按照你的接口：None 是阻塞，0 是快败
        if not self.acquire(timeout=None):
            raise RuntimeError(f"Could not acquire lock on {self.path}")
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
