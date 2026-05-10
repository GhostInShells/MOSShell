from abc import ABC, abstractmethod
from typing import Protocol, Union
import re

import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── 跨平台文件锁 ──────────────────────────────────────────────
if sys.platform != "win32":
    import fcntl

    def _flock_ex(fd: int):
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _flock_ex_nb(fd: int) -> bool:
        """尝试排他锁（非阻塞），成功返回 True，已被占用返回 False"""
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _flock_un(fd: int):
        fcntl.flock(fd, fcntl.LOCK_UN)
else:
    import msvcrt

    def _flock_ex(fd: int):
        """排他锁（阻塞）—— msvcrt 没有原生阻塞锁，手动重试"""
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.05)

    def _flock_ex_nb(fd: int) -> bool:
        """尝试排他锁（非阻塞）"""
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _flock_un(fd: int):
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

# ── 公开接口 ──────────────────────────────────────────────────
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
    跨平台文件锁。
    Linux/macOS: 基于 fcntl.flock，内核级原子性
    Windows:     基于 msvcrt.locking，字节级范围锁

    两种方式都支持非阻塞/阻塞/超时。
    """

    def __init__(self, lock_path: Path):
        self.path = lock_path
        self._fd: Optional[int] = None

    def _is_pid_running(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def is_locked(self, /, by_self: bool = False) -> bool:
        """
        检查锁是否被占用。
        """
        if self._fd is not None:
            return True if by_self else True

        if not self.path.exists():
            return False

        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                locked = _flock_ex_nb(fd)
                if locked:
                    _flock_un(fd)
                    os.close(fd)
                    return False
                else:
                    os.close(fd)
                    return True
            except Exception:
                os.close(fd)
                return True
        except (FileNotFoundError, PermissionError):
            return False

    def acquire(self, timeout: Optional[float] = 0) -> bool:
        """
        获取锁。
        """
        if self._fd is not None:
            return True

        start_time = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            fd = None
            try:
                fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)

                # Windows msvcrt 要求先 seek 到 0 才能 locking
                os.lseek(fd, 0, os.SEEK_SET)

                if not _flock_ex_nb(fd):
                    # 锁被占用
                    os.close(fd)
                    fd = None

                    if timeout == 0:
                        return False
                    if timeout is not None and (time.time() - start_time) >= timeout:
                        return False

                    time.sleep(0.05)
                    continue

                # 拿到锁，写入 PID
                os.ftruncate(fd, 0)
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(fd, str(os.getpid()).encode())

                self._fd = fd
                return True

            except Exception:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                raise

    def release(self) -> None:
        """
        释放锁并关闭文件描述符。
        """
        if self._fd is not None:
            try:
                os.lseek(self._fd, 0, os.SEEK_SET)
                _flock_un(self._fd)
                os.close(self._fd)
            finally:
                self._fd = None

    def __enter__(self):
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
        if not re.match(r'^[a-zA-Z0-9_-]+$', key):
            raise ValueError(f"Invalid lock key: '{key}'. Must match pattern ^[a-zA-Z0-9_-]+$")

        lock_storage = self.runtime().sub_storage("locks")
        lock_file_path = lock_storage.abspath() / f"{key}.lock"
        return FileLocker(lock_file_path)
