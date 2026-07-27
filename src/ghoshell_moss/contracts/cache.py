"""Cross-process shared cache contract — KV storage, hash maps, distributed locks with TTL."""

from __future__ import annotations

import contextlib
from typing import Optional, Generator
from abc import ABC, abstractmethod


class Cache(ABC):
    """
    通用的跨进程共享缓存与仲裁组件.

    提供 KV 存储、Hash map、分布式锁，含 TTL 过期语义。
    """

    @abstractmethod
    def lock(self, key: str, overdue: int = 0) -> bool:
        """
        获取分布式锁.

        :param key: 锁标识
        :param overdue: 锁的超时时间 (秒), 0 表示永不过期
        :return: True 成功获取锁, False 锁已被持有
        """

    @abstractmethod
    def unlock(self, key: str) -> bool:
        """
        释放分布式锁.

        :param key: 锁标识
        :return: True 成功释放, False 锁不存在或已过期
        """

    @contextlib.contextmanager
    def locked(self, key: str, overdue: int = 0) -> Generator[None, None, None]:
        """
        获取锁的 context manager，退出时自动释放.

        :param key: 锁标识
        :param overdue: 锁的超时时间 (秒), 0 表示永不过期
        :raises RuntimeError: 获取锁失败
        """
        if not self.lock(key, overdue):
            raise RuntimeError(f"Failed to acquire lock: {key}")
        try:
            yield
        finally:
            self.unlock(key)

    @abstractmethod
    def set(self, key: str, val: str, exp: int = 0) -> bool:
        """
        设置键值.

        :param key: 键
        :param val: 值
        :param exp: 过期时间 (秒), 0 表示永不过期
        :return: True 成功
        """

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """
        获取键值. 若已过期则返回 None 并惰性删除.

        :param key: 键
        :return: 值或 None
        """

    @abstractmethod
    def expire(self, key: str, exp: int) -> bool:
        """
        更新键的过期时间.

        :param key: 键
        :param exp: 新的过期时间 (秒)
        :return: True 成功, False 键不存在
        """

    @abstractmethod
    def set_member(self, key: str, member: str, value: str) -> bool:
        """
        在 hash map 中设置成员值.

        :param key: hash map 的键
        :param member: 成员名
        :param value: 成员值
        """

    @abstractmethod
    def get_member(self, key: str, member: str) -> Optional[str]:
        """
        从 hash map 中获取成员值.

        :param key: hash map 的键
        :param member: 成员名
        :return: 成员值或 None
        """

    @abstractmethod
    def remove_member(self, key: str, *member: str) -> int:
        """
        从 hash map 中删除成员.

        :param key: hash map 的键
        :param member: 要删除的成员名
        :return: 实际删除的成员数
        """

    @abstractmethod
    def remove(self, *keys: str) -> int:
        """
        删除键 (同时清理其 hash map 成员和锁).

        :param keys: 要删除的键
        :return: 实际删除的键数
        """
