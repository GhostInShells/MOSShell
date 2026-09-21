"""zhihu-cli 的 subprocess 封装。

node 不重写 CLI 已封装的鉴权 / 时间戳 / 重试 / 错误处理（官方文档明确警告不要），
只负责：解析 binary 位置、跑命令、读 JSON stdout、把 exit code 映射成错误。

所有调用走 skill 的 ``run.sh`` —— 它负责 is_ready 检查 + ``exec`` 到二进制。
``auth set --secret-stdin`` 走 stdin，secret 不进 argv、不落盘。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

__all__ = ["ZhihuCli"]

_DEFAULT_BINARY = Path.home() / "Library" / "Application Support" / "zhihu-cli" / "current" / "zhihu-cli"


class CliError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class ZhihuCli:
    def __init__(self, skill_dir: Path, binary: Path | None = None) -> None:
        self._run_sh = skill_dir / "scripts" / "run.sh"
        self._binary = binary or _DEFAULT_BINARY

    async def status(self) -> dict:
        return await self._run_sh("status")

    async def capabilities(self) -> dict:
        return await self._run_binary("capabilities")

    async def run(self, *argv: str) -> dict:
        """跑一个业务命令（如 ``me stats --type all``）。返回服务端原始 JSON。"""
        return await self._run_binary(*argv)

    async def auth_set(self, secret: str) -> dict:
        """把 secret 经 stdin 写入系统密钥链。secret 不进 argv、不落盘。"""
        return await self._run_binary("auth", "set", "--secret-stdin", stdin=secret)

    async def help(self, *argv: str) -> str:
        """跑 ``<command> --help``，返回原始文本（help 不是 JSON）。"""
        proc = await asyncio.create_subprocess_exec(
            str(self._binary), *argv, "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return (out or err).decode(errors="replace").strip()

    async def _run_sh(self, *argv: str) -> dict:
        return await self._exec("bash", str(self._run_sh), *argv)

    async def _run_binary(self, *argv: str, stdin: str | None = None) -> dict:
        return await self._exec(str(self._binary), *argv, stdin=stdin)

    async def _exec(self, *argv: str, stdin: str | None = None) -> dict:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            )
        except OSError as e:
            return {"ok": False, "error": {"code": "SPAWN_FAILED", "message": str(e)}}
        out, err = await proc.communicate(input=(stdin or "").encode())
        text = out.decode(errors="replace")
        if not text.strip():
            return {"ok": False, "error": {
                "code": f"EXIT_{proc.returncode}",
                "message": err.decode(errors="replace")[:500],
            }}
        try:
            return json.loads(text)
        except ValueError:
            return {"ok": False, "error": {
                "code": f"EXIT_{proc.returncode}",
                "message": (text + err.decode(errors="replace"))[:500],
            }}

    def resolve_binary(self) -> str:
        return str(self._binary)
