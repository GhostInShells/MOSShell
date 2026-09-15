"""avatar node entry point — Live2D 形象驱动脚手架.

Start:  moss nodes run nodes/live2d/avatar
        moss nodes run nodes/live2d/avatar -- --avatar hiyori
Debug:  .venv/bin/python main.py --avatar miku     # ad-hoc (CLI 不是 owner)

装配顺序 (固定):

    select kit → parse spec → Avatar (事件面) → AvatarBridge (同源 server)
                                                 ↓
                                        load_channel (kit 的 channel.py 或自动映射)
                                                 ↓
                                        matrix.provide_channel

换形象 = 换 `--avatar` 重启, 没有运行期切换. `--avatar` 优先级: 命令行 > env > 第一个可用套件.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from avatar_node import (
    Avatar,
    AvatarBridge,
    AvatarNotFoundError,
    load_channel,
    load_spec,
    model_url,
    select,
)

from ghoshell_moss.core.blueprint.matrix import Matrix

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770


def parse_args(argv: list[str]) -> dict:
    """三个启动覆盖: --avatar <name> / --host <addr> / --port <n>. 其余透传忽略."""
    cfg = {
        "avatar": os.getenv("LIVE2D_AVATAR") or None,
        "host": os.getenv("LIVE2D_HOST", DEFAULT_HOST),
        "port": int(os.getenv("LIVE2D_PORT", DEFAULT_PORT)),
    }
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--avatar", "--host", "--port") and i + 1 < len(argv):
            key = arg[2:]
            cfg[key] = int(argv[i + 1]) if key == "port" else argv[i + 1]
            i += 2
            continue
        i += 1
    return cfg


_BACKDROP_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _backdrop_names() -> tuple[str, ...]:
    backdrop_dir = _NODE_DIR / "backdrop"
    if not backdrop_dir.is_dir():
        return ()
    return tuple(sorted(p.name for p in backdrop_dir.iterdir() if p.suffix.lower() in _BACKDROP_EXTS))


def _default_backdrop() -> str | None:
    """`backdrop/default.<ext>` 存在就用它当默认背板 (否则 None, 页面纯色底)."""
    backdrop_dir = _NODE_DIR / "backdrop"
    if not backdrop_dir.is_dir():
        return None
    for p in sorted(backdrop_dir.iterdir()):
        if p.stem == "default" and p.suffix.lower() in _BACKDROP_EXTS:
            return f"/backdrop/{p.name}"
    return None


async def main(matrix: Matrix) -> None:
    logger = matrix.logger
    cfg = parse_args(sys.argv[1:])

    try:
        kit = select(_NODE_DIR, cfg["avatar"])
    except AvatarNotFoundError:
        logger.exception("avatar node: no usable avatar kit")
        raise

    logger.info(
        "avatar node: kit=%s model=%s channel=%s",
        kit.name,
        kit.model_json.name,
        kit.channel_file or "auto",
    )

    avatar = Avatar(
        kit.name,
        load_spec(kit),
        model_url=model_url(kit),
        backdrop=_default_backdrop(),
        backdrop_names=_backdrop_names(),
        persona=kit.persona,
        logger=logger,
    )
    bridge = AvatarBridge(
        avatar,
        web_dir=_NODE_DIR / "src" / "avatar_node" / "web",
        vendor_dir=_NODE_DIR / "vendor",
        model_dir=kit.model_dir,
        backdrop_dir=_NODE_DIR / "backdrop",
        host=cfg["host"],
        port=cfg["port"],
        logger=logger,
    )
    await bridge.start()

    try:
        await matrix.publish_event(f"live2d avatar `{kit.name}` alive; page {bridge.url}")
    except Exception as e:  # 事件发布失败不该拦住躯体
        logger.debug("publish_event failed: %s", e)

    channel = await load_channel(kit, avatar)
    try:
        await matrix.provide_channel(channel)
    finally:
        await bridge.stop()


if __name__ == "__main__":
    Matrix.discover().run(main)
