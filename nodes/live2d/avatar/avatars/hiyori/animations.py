"""Hiyori 的自定义动画轨迹.

每个 async def 是一条命令, 挂到 hiyori 主 channel. 编辑后调 reload_animations 热更新.
"""


async def wave():
    """打招呼: 动一下身体, 再抬左手挥一挥."""
    avatar = get_avatar()
    await avatar.play("Tap@Body", 0)
    await asyncio.sleep(0.2)
    avatar.param("ParamArmLA", 0.8, manual=True)
    await asyncio.sleep(0.6)
    avatar.param("ParamArmLA", 0.0, manual=True)


async def look_around():
    """左右张望: 头和眼球一起转, 最后回正."""
    avatar = get_avatar()
    avatar.param("ParamAngleX", -20, manual=True)
    avatar.param("ParamEyeBallX", -0.7, manual=True)
    await asyncio.sleep(0.7)
    avatar.param("ParamAngleX", 20, manual=True)
    avatar.param("ParamEyeBallX", 0.7, manual=True)
    await asyncio.sleep(0.7)
    avatar.reset()


async def blush():
    """脸红: 腮红 + 笑眼."""
    avatar = get_avatar()
    avatar.param("ParamCheek", 0.9, manual=True)
    avatar.param("ParamEyeLSmile", 0.7, manual=True)
    avatar.param("ParamEyeRSmile", 0.7, manual=True)
    await asyncio.sleep(1.2)
    avatar.reset()
