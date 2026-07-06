"""
G1 系统音量 channel.

无门控. 任何模式可用.
注意 GetVolume 返回 (code, dict), 需解包.
"""

from __future__ import annotations

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

from ._bootstrap import get_audio_client


def build_volume_channel() -> MutableChannel:
    """构建系统音量 channel.

    暴露命令:
      get_volume() -> int          # 当前音量 (0-100)
      set_volume(v: int) -> str    # 设置音量

    Raises (在命令执行时):
      RuntimeError: bootstrap 未完成.
    """
    chan = new_channel(
        name="volume",
        description="G1 系统音量 (影响 PlayStream 输出, 不是 PC2 系统音量)",
    )

    @chan.build.command(always_observe=True)
    async def get_volume() -> int:
        """获取当前 G1 系统音量 (0-100). 解包 GetVolume 的 dict 返回."""
        client = get_audio_client()
        code, data = client.GetVolume()
        if code != 0:
            raise RuntimeError(f"GetVolume failed: code={code}")
        # data 形如 {"volume": N}
        if isinstance(data, dict):
            return int(data.get('volume', 0))
        return int(data)

    @chan.build.command()
    async def set_volume(v: int) -> str:
        """设置 G1 系统音量 (0-100)."""
        if not 0 <= v <= 100:
            raise ValueError(f"volume must be in [0, 100], got {v}")
        client = get_audio_client()
        code = client.SetVolume(int(v))
        return "ok" if code == 0 else f"error:{code}"

    chan.build.instruction(
        # ⚠️ 2026-06-29 校正: "0-100" 是 set_volume 参数语义, 应该在 docstring 里, 不是 instruction.
        # instruction 应该只是 channel 的存在意义, 比如 "G1 系统音量".
        "G1 系统音量. 影响 PlayStream 输出. 0-100."
    )

    return chan
