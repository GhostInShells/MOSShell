# Mode 的 main channel。
# Mode 的 __main__ channel 是唯一的生效 main channel。

from ghoshell_moss import new_default_shell_main_channel
from ghoshell_moss.core.blueprint.channel_builder import new_channel

main = new_default_shell_main_channel()

# ── g1_system ──────────────────────────────────────────────────────────────
# Stateless query — 电池/主板状态. 不依赖 daemon, 每次命令调一次 RPC.

g1_system = new_channel(
    name="g1_system",
    description="G1 battery & mainboard health. Stateless read-only.",
)


@g1_system.build.command(name="read")
def _read() -> str:
    """Read G1 battery SOC, voltage, current, temperature, and mainboard status.

    Returns a compact XML summary. No side effects.
    """
    from ghoshell_moss_contrib.unitree.g1.runtime import system_info

    try:
        snap = system_info.read()
    except Exception as e:
        return f"<g1.system error='{e}' />"
    return system_info.to_xml_text(snap)


main.import_channels(g1_system)
