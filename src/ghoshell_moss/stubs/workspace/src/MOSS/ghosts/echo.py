from ghoshell_moss.ghosts.atom import AtomMeta
from ghoshell_moss.channels.introspect_channel import new_introspect_channel

ghost = AtomMeta(
    name="echo",
    description="壳中的第一声回响。MOSS 默认 Ghost 原型 — 感知、思考、回应。",
    soul_path="soul.md",
    channel=new_introspect_channel(scope="ghoshell_moss"),
)
