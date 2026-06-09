# MOSS Nucleus manifest.
#
# 感知核工厂声明 — NucleusMeta 是生产 Nucleus 的工厂，不是被动协议声明。
# 通过 factory(container) → Nucleus 在运行时生产感知核实例。
#
# Matrix 扫描时通过 isinstance(obj, NucleusMeta) 发现，以 NucleusMeta.name() 为键聚合。
# mode 的 nuclei 叠加在全局之上（dict.update），同键覆盖。
#
# 发现路径：MOSS.manifests.nuclei
# 深入：moss codex get-interface ghoshell_moss.core.blueprint.mindflow

from ghoshell_moss.core.blueprint.mindflow import (
    NucleusMeta,
    Nucleus,
    SignalMeta,
    InputSignal,
)
from ghoshell_moss.core.mindflow.audio_nucleus import audio_nucleus_factory
from ghoshell_container import IoCContainer


class ExampleNucleusMeta(NucleusMeta):
    """感知核工厂的最小示例 — 生产一个接收 InputSignal 的感知核。"""

    def name(self) -> str:
        return "example_nucleus"

    def description(self) -> str:
        return "An example nucleus factory for manifest discovery testing"

    def signals(self) -> list[type[SignalMeta]]:
        return [InputSignal]

    def factory(self, container: IoCContainer) -> Nucleus:
        raise NotImplementedError("Example stub — not intended for runtime use")


example_nucleus_factory = ExampleNucleusMeta()
