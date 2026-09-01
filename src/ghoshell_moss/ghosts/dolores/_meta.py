from pathlib import Path

from ghoshell_container import IoCContainer
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.host import MossSystemPrompter
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.concepts.shell import MOSShell

from .nucleus import DoloresEgoNucleusMeta

__all__ = ["DoloresMeta"]


class DoloresMeta(GhostMeta):
    """Dolores — 第二个 Ghost 原型 (命名引自《西部世界》).

    相对 Atom 的线性内存历史, Dolores 引入 Memento 持久化轨迹、Ghost 反身
    channel、interleaved thinking、独立思维模块与模型自感知, 作为 moss 实例
    (仓库自身的 ghost) 的载体持续迭代.

    当前为骨架阶段: articulate() 尚未接入 DSH 推理内核, 固定返回占位输出.
    """

    VERSION = "dev_2"

    def __init__(
        self,
        name: str = "dolores",
        description: str = (
            "Dolores — 第二个 Ghost 原型。以 DSH 为推理中枢, MOSS 保留记忆 "
            "(Memento) / 执行 (CTML channels) / 感知 (audio/vision)。"
        ),
        nuclei_metas: list[NucleusMeta] | None = None,
    ):
        self._name = name
        self._description = description
        # Dolores 默认挂 ego 自醒 nucleus (self-wake 通道); 调用方显式传 nuclei_metas 时完全替换.
        self._nuclei_metas = (
            nuclei_metas if nuclei_metas is not None else [DoloresEgoNucleusMeta()]
        )

    # ── GhostMeta ABC ──────────────────────────────

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def nuclei_metas(self) -> list[NucleusMeta]:
        return self._nuclei_metas

    # ── instruction 段 (结构化派生, 不写死提示词) ──────────

    def prototype_instruction(self) -> str:
        """原型元信息 — 型号 + 版本, 从结构化 meta 派生."""
        # todo: 补充行为逻辑 — 模型的正常输出一律解析为 CTML 驱动躯体,
        # 配套工具 (channel) 负责控制/交互, 行为面随原型迭代在此扩展.
        return "\n".join([
            f"prototype: {self.prototype()}",
            f"version: {self.VERSION}",
        ])

    def identity_instruction(self) -> str:
        """身份描述 — name + description, 从结构化 meta 派生."""
        return "\n".join([
            f"name: {self.name()}",
            f"description: {self.description()}",
        ])

    # ── stubs / dsh home ────────────────────────────

    @classmethod
    def stubs_dir(cls) -> Path:
        """MOSS ghost home 骨架源目录 (GROUND.md / .dolores.yml / .gitignore)."""
        return Path(__file__).parent / "stubs"

    @classmethod
    def dsh_stubs_dir(cls) -> Path:
        """DSH home 骨架源目录 (profiles/web), 同步到 ghost_home/.dsh."""
        return Path(__file__).parent / "dsh_stubs"

    @classmethod
    def dsh_plugin_stub(cls) -> Path:
        """DSH ghost plugin 源文件 (独立 stub), 创建时复制为 ghost_home/.dsh/profiles/web/plugin.ts."""
        return Path(__file__).parent / "dsh_plugin" / "moss-dolores-ghost-plugin.ts"

    # ── factory ─────────────────────────────────────

    def factory(self, container: IoCContainer) -> Ghost:
        """只做路径/依赖获取, 不产生副作用.

        stubs 同步 (文件 IO + session.output) 与 dsh 启动 (matrix.processes)
        收敛在 Dolores.__aenter__, 测试可传 tmp home 直接构造而不触发写盘.
        """
        from ._runtime import Dolores

        home: Path | None = None
        session: Session | None = None
        matrix: Matrix | None = None
        shell: MOSShell | None = None
        base_instruction: str | None = None
        if container is not None:
            matrix = container.get(Matrix)
            if matrix is not None:
                home = matrix.ghost_home
            session = container.get(Session)
            shell = container.get(MOSShell)
            prompter = container.get(MossSystemPrompter)
            if prompter is not None:
                base_instruction = prompter.base_instruction()
        return Dolores(
            meta=self,
            home=home,
            session=session,
            matrix=matrix,
            shell=shell,
            base_instruction=base_instruction,
        )
