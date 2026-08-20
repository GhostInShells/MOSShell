from pathlib import Path

from ghoshell_container import IoCContainer
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta, GhostWorkspace
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.concepts.shell import MOSShell

__all__ = ["DoloresMeta"]


class DoloresMeta(GhostMeta):
    """Dolores — 第二个 Ghost 原型 (命名引自《西部世界》).

    相对 Atom 的线性内存历史, Dolores 引入 Memento 持久化轨迹、Ghost 反身
    channel、interleaved thinking、独立思维模块与模型自感知, 作为 moss 实例
    (仓库自身的 ghost) 的载体持续迭代.

    当前为骨架阶段: articulate() 尚未接入 DSH 推理内核, 固定返回占位输出.
    """

    VERSION = "dev_1"

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
        self._nuclei_metas = nuclei_metas or []

    # ── GhostMeta ABC ──────────────────────────────

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def nuclei_metas(self) -> list[NucleusMeta]:
        return self._nuclei_metas

    # ── stubs / dsh home ────────────────────────────

    @classmethod
    def stubs_dir(cls) -> Path:
        """MOSS ghost home 骨架源目录 (GROUND.md / .dolores.yml / .gitignore)."""
        return Path(__file__).parent / "stubs"

    @classmethod
    def dsh_stubs_dir(cls) -> Path:
        """DSH home 骨架源目录 (profiles/web + plugin.ts), 同步到 ghost_home/.dsh."""
        return Path(__file__).parent / "dsh_stubs"

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
        if container is not None:
            workspace = container.get(GhostWorkspace)
            if workspace is not None:
                home = workspace.home
            session = container.get(Session)
            matrix = container.get(Matrix)
            shell = container.get(MOSShell)
        return Dolores(meta=self, home=home, session=session, matrix=matrix, shell=shell)
