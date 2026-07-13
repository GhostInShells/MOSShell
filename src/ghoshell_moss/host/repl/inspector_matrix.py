from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_common.helpers import generate_import_path
import inspect

__all__ = ['MatrixInspector']


class MatrixInspector:
    """用于诊断 Matrix 内部节点状态的工具集。"""

    def __init__(self, matrix: Matrix):
        self._matrix = matrix

    def this(self) -> dict:
        """本 cell 在网络上的身份 (CellPresence)."""
        return self._matrix.this.model_dump()

    def identity(self) -> dict:
        """本 cell 的运行时坐标快照."""
        env = self._matrix.env
        return {
            "mode": env.mode_name,
            "ghost": env.ghost_name,
            "network": env.network,
            "scope": env.network_scope,
            "cell_address": env.this_cell_address,
            "project_id": env.project_id,
            "pid": env.pid,
            "is_sealed": env.is_sealed,
        }

    def info(self) -> dict:
        """Matrix 运行环境的基本状态快照."""
        return {
            "is_running": self._matrix.is_running(),
            "is_host": self._matrix.is_host(),
            "is_host_running": self._matrix.is_host_running(),
        }

    def network(self) -> dict:
        """本 matrix 所接入网络的配置元信息."""
        meta = self._matrix.network
        return {
            "name": meta.name,
            "driver": meta.driver,
            "description": meta.description,
        }

    def contracts(self) -> list[dict]:
        """进程级 IoC 容器中已注册的契约类型."""
        all_contracts_info = []
        for contract in self._matrix.container.contracts(recursively=True):
            if not isinstance(contract, type):
                continue
            doc = inspect.getdoc(contract) or ''
            all_contracts_info.append(dict(
                name=contract.__name__,
                import_path=generate_import_path(contract),
                description=doc.split('\n')[0],
            ))
        return all_contracts_info

    def processes(self) -> dict:
        """子进程灶台状态: 活跃进程 + 最近退出历史."""
        sp = self._matrix.processes
        return {
            "executing": [
                {"index": idx, "name": meta.name, "pid": meta.pid}
                for idx, meta in sp.executing().items()
            ],
            "executed": [
                {"name": meta.name, "pid": meta.pid, "exit_code": meta.exit_code}
                for meta in sp.executed()
            ],
        }
