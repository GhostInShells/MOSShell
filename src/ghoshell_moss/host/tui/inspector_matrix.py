from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_common.helpers import generate_import_path
import inspect

__all__ = ['MatrixREPL']


class MatrixREPL:
    """
    用于诊断 Matrix 内部节点状态的工具集。
    """

    def __init__(self, matrix: Matrix):
        self._matrix = matrix

    def list_cells(self, limit: int = 0) -> list[dict]:
        """列出当前网络中所有已发现的 Cell 节点状态。"""
        result = [cell.to_dict() for cell in self._matrix.list_cells().values()]
        if limit <= 0:
            return result
        return result[:limit]

    def this_cell(self) -> dict:
        """获取当前运行节点 (This Cell) 的详细元数据。"""
        cell = self._matrix.this
        return cell.to_dict()

    def info(self) -> dict:
        """返回 Matrix 运行环境的基本配置快照。"""
        return {
            "mode": self._matrix.mode,
            "is_running": self._matrix.is_running(),
            "moss_running": self._matrix.is_moss_running()
        }

    def contracts(self) -> list[dict]:
        """返回进程级可依赖注入的对象."""
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
