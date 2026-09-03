# Openbox parameter manifest — canonical default parameter declarations.
#
# Shipped baseline: ParameterModel subclasses declaring typed parameters.  Matrix
# scans via issubclass(obj, ParameterModel) and converts to ParameterSchema via
# to_parameter_schema().
#
# Project extends by:  from ghoshell_moss.matrix.openbox.parameters import *
#
# --
# Openbox Parameter 清单 — 开箱默认 parameter 声明（canonical 基线）。
# ParameterModel 子类声明类型化参数，Matrix 扫描自动发现。

from ghoshell_moss.core.blueprint.parameter import ExampleParameter

__all__ = [
    'ExampleParameter',
]
