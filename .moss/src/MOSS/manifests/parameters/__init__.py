# Parameter manifest — parameter protocol declarations.
#
# Define ParameterModel subclasses to declare typed parameters.  Matrix scans
# via issubclass(obj, ParameterModel), converts each to ParameterSchema via
# to_parameter_schema().
#
# Mode extends by: from MOSS.manifests.parameters import *
#
# --
# Parameter 清单 — 参数协议声明。
# 用 ParameterModel 子类声明类型化参数，Matrix 扫描自动发现。

from ghoshell_moss.core.blueprint.parameter import ExampleParameter
