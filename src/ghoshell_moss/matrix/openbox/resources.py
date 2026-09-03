# Openbox resource manifest — canonical default resource storage.
#
# Shipped baseline: ResourceStorageMeta instances declaring addressable datasets.
# Matrix scans via isinstance(obj, ResourceStorageMeta), keyed by scheme.
#
# Project extends by:  from ghoshell_moss.matrix.openbox.resources import *
#
# --
# Openbox Resource 清单 — 开箱默认资源存储（canonical 基线）。
# 声明可寻址的资源数据集，Matrix 扫描自动发现。

from ghoshell_moss.resources.local_image import LocalImageResourceMeta

__all__ = [
    'local_image_storage_meta',
]

# local image resource storage
local_image_storage_meta = LocalImageResourceMeta()
