# Resource manifest — resource storage declarations.
#
# Define ResourceStorageMeta instances to declare addressable resource datasets.
# Matrix scans via isinstance(obj, ResourceStorageMeta), keyed by scheme.
#
# Mode extends by: from MOSS.manifests.resources import *
#
# --
# Resource 清单 — 资源存储声明。
# 声明可寻址的资源数据集，Matrix 扫描自动发现。

from ghoshell_moss.core.resources.local_image import LocalImageResourceMeta

# local image resource storage
local_image_storage_meta = LocalImageResourceMeta()
