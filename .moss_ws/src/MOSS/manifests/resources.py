# MOSS Resource manifest.
#
# 资源存储声明：声明环境中可寻址的资源数据集。
#
# 模式约定：定义一个 ResourceStorageMeta 实例。Matrix 扫描时通过
# isinstance(obj, ResourceStorageMeta) 发现，以 {scheme}:{host} 为键聚合。
# 例如 pil-image:workspace-assets 表示本地图片资源，scheme=pil-image, host=workspace-assets。
#
# mode 的 resources 叠加在全局之上（dict.update），同键覆盖。
#
# 发现路径：MOSS.manifests.resources
# 深入：moss howtos read host-dev/add-a-resource-storage.md

from ghoshell_moss.core.resources.local_image import LocalImageResourceMeta

local_image_storage_meta = LocalImageResourceMeta()
