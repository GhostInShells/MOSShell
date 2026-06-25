# MOSS Resource manifest.
#
# 资源存储声明：声明环境中可寻址的资源数据集。
#
# 模式约定：定义一个 ResourceStorageMeta 实例。Matrix 扫描时通过
# isinstance(obj, ResourceStorageMeta) 发现，以 {scheme}:{host} 为键聚合。
# 例如 pil-image:workspace-assets 表示本地图片资源，scheme=pil-image, host=workspace-assets。
#
# 发现路径：MOSS.manifests.resources

from ghoshell_moss.core.resources.local_image import LocalImageResourceMeta

# 本地图片的配置.
local_image_storage_meta = LocalImageResourceMeta()
