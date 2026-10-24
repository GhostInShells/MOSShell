# HOST Mode Manifests — mode-specific capability declarations.
#
# Each sub-package overrides or extends the corresponding global manifest:
#   channels.py — main channel entry point (not a package)
#   providers/  — IoC providers (extends MOSS.manifests.providers)
#   configs/    — configuration overrides
#   signals/    — mode-specific signal declarations
#   topics/     — mode-specific topic declarations
#   nuclei/     — nucleus (perception) declarations
#   resources/  — mode-specific resource storage declarations
#
# Discovered at runtime by ScannedModeManifests.
#
# --
# HOST Mode Manifest — mode 专属能力声明。
# 每个子包覆盖或扩展全局 manifest，Matrix 启动时扫描发现。
