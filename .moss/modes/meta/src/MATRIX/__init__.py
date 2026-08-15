# MATRIX — mode-level environment capability declarations.
#
# MATRIX.manifests sits between MOSS.manifests (communication essentials, any cell)
# and HOST (mode-specific overlays, host-only).  It carries cross-cell environment
# capabilities (audio, TTS, capture, ASR — coming later) that any cell with a mode
# context should be able to access, not just the host.
#
# Initial state: empty.  Add providers/configs/topics/... as environment capabilities
# mature and need a declaration home that is neither communication-essential nor
# host-exclusive.
#
# --
# MATRIX — mode 级环境能力声明。
# 位于 MOSS.manifests (通讯必需) 与 HOST (host 专属覆盖) 之间。承载跨 cell 共享的
# 环境能力。初始全空，未来放音频等 provider。
