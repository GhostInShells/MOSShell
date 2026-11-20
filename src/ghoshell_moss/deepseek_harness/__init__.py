"""deepseek_harness — the dsh seam, one generation per subpackage.

Only generation packages live here; ``v0_2_0`` is the live one. A new generation is a new
subpackage, never an edit of the one before it — the previous generation is deleted once
nothing imports it (the 0.1.x protocol/process layer was removed in D21).
"""
