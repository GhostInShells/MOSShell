"""EgoBesideMindConfig — the beside organ's switch and budgets.

Stored in the ghost's own config store (``<ghost_home>/.dolores/configs``), which is the layer
meant to be mutated at runtime: a ghost may want a different direction count or effort without
touching the ghost-wide assembly (``.dolores.yml``). Each concurrent-mind mechanism gets its own
config, so turning one off never reaches into another.

``enabled`` is a **boot-time** switch: the channel factory reads it to decide whether the organ
exists, and the runtime is only built when it is on. Flipping it takes effect on the next start.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field
from ghoshell_moss.contracts.configs import ConfigType

__all__ = ["EgoBesideMindConfig"]


class EgoBesideMindConfig(ConfigType):
    """The beside think knobs. Defaults are the fallback; a missing key uses them."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(
        default=True,
        description=(
            "build the beside organ at all. The theoretical default is off — a ghost that does "
            "not ask for parallel thought should not carry the machinery; it is on during "
            "development so the mechanism can be exercised, and will flip before release."
        ),
    )
    directions_max: int = Field(
        default=5,
        ge=1,
        description=(
            "the cap on directions in one `think` call. Each direction is one off-record model "
            "round, so the cap is a cost bound, not a UI limit; extra directions are dropped. "
            "The command's docstring renders this number in, so raising it here is the only edit."
        ),
    )
    effort: str = Field(
        default="max",
        description=(
            "reasoning effort for each direction. `max`: a direction is one shot — it gets no "
            "second pass and its answer is read by the ego as a considered one, so a shallow "
            "take is noise with a confident face. The cost is per direction, hence the cap."
        ),
    )
    max_tokens: int = Field(
        default=0,
        ge=0,
        description="per-direction output cap; 0 = the session's own cap.",
    )
    timeout: float = Field(
        default=120.0,
        gt=0,
        description="seconds one direction may take before it is reported as an error.",
    )

    @classmethod
    def conf_name(cls) -> str:
        return 'beside_mind'
