from ghoshell_moss.core.blueprint.states_channel import new_prime_channel, PrimeChannel
from ghoshell_moss.core.blueprint.mindflow import Mindflow

__all__ = ['build_mindflow_channel']


def build_mindflow_channel(
        mindflow: Mindflow,
        name: str = "mindflow",
) -> PrimeChannel:
    channel = new_prime_channel(
        name,
        description=mindflow.description(),
    )

    @channel.build.virtual_children
    def mindflow_nuclei_children():
        channels = {}
        for key, nucleus in mindflow.nuclei().items():
            if not nucleus.is_running():
                continue
            if chan := nucleus.as_channel():
                channels[key] = chan
        return channels

    return channel
