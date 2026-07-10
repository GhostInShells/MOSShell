import asyncio

import pytest

from ghoshell_moss.core.mindflow.audio_nucleus import AudioNucleus
from ghoshell_moss.core.mindflow.audio_signal import AudioAction, AudioSignal


def _speech_final_signal():
    return AudioSignal(action=AudioAction.SPEECH_FINAL).to_signal("hello")


def _audio_nucleus(*, interrupt_on_complete: bool = True) -> AudioNucleus:
    return AudioNucleus(
        name="audio_nucleus",
        description="audio perception signal nucleus",
        target_signal="audio",
        default_prompt="User spoke via voice input. Process the speech.",
        interrupt_on_complete=interrupt_on_complete,
    )


@pytest.mark.asyncio
async def test_audio_nucleus_complete_impulse_interrupts_by_default():
    nucleus = _audio_nucleus()

    async with nucleus:
        nucleus.add_signal(_speech_final_signal())
        await asyncio.sleep(0.1)

        impulse = nucleus.peek()
        assert impulse is not None
        assert impulse.complete is True
        assert impulse.interrupt is True


@pytest.mark.asyncio
async def test_audio_nucleus_can_disable_complete_impulse_interrupt():
    nucleus = _audio_nucleus(interrupt_on_complete=False)

    async with nucleus:
        nucleus.add_signal(_speech_final_signal())
        await asyncio.sleep(0.1)

        impulse = nucleus.peek()
        assert impulse is not None
        assert impulse.complete is True
        assert impulse.interrupt is False
