"""MossRuntime 交付判据 — shell 拿到 Speech 的唯一入口.

锚定 ``resolve_shell_speech`` 的契约, 也就是 ``--voice`` 四态在说侧的执行点:

- ``speech=False`` → None: shell 不挂 say/mute, 即使 Voice 手里握着 TTSSpeech.
  (``--voice none`` 与 ``--voice listen`` 走这一条 —— 只听模式说不出来。)
- Voice 缺席 → None.
- ``speech=True`` 且 Voice 持有 → 交付同一个实例 (不复制、不重建).

``--voice`` 的 flag 落在两处: 这一处决定 shell 有没有说话的**器官**, ``voice.run()``
那一处决定装不装说侧桥的**线**。只关前者是 "说不出来但联动还在", 只关后者是
"关得掉桥、关不掉嘴" —— 本测试钉住前者。

不测 InterleavedVoice 装线 (那是 test_voice.py 的范围).
"""

from ghoshell_moss.core.speech import MockSpeech
from ghoshell_moss.host.moss_runtime import resolve_shell_speech


class _HoldingVoice:
    """最小 Voice — 只持有 speech, 不管装线 (本测试不触 run)。"""

    def __init__(self, speech):
        self._speech = speech

    def speech(self):
        return self._speech


def test_speech_disabled_hands_nothing_to_shell():
    """--voice none / listen: 说侧关掉 → shell 拿不到 Speech, 因此不挂 say/mute."""
    voice = _HoldingVoice(MockSpeech())
    assert resolve_shell_speech(voice, speech=False) is None


def test_speech_enabled_hands_the_held_instance():
    """--voice speak / all: 交付 Voice 持有的那个实例本身."""
    held = MockSpeech()
    voice = _HoldingVoice(held)
    assert resolve_shell_speech(voice, speech=True) is held


def test_missing_voice_hands_nothing():
    """IoC 里没有 Voice: 两种 flag 都拿不到 Speech."""
    assert resolve_shell_speech(None, speech=True) is None
    assert resolve_shell_speech(None, speech=False) is None
