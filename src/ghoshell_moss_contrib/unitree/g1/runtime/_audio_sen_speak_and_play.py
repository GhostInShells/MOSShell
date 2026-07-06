"""
_audio_sen_speak_and_play — Audio runtime 双工体验: 一次性把 G1 全部音频能力跑一圈.

设计意图:
  你实机时间有限. 这个脚本目标是 "一次会话里, 让你把 audio.py 的每条能力
  都用上, 并形成关于 G1 真实物理行为的肌肉记忆". 默认动作 (直接打字) 是
  TTS — 跟人跟 G1 说话最自然的入口. 探针型实测命令用 `:` 前缀, 不冲突.

  关键想让你感受到的几条:
  1. **TTS 是否能被打断** — speak 期间再次 speak / 推 tone / cancel, 看 G1 行为.
     这是 audio.py docstring 里标 TODO 的点 ("PlayStop 能否中断 TtsMaker 待实测").
  2. **PlayStream 的抢占 vs 拼接** — `:tone` 新 stream_id 抢占, `:tone+` 续到
     当前 stream_id 拼接. 是 audio.play_pcm 的核心语义.
  3. **is_playing 估算偏差** — `:status on` 打开后, 后台线程每秒打印 runtime 的
     "我估计还在播 N.Ns" vs 你耳朵听到的真实状态. 偏差大表示估算公式要调.
  4. **音量范围** — `:vol` 读 → `:vol N` 设 → `:vol` 读, 校准 0-9 还是 0-100.

  这是 channel 真实使用 scenario 的最小模拟:
    "channel:audio 命令喂模型, 模型说话 / 放预录音 / 抢断重说".

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._audio_sen_speak_and_play <nic>

  nic 示例: eth0 / enp3s0 — 见 docs/hardware.md.

前置:
  - G1 已开机 (任何模式 — audio 不依赖运动模式)
  - PC2 已 `uv sync`, sdk 链路 ok (跑过 scripts/sdk/00_import_verify.py 通)
  - **G1 喇叭旁边别放贵重物品** — TTS 默认音色不柔和, 长文本会持续一段时间.
    建议提前调小: 进 REPL 第一件事 `:vol 30` 之类.

预期:
  >>> 你好 G1                           ← 直接打字 = TTS
  [speak] code=0 text='你好 G1'  est_dur=0.5s
  → G1 嘴里说话 ←

  >>> :tone                              ← 推 440Hz 1s sine wave
  [play_pcm] stream_id=moss_1734...  len=32000B  est_dur=1.00s
  → G1 嘴里 1 秒哔声 ←

  >>> :tone+                             ← 续到同 stream_id, 应该拼接 (没新一声开始)
  [play_pcm] stream_id=moss_1734... (resumed)  len=32000B  est_dur=1.00s

  >>> :status on
  [status] 0.5s running=True playing=True rem=0.5s last_stream=moss_...
  [status] 1.5s running=True playing=False rem=0.0s last_stream=moss_...

  Ctrl+C 退出 → audio.stop() + 摘要.

命令快查 (REPL 内 `:help` 重新打印):
  <文本>           speak(text)                      默认动作
  :sp ID <文本>    speak(text, speaker_id=ID)       测音色
  :tone [HZ] [SEC] 推 sine wave PCM (默认 440Hz 1s) 新 stream_id (抢占)
  :tone+ [HZ] [SEC] 同上, 但续到当前 stream_id     拼接
  :wav PATH        play_wav_file(PATH)              测 wav 路径 + 格式校验
  :cancel  /  :c   cancel()                         PlayStop
  :vol             get_volume()                     读
  :vol N           set_volume(N)                    设 (范围待实测)
  :health  /  :h   health()                         打印
  :status on|off   每秒后台状态行 on/off            看 is_playing 估算
  :help            重新打印命令快查
  :quit  /  :q     干净退出
"""
from __future__ import annotations

import math
import struct
import sys
import threading
import time
from typing import Optional

from prompt_toolkit import PromptSession, patch_stdout

from ghoshell_moss_contrib.unitree.g1.runtime import audio
from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap


# ── 测试用 sine wave 生成 ─────────────────────────────────────────────────
# 用 stdlib (math + struct) 而非 numpy — runtime/asr.py 没引 numpy, 保持一致.
# 16kHz mono s16le, 跟 G1 PlayStream 契约对齐.

def _gen_tone(freq_hz: float = 440.0, sec: float = 1.0, *, amp: float = 0.3) -> bytes:
    """生成一段 sine wave PCM (16kHz mono s16le).

    :param freq_hz: 频率 Hz. 440 = 标准 A4, 比较"友好"; 200 沉, 1000 刺耳.
    :param sec: 时长秒. 默认 1.0 让你听清楚抢占 vs 拼接.
    :param amp: 振幅 0-1. 默认 0.3 (~-10dB), 别一上来吓人.
    """
    sample_rate = 16000
    n = int(sample_rate * sec)
    max_val = int(amp * 32767)
    samples = bytearray(n * 2)
    for i in range(n):
        v = int(max_val * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        struct.pack_into("<h", samples, i * 2, v)
    return bytes(samples)


# ── 后台状态线程 ──────────────────────────────────────────────────────────
# 默认关. `:status on` 打开 — 让你边操作边看 audio.is_playing() 估算的衰减,
# 跟你耳朵听到的真实结束时刻对比.

_status_stop_event: Optional[threading.Event] = None
_status_thread: Optional[threading.Thread] = None
_status_start_at: float = 0.0


def _status_loop() -> None:
    """每秒打印 audio.health 关键字段. patch_stdout 包裹下不破坏底部 prompt."""
    assert _status_stop_event is not None
    while not _status_stop_event.is_set():
        h = audio.health()
        dt = time.time() - _status_start_at
        print(
            f"[status] +{dt:5.1f}s  "
            f"running={h['running']}  "
            f"playing={h['is_playing_est']}  "
            f"rem={h['estimated_remaining_sec']:.1f}s  "
            f"last_stream={h['last_stream_id'][-12:] if h['last_stream_id'] else '-'}"
        )
        _status_stop_event.wait(1.0)


def _status_on() -> None:
    global _status_stop_event, _status_thread, _status_start_at
    if _status_thread is not None and _status_thread.is_alive():
        print("[status] 已在运行.")
        return
    _status_stop_event = threading.Event()
    _status_start_at = time.time()
    _status_thread = threading.Thread(target=_status_loop, name="audio-sen-status", daemon=True)
    _status_thread.start()
    print("[status] 周期状态 ON (每秒一行).")


def _status_off() -> None:
    global _status_stop_event, _status_thread
    if _status_stop_event is not None:
        _status_stop_event.set()
    if _status_thread is not None:
        _status_thread.join(timeout=1.5)
    _status_stop_event = None
    _status_thread = None
    print("[status] OFF.")


# ── 命令分发 ────────────────────────────────────────────────────────────

_HELP = """\
命令快查:
  <文本>           speak(text)
  :sp ID <文本>    speak(text, speaker_id=ID)
  :tone [HZ] [SEC] 推 sine wave PCM (默认 440Hz 1s), 新 stream_id 抢占
  :tone+ [HZ] [SEC] 同上, 续到当前 stream_id 拼接
  :wav PATH        play_wav_file(PATH)
  :cancel | :c     cancel()
  :vol             get_volume()
  :vol N           set_volume(N)
  :health | :h     health()
  :status on|off   周期状态行 (看 is_playing 估算)
  :help            重新打印
  :quit | :q       退出
"""


def _handle_speak(text: str, *, speaker_id: int = 0) -> None:
    code = audio.speak(text, speaker_id=speaker_id)
    est = audio.health()["estimated_remaining_sec"]
    print(f"[speak] code={code}  speaker_id={speaker_id}  text={text!r}  est_dur={est:.2f}s")


def _handle_tone(rest: str, *, resume: bool) -> None:
    parts = rest.split()
    try:
        hz = float(parts[0]) if len(parts) >= 1 else 440.0
        sec = float(parts[1]) if len(parts) >= 2 else 1.0
    except ValueError:
        print(f"[tone] 参数解析失败, 用法: :tone[+] [HZ] [SEC]")
        return
    pcm = _gen_tone(hz, sec)
    current = audio.health()["last_stream_id"]
    sid = current if (resume and current) else None
    new_sid = audio.play_pcm(pcm, stream_id=sid)
    tag = "(resumed)" if resume and current else "(new)"
    print(
        f"[play_pcm] stream_id={new_sid[-12:]} {tag}  "
        f"len={len(pcm)}B  est_dur={sec:.2f}s  hz={hz}"
    )


def _handle_wav(path: str) -> None:
    try:
        sid = audio.play_wav_file(path)
    except (ValueError, FileNotFoundError) as e:
        print(f"[wav] 失败: {e}")
        return
    print(f"[play_wav] stream_id={sid[-12:]}  path={path}")


def _handle_vol(rest: str) -> None:
    rest = rest.strip()
    if not rest:
        v = audio.get_volume()
        print(f"[vol] get → {v}")
        return
    try:
        n = int(rest)
    except ValueError:
        print(f"[vol] 参数解析失败, 用法: :vol N (N 是整数)")
        return
    code = audio.set_volume(n)
    print(f"[vol] set {n} → code={code}")


def _handle_health() -> None:
    h = audio.health()
    # 一行式紧凑打印, 方便扫.
    print(
        f"[health] running={h['running']}  app={h['app_name']}  "
        f"playing_est={h['is_playing_est']}  rem={h['estimated_remaining_sec']:.2f}s  "
        f"streams={h['stream_counter']}  tts={h['tts_call_count']}  "
        f"play={h['play_stream_call_count']}  stop={h['play_stop_call_count']}  "
        f"err={h['error_count']}  last={h['last_stream_id'][-12:] or '-'}"
    )


def _handle_status(rest: str) -> None:
    rest = rest.strip().lower()
    if rest == "on":
        _status_on()
    elif rest == "off":
        _status_off()
    else:
        print(f"[status] 用法: :status on | :status off")


def _dispatch(line: str) -> bool:
    """处理一行输入. 返回 False 表示请求退出."""
    line = line.strip()
    if not line:
        return True

    # 不以 ':' 开头 → 默认 TTS.
    if not line.startswith(":"):
        _handle_speak(line)
        return True

    # 命令.
    head, _, rest = line[1:].partition(" ")
    head = head.lower()

    if head in ("q", "quit"):
        return False
    elif head in ("c", "cancel"):
        audio.cancel()
        print("[cancel] PlayStop.")
    elif head in ("h", "health"):
        _handle_health()
    elif head == "help":
        print(_HELP)
    elif head == "sp":
        # :sp ID text...
        sub = rest.split(" ", 1)
        if len(sub) < 2:
            print("[sp] 用法: :sp ID <text>")
        else:
            try:
                sid = int(sub[0])
            except ValueError:
                print("[sp] ID 必须是整数")
            else:
                _handle_speak(sub[1], speaker_id=sid)
    elif head == "tone":
        _handle_tone(rest, resume=False)
    elif head == "tone+":
        _handle_tone(rest, resume=True)
    elif head == "wav":
        if not rest.strip():
            print("[wav] 用法: :wav <path>")
        else:
            _handle_wav(rest.strip())
    elif head in ("v", "vol", "volume"):
        _handle_vol(rest)
    elif head == "status":
        _handle_status(rest)
    else:
        print(f"[?] 未识别命令: :{head}. 输入 :help 看清单.")

    return True


# ── 主入口 ──────────────────────────────────────────────────────────────

def main(nic: str) -> int:
    print(f"[1/3] sdk.bootstrap(nic={nic!r}) ...")
    bootstrap(nic)

    print("[2/3] audio.start() ...")
    audio.start()

    print()
    print("=" * 72)
    print(" Audio runtime 双工体验. 一进 REPL, 把 G1 全部音频能力跑一圈.")
    print(" 安全提示: G1 喇叭可能比想象的响, 第一件事建议 `:vol 30` 之类调小.")
    print("=" * 72)
    print(_HELP)

    session: PromptSession = PromptSession()
    line_count = 0
    try:
        with patch_stdout.patch_stdout(raw=True):
            while True:
                try:
                    line = session.prompt(">>> ")
                except (KeyboardInterrupt, EOFError):
                    print()
                    break
                line_count += 1
                if not _dispatch(line):
                    break
    finally:
        print(f"\n[3/3] cleanup ...")
        if _status_thread is not None and _status_thread.is_alive():
            _status_off()
        audio.stop()
        print()
        print("=" * 72)
        print(f" 摘要: 处理 {line_count} 行输入. health={audio.health()}")
        print("=" * 72)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
