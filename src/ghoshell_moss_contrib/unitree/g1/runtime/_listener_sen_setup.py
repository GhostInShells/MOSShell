"""
_listener_sen_setup — 蓝牙耳机 listener 的配置生成探针.

场景:
  你戴好蓝牙耳机, 跑这个脚本. 它列出系统所有 capture 设备, 高亮蓝牙类候选,
  让你选一个; 然后用 miniaudio 实测开启该设备, 实时刷新 RMS / 静音/ 实际采样率;
  你对着耳机说几句, 看 RMS 跳起来确认 mic 真的通; Ctrl+C 结束时, 询问是否
  把配置写入 ~/.moss_g1_listener.json (如已存在会先 diff).

  这个脚本回答了 listener 启动前的所有人为决策, 不读环境变量, 跨平台.

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._listener_sen_setup
  python -m ghoshell_moss_contrib.unitree.g1.runtime._listener_sen_setup --pattern AirPods

  --pattern: 跳过设备选择, 直接用首个匹配的设备 (大小写不敏感 substring).
  --asr-sample-rate: 写入配置的 ASR 目标采样率, 默认 16000 (火山引擎要求).
  --frame-ms: capture buffer 时长, 默认 50ms.
  --no-save: 测完不询问写入, 只显示推荐配置. 用于调试.

前置:
  - 蓝牙耳机已连 (mac: Settings → Bluetooth; linux: bluetoothctl connect; PC2: 见硬件文档)
  - 耳机已成为系统 input device (mac 上一般自动; linux 上可能要切 HFP profile,
    `pactl set-card-profile bluez_card.XX_XX_XX handsfree_head_unit`)
  - .venv 装好了 miniaudio (uv sync --all-extras 已含)

预期:
  跑起来后:
    [devices] 共 3 个 capture 设备:
      [0] MacBook Pro Microphone   (default)
      [1] AirPods Pro (Hands-Free) ★ bluetooth-like
      [2] BlackHole 2ch

    选择设备 (回车确认 [1]):
    [probe] 启动 capture: AirPods Pro (Hands-Free)
            请求采样率 16000 Hz → 实际 16000 Hz, 1ch
            对着耳机说话, 看 RMS 跳起来 (静音 < -50dB, 说话 > -30dB)

    [0.5s] chunks=10  rms=-48dB  silent=True
    [1.0s] chunks=20  rms=-26dB  silent=False     ← 说话时跳起来
    ...

    ^C
    [summary]
      device: AirPods Pro (Hands-Free)
      sample_rate (实测): 16000
      channels: 1
      voiced 帧占比: 38% (说话过 → 设备真的通)

    [diff vs ~/.moss_g1_listener.json]
      device_pattern  : "AirPods Pro"   (was "old-headset")
      sample_rate     : 16000           (unchanged)

    写入 ~/.moss_g1_listener.json? [y/N]:

读完 docstring 还看不懂请回去读 runtime/README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

# ── 配置文件位置 (跟 listener.py 同一常量) ────────────────────────────────
DEFAULT_CONFIG_PATH = Path.home() / ".moss_g1_listener.json"

# 蓝牙类设备名关键字, 用于高亮
_BT_HINT_KEYWORDS = (
    "bluetooth", "bt", "airpods", "wireless", "hands-free", "hands free",
    "headset", "buds", "beats", "sony wf", "sony wh",
)

_SILENCE_THRESHOLD_DB = -50.0


def _is_bt_like(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in _BT_HINT_KEYWORDS)


def _dev_name(d) -> str:
    """兼容 dict / 老对象两种 device 形态."""
    if isinstance(d, dict):
        return d.get("name", "")
    return getattr(d, "name", "")


def _dev_id(d):
    if isinstance(d, dict):
        return d.get("id")
    return getattr(d, "id", None)


def _enumerate_devices() -> list:
    """列设备. 本仓库装的 miniaudio 返回 list[dict], 老版本是带 .name .id 的对象."""
    import miniaudio
    devs_obj = miniaudio.Devices()
    if hasattr(devs_obj, "get_captures"):
        return list(devs_obj.get_captures())
    return list(devs_obj.capture)  # 老 API fallback


def _print_devices(devs: list) -> None:
    print(f"\n[devices] 共 {len(devs)} 个 capture 设备:")
    for i, d in enumerate(devs):
        name = _dev_name(d)
        mark = " ★ bluetooth-like" if _is_bt_like(name) else ""
        print(f"  [{i}] {name}{mark}")
    print()


def _pick_default_index(devs: list) -> int:
    """没 --pattern 时, 默认推荐: 首个 bluetooth-like; 没有则 0."""
    for i, d in enumerate(devs):
        if _is_bt_like(_dev_name(d)):
            return i
    return 0


def _pick_by_pattern(devs: list, pattern: str) -> Optional[int]:
    pat = pattern.lower()
    for i, d in enumerate(devs):
        if pat in _dev_name(d).lower():
            return i
    return None


def _prompt_index(devs: list, default: int) -> int:
    """提示用户选择. 回车 = default. 输入数字 = 该下标."""
    while True:
        try:
            raw = input(f"选择设备 (回车确认 [{default}]): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise
        if not raw:
            return default
        try:
            idx = int(raw)
            if 0 <= idx < len(devs):
                return idx
        except ValueError:
            pass
        print(f"无效输入, 请输入 0..{len(devs)-1} 或直接回车")


# ── 实测 capture ──────────────────────────────────────────────────────────

class _ProbeState:
    """capture 探测过程中的累计状态. 单写 (audio 线程) 多读 (主线程).

    RMS 等数值用 list 包一层避免锁 — Python 写引用是原子的, 累计计数偶尔少 1 无影响.
    """

    def __init__(self) -> None:
        self.chunks_total = 0
        self.chunks_voiced = 0
        self.rms_recent: deque[float] = deque(maxlen=20)  # ~1 秒
        self.last_pcm_at: float = 0.0
        self.actual_sample_rate: Optional[int] = None
        self.last_err: Optional[str] = None
        self.lock = threading.Lock()  # 保护 deque 操作


def _make_probe_generator(state: _ProbeState, channels: int):
    """miniaudio 喂帧的 generator. 跑在 audio 线程内."""
    def _gen():
        while True:
            data = yield
            try:
                samples = np.frombuffer(data, dtype=np.int16)
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
                f32 = samples.astype(np.float64) / 32768.0
                rms = float(np.sqrt(np.mean(f32 ** 2)))
                rms_db = 20.0 * np.log10(max(rms, 1e-10))
                voiced = rms_db >= _SILENCE_THRESHOLD_DB

                with state.lock:
                    state.chunks_total += 1
                    if voiced:
                        state.chunks_voiced += 1
                    state.rms_recent.append(rms_db)
                    state.last_pcm_at = time.time()
            except Exception as e:
                state.last_err = str(e)
    return _gen()


def _live_print_loop(state: _ProbeState, started_at: float) -> None:
    """每 0.5s 在主线程刷新一行 RMS 摘要. 主线程退出条件: KeyboardInterrupt."""
    print("\n[live] 对着耳机说话, 看 RMS 跳起来. Ctrl+C 停止并询问是否保存.\n")
    last_lines_printed = 0
    try:
        while True:
            time.sleep(0.5)
            with state.lock:
                total = state.chunks_total
                voiced = state.chunks_voiced
                rms_now = state.rms_recent[-1] if state.rms_recent else -96.0
                rms_avg = sum(state.rms_recent) / len(state.rms_recent) if state.rms_recent else -96.0
                last_pcm_at = state.last_pcm_at
            elapsed = time.time() - started_at
            silence_age = (time.time() - last_pcm_at) if last_pcm_at else float("inf")
            warn = ""
            if silence_age > 2.0 and last_pcm_at > 0:
                warn = f"  [!] 已 {silence_age:.1f}s 无新数据 (蓝牙断? mic 静音?)"
            elif last_pcm_at == 0 and elapsed > 2.0:
                warn = "  [!] 启动 2s 仍无任何数据, 设备可能不可用"
            line = (
                f"[{elapsed:5.1f}s] chunks={total:4d}  "
                f"rms_now={rms_now:6.1f}dB  rms_avg={rms_avg:6.1f}dB  "
                f"voiced={voiced}/{total}"
                f"{warn}"
            )
            print(line)
            last_lines_printed += 1
    except KeyboardInterrupt:
        print()


# ── 配置写入 ──────────────────────────────────────────────────────────────

def _make_config_dict(
    *,
    device_name: str,
    device_pattern: str,
    capture_sr: int,
    asr_sr: int,
    channels: int,
    frame_ms: int,
) -> dict:
    return {
        "device_pattern": device_pattern,
        "device_name_resolved": device_name,
        "sample_rate": capture_sr,
        "channels": channels,
        "frame_ms": frame_ms,
        "asr_sample_rate": asr_sr,
        "end_window_ms": 800,
        "_note": (
            "Generated by _listener_sen_setup.py at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}. "
            "Edit by hand if needed. device_pattern 是 substring (小写) 匹配, "
            "蓝牙重连后设备名后缀有时会变 (e.g. 'Hands-Free' vs 'Hands Free (HFP)'), "
            "pattern 取稳定部分 (e.g. 'AirPods')."
        ),
    }


def _suggest_pattern(device_name: str) -> str:
    """从完整设备名抽稳定 substring. 'AirPods Pro (Hands-Free)' → 'AirPods Pro'."""
    name = device_name
    # 去括号内容
    if "(" in name:
        name = name.split("(", 1)[0].strip()
    # 去常见后缀
    for suffix in (" Hands-Free", " Hands Free", " HFP", " A2DP", " Microphone", " Mic"):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name or device_name


def _diff_against_existing(path: Path, new_cfg: dict) -> Optional[str]:
    """返回 diff 文本; 不存在或无差异返回 None."""
    if not path.exists():
        return None
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "(无法解析现有文件)"
    keys = sorted(set(old.keys()) | set(new_cfg.keys()) - {"_note", "device_name_resolved"})
    lines = []
    for k in keys:
        if k.startswith("_") or k == "device_name_resolved":
            continue
        ov = old.get(k, "<missing>")
        nv = new_cfg.get(k, "<missing>")
        if ov == nv:
            lines.append(f"  {k:20s} {nv}    (unchanged)")
        else:
            lines.append(f"  {k:20s} {nv!r}    (was {ov!r})")
    return "\n".join(lines) if lines else None


def _ask_save(path: Path, cfg: dict) -> bool:
    diff = _diff_against_existing(path, cfg)
    if diff:
        print(f"\n[diff vs {path}]\n{diff}\n")
    else:
        if path.exists():
            print(f"\n[note] {path} 无差异 (但仍可覆盖更新 _note 时间戳).\n")
        else:
            print(f"\n[note] {path} 尚不存在, 将创建.\n")
        print("[recommended config]")
        for k, v in cfg.items():
            if k.startswith("_"):
                continue
            print(f"  {k:20s} {v!r}")
        print()
    try:
        raw = input(f"写入 {path}? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return raw in ("y", "yes")


def _write_config(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n✓ written: {path}")


# ── main ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(prog="listener_sen_setup", description=__doc__.splitlines()[0])
    ap.add_argument("--pattern", help="跳过设备选择, 用首个匹配的设备")
    ap.add_argument("--asr-sample-rate", type=int, default=16000)
    ap.add_argument("--frame-ms", type=int, default=50)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--no-save", action="store_true", help="只测不询问写入")
    ap.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args()

    try:
        import miniaudio  # noqa: F401
    except ImportError:
        print("miniaudio 未安装. 运行: uv sync --all-extras", file=sys.stderr)
        return 2

    devs = _enumerate_devices()
    if not devs:
        print("[error] 系统没有 capture 设备. 检查蓝牙连接和系统设置.", file=sys.stderr)
        return 1

    _print_devices(devs)

    # 选设备
    if args.pattern:
        idx = _pick_by_pattern(devs, args.pattern)
        if idx is None:
            print(f"[error] --pattern '{args.pattern}' 无匹配设备", file=sys.stderr)
            return 1
        print(f"[--pattern] 使用 [{idx}] {_dev_name(devs[idx])}\n")
    else:
        default_idx = _pick_default_index(devs)
        try:
            idx = _prompt_index(devs, default_idx)
        except KeyboardInterrupt:
            return 130

    chosen = devs[idx]
    chosen_name = _dev_name(chosen)
    chosen_id = _dev_id(chosen)
    print(f"[probe] 启动 capture: {chosen_name!r}")

    # 启动 capture
    import miniaudio
    requested_sr = args.asr_sample_rate
    try:
        capture = miniaudio.CaptureDevice(
            input_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=args.channels,
            sample_rate=requested_sr,
            buffersize_msec=args.frame_ms,
            device_id=chosen_id,
        )
    except Exception as e:
        print(f"[error] CaptureDevice 初始化失败: {e}", file=sys.stderr)
        print("        蓝牙耳机可能未切到 HFP profile, 或被独占.", file=sys.stderr)
        return 1

    actual_sr = getattr(capture, "sample_rate", requested_sr)
    actual_ch = getattr(capture, "nchannels", args.channels)
    print(
        f"        请求采样率 {requested_sr} Hz → 实际 {actual_sr} Hz, {actual_ch}ch"
    )
    if actual_sr != requested_sr:
        print(
            f"        ⚠  实际采样率不同于请求. listener 启动后会自动重采样到 "
            f"{args.asr_sample_rate} Hz 送给 ASR."
        )

    state = _ProbeState()
    state.actual_sample_rate = actual_sr
    gen = _make_probe_generator(state, channels=actual_ch)
    next(gen)

    started_at = time.time()
    try:
        capture.start(gen)
    except Exception as e:
        print(f"[error] capture.start 失败: {e}", file=sys.stderr)
        capture.close()
        return 1

    try:
        _live_print_loop(state, started_at)
    finally:
        try:
            capture.stop()
        except Exception:
            pass
        try:
            capture.close()
        except Exception:
            pass

    # 摘要
    with state.lock:
        total = state.chunks_total
        voiced = state.chunks_voiced
        last_pcm = state.last_pcm_at
    voiced_pct = (voiced / total * 100) if total else 0.0
    print("\n[summary]")
    print(f"  device           : {chosen_name}")
    print(f"  sample_rate (实测): {actual_sr}")
    print(f"  channels         : {actual_ch}")
    print(f"  chunks 总数      : {total}")
    print(f"  voiced 帧占比    : {voiced_pct:.1f}%")
    if total == 0:
        print("  [!] 一帧没收到 — 设备不可用, 不建议保存.")
    elif voiced_pct < 5:
        print("  [!] voiced 占比很低 — 可能你没说话, 或 mic 没通. 建议重试.")
    elif last_pcm > 0 and time.time() - last_pcm > 5:
        print("  [!] 最近 5s 无数据 — 蓝牙可能中途断了.")

    # 写配置
    if args.no_save:
        return 0
    if total == 0:
        print("\n跳过保存 (一帧没收到).")
        return 1

    pattern = args.pattern or _suggest_pattern(chosen_name)
    cfg = _make_config_dict(
        device_name=chosen_name,
        device_pattern=pattern,
        capture_sr=actual_sr,
        asr_sr=args.asr_sample_rate,
        channels=actual_ch,
        frame_ms=args.frame_ms,
    )
    path = Path(args.config_path).expanduser()
    if _ask_save(path, cfg):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_config(path, cfg)
        except Exception as e:
            print(f"[error] 写入失败: {e}", file=sys.stderr)
            return 1
    else:
        print("\n未写入. 推荐配置见上, 你可以手动 copy 进文件.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
