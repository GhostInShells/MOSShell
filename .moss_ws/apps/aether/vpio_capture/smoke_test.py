"""Smoke test — verify PyObjC + AVAudioEngine + VPIO APIs work on this Mac.

Doesn't start the full MOSS pipeline. Just exercises the API surface
to catch PyObjC binding errors early.
"""
import sys
import time

import numpy as np


def main() -> int:
    print("[1] importing AVFoundation...")
    try:
        import AVFoundation
        import AVFAudio
        from AVFoundation import AVAudioEngine, AVAudioConverter
        print("    OK · AVFoundation imported")
    except ImportError as e:
        print(f"    FAIL · {e}")
        print("    Install with: uv pip install pyobjc-framework-AVFoundation")
        return 1

    print("[2] creating AVAudioEngine...")
    engine = AVAudioEngine.new()
    input_node = engine.inputNode()
    output_node = engine.outputNode()
    print(f"    OK · input={type(input_node).__name__}, output={type(output_node).__name__}")

    print("[3] reading native input format...")
    tap_format = input_node.outputFormatForBus_(0)
    sr = int(tap_format.sampleRate())
    ch = int(tap_format.channelCount())
    cf = tap_format.commonFormat()
    print(f"    OK · native_sr={sr}, ch={ch}, commonFormat={cf}")
    if sr not in (44100, 48000):
        print(f"    WARN · VPIO expects 48k or 44.1k, got {sr}")

    # Selector is `setVoiceProcessingEnabled:error:` → PyObjC maps to
    # setVoiceProcessingEnabled_error_(value, error_ptr) → returns BOOL.
    # The error_ptr arg is consumed by PyObjC and surfaced via the BOOL + thrown exc.
    print("[4] enabling VPIO on input node (setVoiceProcessingEnabled:error:)...")
    try:
        from Foundation import NSError
        err_ptr = None
        ok = input_node.setVoiceProcessingEnabled_error_(True, err_ptr)
        # PyObjC: returns (BOOL, NSError) tuple when error out-param present
        if isinstance(ok, tuple):
            ok, err = ok
        else:
            err = None
        if not ok:
            print(f"    FAIL · returned ok=False, error={err and err.localizedDescription()}")
            return 2
        print(f"    OK · input VPIO enabled = {input_node.isVoiceProcessingEnabled()}")
    except Exception as e:
        print(f"    FAIL · {e}")
        return 2

    print("[5] enabling VPIO on output node (required for AEC)...")
    try:
        err_ptr = None
        ok = output_node.setVoiceProcessingEnabled_error_(True, err_ptr)
        if isinstance(ok, tuple):
            ok, err = ok
        else:
            err = None
        if not ok:
            print(f"    FAIL · returned ok=False, error={err and err.localizedDescription()}")
            return 3
        print(f"    OK · output VPIO enabled = {output_node.isVoiceProcessingEnabled()}")
    except Exception as e:
        print(f"    FAIL · {e}")
        return 3

    print("[6] building AVAudioConverter 48k → 16k...")
    # AVAudioCommonFormat enum (NSUInteger):
    #   0=OtherFormat, 1=PCMFormatFloat32, 2=PCMFormatFloat64,
    #   3=PCMFormatInt16, 4=PCMFormatInt32
    PCM_FORMAT_FLOAT32 = 1
    out_format = AVFAudio.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
        PCM_FORMAT_FLOAT32,
        16000.0,
        1,
        False,
    )
    if out_format is None:
        print("    FAIL · could not create 16kHz output format")
        return 4
    converter = AVAudioConverter.alloc().initFromFormat_toFormat_(tap_format, out_format)
    if converter is None:
        print("    FAIL · could not create AVAudioConverter")
        return 5
    print(f"    OK · converter={converter}")

    print("[7] installing tap on bus 0 for 2 seconds...")
    frames_received = [0]

    def _tap(buffer, when):
        try:
            n = int(buffer.frameLength())
            if n > 0:
                frames_received[0] += n
        except Exception:
            pass

    buf_size = int(sr * 0.05)  # 50ms frames
    input_node.installTapOnBus_bufferSize_format_block_(
        0, buf_size, tap_format, _tap,
    )

    try:
        engine.prepare()
        # PyObjC: startAndReturnError_ returns (BOOL success, NSError* error)
        ok, err = engine.startAndReturnError_(None)
        if not ok:
            print(f"    FAIL · engine.start returned error: {err and err.localizedDescription()}")
            return 6
        print("    OK · engine started, capturing for 2s...")
        time.sleep(2.0)
    finally:
        try:
            input_node.removeTapOnBus_(0)
        except Exception:
            pass
        engine.stop()
        engine.reset()

    print(f"    captured {frames_received[0]} native frames in 2s "
          f"(expected ~{sr * 2} = {sr * 2})")

    print()
    print("=" * 50)
    print("SMOKE TEST PASSED — VPIO + AVAudioEngine APIs work on this Mac.")
    print("Ready to run the full VPIOCaptureSource via MOSS App.")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
