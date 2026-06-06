# MOSS App

## 关键文件

- `APP.md` — App 元信息声明
- `main.py` — 入口脚本。获取 AudioCaptureSource，打开 miniaudio 设备，PCM → Zenoh
- `CLAUDE.md` — 这个文件

## 运行时

```bash
moss apps test sensors/audio_capture
```

## 消费者

- waveform: `moss apps test sensors/waveform`
- listener (ASR, 后续 feature)
