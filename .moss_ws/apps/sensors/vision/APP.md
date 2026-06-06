---
executable: uv
script: main.py
arguments: ''
description: 'Camera vision input — captures frames via cv2, detects faces, publishes face position to stream.'
respawn: false
workers: 1
---

Camera vision app — captures frames via OpenCV (`cv2.VideoCapture`), runs local Haar cascade face detection, and publishes quantized face coordinates to `vision/face` stream for cross-app consumption (e.g. ai_eye face tracking). Also provides `describe` and `detect_faces` Channel commands for Ghost to query the camera via VLM.
