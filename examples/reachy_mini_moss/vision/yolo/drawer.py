import cv2
import numpy as np
from supervision import Detections
from examples.reachy_mini_moss.vision.yolo.head_detector import HeadDetector


def draw_tracks(frame: np.ndarray, detections: Detections) -> np.ndarray:
    """在图像上绘制边界框和跟踪ID。"""
    if len(detections) == 0:
        return frame

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        track_id = detections.tracker_id[i] if detections.tracker_id is not None else -1

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制ID标签
        label = f"ID: {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def main():
    detector = HeadDetector(confidence_threshold=0.3, device='cpu')  # 如有GPU可设'cuda'
    cap = cv2.VideoCapture(0)  # 0 为默认摄像头，也可替换为视频文件路径

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 处理帧，获取跟踪结果
        tracked_detections, _ = detector.get_head_positions(frame)

        # 绘制结果
        frame = draw_tracks(frame, tracked_detections)

        # 显示
        cv2.imshow('Head Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    main()