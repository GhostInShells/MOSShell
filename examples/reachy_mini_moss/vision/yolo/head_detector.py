import logging
from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from supervision import Detections, ByteTrack
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from examples.reachy_mini_moss.vision.yolo.model import Position

logger = logging.getLogger(__name__)


class HeadDetector:
    """跟踪画面中所有人脸的位置，为每个人脸分配稳定ID。"""

    def __init__(
        self,
        model_repo: str = "AdamCodd/YOLOv11n-face-detection",
        model_filename: str = "model.pt",
        confidence_threshold: float = 0.3,
        device: str = "cpu",
        track_activation_threshold: float = 0.3,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ) -> None:
        self.confidence_threshold = confidence_threshold

        # 加载 YOLO 人脸检测模型
        try:
            model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
            self.model = YOLO(model_path).to(device)
            logger.info(f"YOLO face detection model loaded from {model_repo}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

        # 初始化跟踪器（ByteTrack）
        self.tracker = ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )

    def _bbox_to_mp_coords(self, bbox: NDArray[np.float32], w: int, h: int) -> NDArray[np.float32]:
        """将边界框中心转换为 [-1, 1] 范围的坐标（MediaPipe风格）。"""
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        norm_x = (center_x / w) * 2.0 - 1.0
        norm_y = (center_y / h) * 2.0 - 1.0
        return np.array([norm_x, norm_y], dtype=np.float32)

    def get_head_positions(self, img: NDArray[np.uint8]) -> Tuple[Detections, List[Position]]:
        """
        检测并跟踪所有人脸，返回每个被跟踪人脸的 (track_id, center_coord, roll_angle)。

        Args:
            img: 输入图像 (H, W, 3)

        Returns:
            列表，每个元素为 (track_id, center_coord, roll)，其中 center_coord 是 [-1,1] 坐标，
            roll 目前固定为 0.0（可扩展）。
        """
        h, w = img.shape[:2]

        try:
            # YOLO 推理
            results = self.model(img, verbose=False)
            # 转换为 supervision 的 Detections 格式
            detections = Detections.from_ultralytics(results[0])

            # 过滤低置信度检测
            if detections.confidence is not None:
                mask = detections.confidence >= self.confidence_threshold
                detections = detections[mask]

            # 如果没有检测到人脸，返回空列表
            if len(detections) == 0:
                logger.debug("No faces detected above confidence threshold")
                return Detections.empty(), []

            # 更新跟踪器，为每个检测分配或更新 track_id
            tracked_detections = self.tracker.update_with_detections(detections)

            # 构建结果列表
            positions: List[Position] = []
            for i in range(len(tracked_detections)):
                bbox = tracked_detections.xyxy[i]
                track_id = tracked_detections.tracker_id[i]  # 每个跟踪对象都有唯一ID
                center = self._bbox_to_mp_coords(bbox, w, h)
                # 如果需要，可以加入置信度等信息
                positions.append(Position.new(track_id, center))

            logger.debug(f"Tracked {len(positions)} faces: IDs {[p.track_id for p in positions]}")
            return tracked_detections, positions

        except Exception as e:
            logger.error(f"Error in head position tracking: {e}")
            return Detections.empty(), []
