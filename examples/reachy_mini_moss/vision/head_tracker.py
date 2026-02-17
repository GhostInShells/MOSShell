import asyncio
from typing import Tuple, List

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from examples.reachy_mini_moss.vision.camera_worker import CameraWorker
from examples.reachy_mini_moss.vision.yolo.head_detector import HeadDetector
from examples.reachy_mini_moss.vision.yolo.model import Position


class HeadTracker:

    def __init__(self, mini: ReachyMini):
        self._mini = mini
        self._camera_worker = CameraWorker(mini, HeadDetector())

        self.face_tracking_offsets: List[float] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self.face_tracking_positions: List[Position] = []
        self.current_tracking_id = -1
        self._run_task = None

        self.enabled = asyncio.Event()
        self._quit = asyncio.Event()

    def set_tracking_id(self, tracking_id: int):
        self._camera_worker.set_tracking_id(tracking_id)

    async def run(self):
        while not self._quit.is_set():
            elapsed = 0.03
            await asyncio.sleep(elapsed)
            if not self.enabled.is_set():
                continue

            self.face_tracking_offsets, self.face_tracking_positions, self.current_tracking_id = self._camera_worker.get_face_tracking_data()

            new_pose = create_head_pose(
                x=self.face_tracking_offsets[0],
                y=self.face_tracking_offsets[1],
                z=self.face_tracking_offsets[2],
                roll=self.face_tracking_offsets[3],
                pitch=self.face_tracking_offsets[4],
                yaw=self.face_tracking_offsets[5],
                degrees=False,
                mm=False,
            )
            self._mini.set_target(head=new_pose)

    async def start(self):
        self._camera_worker.start()
        self._run_task = asyncio.create_task(self.run())

    async def stop(self):
        self._camera_worker.set_head_tracking_enabled(False)
        self.enabled.clear()
        self._quit.set()
        await self._run_task