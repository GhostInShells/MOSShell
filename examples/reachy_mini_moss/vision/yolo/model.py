from typing import List
import numpy as np
from numpy.typing import NDArray

class Position:
    track_id: int
    center: NDArray[np.float32]

    @classmethod
    def new(cls, track_id: int, center: NDArray[np.float32]) -> "Position":
        self = cls()
        self.track_id = track_id
        self.center = center
        return self

def get_position_by_track_id(positions: List[Position], track_id: int) -> Position | None:
    for p in positions:
        if p.track_id == track_id:
            return p
    return None


def stringify_positions(positions: List[Position]) -> str:
    lines = []
    for p in positions:
        lines.append(f"track_id={p.track_id}, mediapipe_coordinate={p.center}")
    return "\n".join(lines)