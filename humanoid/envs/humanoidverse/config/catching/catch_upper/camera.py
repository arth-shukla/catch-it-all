from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CameraConfig:
    attach_body: Optional[str] = "torso_link"
    # local_rpy_axis: Tuple[int, int, int] = (0, 0, 1)
    # local_rpy_deg: float = -90.0
    # local_pos: Tuple[float, float, float] = (1.0, 1.5, 0.3)
    # calibrated camera pose relative to attach_body
    local_cam_quat: Tuple[float, float, float, float] = (
        0.008412109699131998,
        0.24465285498855066,
        0.012250686868740713,
        0.9694968476615098,
    )
    local_cam_pos: Tuple[float, float, float] = (
        0.007858286155598405,
        -0.008423433277233616,
        0.6791322641963928,
    )

    width: int = 376
    height: int = 672

    fx: float = 259.41357421875
    fy: float = 259.41357421875
    cx: float = 185.2420196533203
    cy: float = 335.4513854980469
    horizontal_fov: float = 71.86466979980469
    vertical_fov: float = 104.66220092773438

    edge_margin_px: int = 24

    noise_clear: float = 0.05
    noise_edge: float = 0.075
    noise_near_hand: float = 0.05

    noise_distance_factor: float = 0.0
    noise_velocity_factor: float = 0.005

    no_pred_prob: float = 0.05
