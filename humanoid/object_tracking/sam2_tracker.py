from typing import Tuple

import numpy as np
import torch

from sam2.build_sam import build_sam2_camera_predictor
from sam2.sam2_camera_predictor import SAM2CameraPredictor

from humanoid import HUMANOID_RESOURCES_DIR


class SAM2ObjTracker:
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        checkpoint=str(HUMANOID_RESOURCES_DIR / "models/sam2.1_hiera_tiny.pt"),
        model_cfg="configs/sam2.1/sam2.1_hiera_t_512.yaml",
        device="cuda",
    ):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.predictor: SAM2CameraPredictor = build_sam2_camera_predictor(
            model_cfg, checkpoint
        )
        self.device = device
        self.initialized = False

    def mask_to_obj_pred(
        self, mask: torch.Tensor, depth: torch.Tensor, obj_width: float = 0.0
    ) -> np.ndarray:
        if isinstance(obj_width, torch.Tensor):
            obj_width = obj_width.item()
        if not mask.any():
            return None

        yx_pts = torch.nonzero(mask)
        yx_mean = yx_pts.float().mean(dim=0).cpu().numpy()

        H, W = depth.shape
        u = np.clip(round(yx_mean[1]), 0, W - 1)
        v = np.clip(round(yx_mean[0]), 0, H - 1)

        z = depth[v, u].item()
        if z <= 0 or np.isnan(z):
            return None

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        p_surf = np.array([x, y, z], dtype=np.float32)

        if obj_width != 0.0:
            n = np.linalg.norm(p_surf)
            if n <= 1e-9 or np.isnan(n):
                return None
            u_dir = p_surf / n
            p_center = p_surf + u_dir * obj_width
            return p_center

        return p_surf

    def init_predictor(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        click_point: Tuple[int, int],
        obj_width: float = 0.0,
    ) -> np.ndarray:
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.predictor.load_first_frame(rgb)
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=0,
                points=np.array([list(click_point)], dtype=np.float32),
                labels=np.array([[1]], dtype=np.int32),
            )
            mask = out_mask_logits[0][0] > 0.0
            pos = self.mask_to_obj_pred(mask, depth, obj_width)
            self.initialized = True
            return pos

    def predict_pos(
        self, rgb: torch.Tensor, depth: torch.Tensor, obj_width: float = 0.0
    ) -> np.ndarray:
        assert self.initialized
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            out_obj_ids, out_mask_logits = self.predictor.track(rgb)
            mask = out_mask_logits[0][0] > 0.0
            return self.mask_to_obj_pred(mask, depth, obj_width)

    def reset(self) -> None:
        if self.initialized:
            self.predictor.reset_state()
        self.initialized = False
