from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class EnvConfig:
    suite: str
    """Suite of environments to use"""

    env_id: str
    """Environment ID"""
    num_envs: int
    """Number of environments to create"""
    max_episode_steps: int
    """Maximum number of steps per episode"""

    sim_backend: Literal["cpu", "gpu"] = "gpu"
    """Sim backend (cpu/gpu)"""

    reward_mode: Literal["dense", "normalized_dense", "sparse"] = "dense"
    """Reward mode"""

    render_mode: Literal["rgb_array", "human", None] = "rgb_array"
    """Render mode"""
    save_video: bool = False
    """Whether to save video"""
    info_on_video: bool = False
    """Whether to save info on video"""
    chunk_infos: Optional[List[str]] = None
    """If info_on_video, can optionally chunk infos to wrap onto new line"""

    options_config: dict = field(default_factory=dict)
    """Options config"""

    def __post_init__(self):
        if self.render_mode == "human":
            assert not self.save_video, "Human mode does not support saving video"
