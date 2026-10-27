from dataclasses import dataclass

from humanoid.envs.humanoidverse.config.catching.catch_upper.camera import CameraConfig
from humanoid.envs.humanoidverse.config.catching.catch_upper.domain_randomization import (
    DomainRandomizationConfig,
)
from humanoid.envs.humanoidverse.config.catching.catch_upper.env import CatchEnvConfig
from humanoid.envs.humanoidverse.config.catching.catch_upper.observation import (
    G1CatchUpperLockLegsObservationConfig,
    G1CatchUpperLockLegsObservationHistConfig,
)
from humanoid.envs.humanoidverse.config.catching.catch_upper.rewards import (
    G1CatchUpperLockLegsRewardsConfig,
)
from humanoid.envs.humanoidverse.config.catching.catch_upper.robot import (
    G1CatchUpperLockLegsRobotConfig,
)
from humanoid.envs.humanoidverse.config.catching.simulator import SimulatorConfig
from humanoid.utils.dataclasses import class_factory, resolve_templates


@dataclass
class G1CatchUpperLockLegsConfig:
    robot: G1CatchUpperLockLegsRobotConfig = class_factory(
        G1CatchUpperLockLegsRobotConfig
    )
    domain_rand: DomainRandomizationConfig = class_factory(DomainRandomizationConfig)
    rewards: G1CatchUpperLockLegsRewardsConfig = class_factory(
        G1CatchUpperLockLegsRewardsConfig
    )
    obs: G1CatchUpperLockLegsObservationConfig = class_factory(
        G1CatchUpperLockLegsObservationConfig
    )
    simulator: SimulatorConfig = class_factory(SimulatorConfig)
    env: CatchEnvConfig = class_factory(CatchEnvConfig)
    camera: CameraConfig = class_factory(CameraConfig)

    def __post_init__(self):
        # resolve any "${...}" strings
        resolve_templates(self, self)

        obs_dim_dict = dict()
        _obs_key_list = self.obs.obs_dict
        _aux_obs_key_list = self.obs.obs_auxiliary

        assert set(self.obs.noise_scales.keys()) == set(self.obs.obs_scales.keys())

        # convert obs_dims to dict if it's not already
        if isinstance(self.obs.obs_dims, list):
            each_dict_obs_dims = {k: v for d in self.obs.obs_dims for k, v in d.items()}
            self.obs.obs_dims = each_dict_obs_dims
        else:
            each_dict_obs_dims = self.obs.obs_dims

        auxiliary_obs_dims = {}
        for aux_obs_key, aux_config in _aux_obs_key_list.__dict__.items():
            auxiliary_obs_dims[aux_obs_key] = 0
            for _key, _num in aux_config.items():
                assert _key in self.obs.obs_dims.keys()
                auxiliary_obs_dims[aux_obs_key] += self.obs.obs_dims[_key] * _num

        for obs_key, obs_config in _obs_key_list.__dict__.items():
            obs_dim_dict[obs_key] = 0
            for key in obs_config:
                if key.endswith("_raw"):
                    key = key[:-4]
                if key in self.obs.obs_dims.keys():
                    obs_dim_dict[obs_key] += self.obs.obs_dims[key]
                else:
                    obs_dim_dict[obs_key] += auxiliary_obs_dims[key]

        self.robot.algo_obs_dim_dict = obs_dim_dict
        self.robot.policy_obs_dim = obs_dim_dict["actor_obs"]
        self.robot.critic_obs_dim = obs_dim_dict["critic_obs"]
        self.robot.teacher_obs_dim = obs_dim_dict["teacher_obs"]


@dataclass
class G1CatchUpperLockLegsHistConfig(G1CatchUpperLockLegsConfig):
    obs: G1CatchUpperLockLegsObservationHistConfig = class_factory(
        G1CatchUpperLockLegsObservationHistConfig
    )
