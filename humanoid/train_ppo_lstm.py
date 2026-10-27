# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# Modifications made by Arth Shukla on [date]

import sys


if sys.version_info < (3, 9):
    # NOTE (arth): import first to avoid importing after torch:
    import isaacgym

from dataclasses import dataclass
from typing import List, Literal

from dacite import from_dict
from omegaconf import OmegaConf

from humanoid.train_ppo import PPOConfig
from humanoid.train_ppo import TrainConfig as PPOTrainConfig
from humanoid.train_ppo import train
from humanoid.utils.config import parse_cfg
from humanoid.utils.dataclasses import class_factory, default_field, recursive_asdict


@dataclass
class PPOLSTMConfig(PPOConfig):
    name: str = "ppo_lstm"

    actor_critic_cls: str = (
        "humanoid.agents.ppo.modules.actor_critic_recurrent.ActorCriticRecurrent"
    )

    total_timesteps: int = 1_000_000_000
    """total timesteps of the experiments"""

    @dataclass
    class PPOLSTMPolicyConfig(PPOConfig.PPOPolicyConfig):
        init_noise_std: float = 0.8
        actor_hidden_dims: List[int] = default_field([256, 256, 256])
        critic_hidden_dims: List[int] = default_field([256, 256, 256])
        # only for 'ActorCriticRecurrent':
        rnn_type: Literal["lstm"] = "lstm"
        rnn_hidden_size: int = 256
        rnn_num_layers: int = 1

    policy: PPOLSTMPolicyConfig = class_factory(PPOLSTMPolicyConfig)

    @dataclass
    class PPOLSTMAlgorithmConfig(PPOConfig.PPOAlgorithmConfig):
        entropy_coef = 0.01

    algorithm: PPOLSTMAlgorithmConfig = class_factory(PPOLSTMAlgorithmConfig)

    def __post_init__(self):
        assert self.name == "ppo_lstm", "Wrong algo config"


@dataclass
class TrainConfig(PPOTrainConfig):
    algo: PPOLSTMConfig


def get_humanoid_train_cfg(cfg: TrainConfig) -> TrainConfig:
    return from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_humanoid_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    train(cfg)
