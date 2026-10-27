import math

import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, single_obs_space, single_act_space, gaussian=False):
        super().__init__()

        self.mlp = nn.Sequential(
            layer_init(nn.Linear(math.prod(single_obs_space.shape), 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(inplace=True),
            layer_init(
                nn.Linear(128, math.prod(single_act_space.shape)),
                std=0.01 * math.sqrt(2),
            ),
        )
        # self.gaussian = gaussian
        # if self.gaussian:
        #     self.log_std = nn.Parameter(
        #         torch.ones(1, math.prod(single_act_space.shape)) * -0.5
        #     )

    def forward(self, observations, deterministic: bool = True) -> torch.Tensor:
        return self.mlp(observations)

    # def get_action(self, x, deterministic=True):
    #     if not self.gaussian:
    #         assert deterministic
    #     action_mean = self.mlp(x)
    #     if deterministic:
    #         return action_mean
    #     action_logstd = self.log_std.expand_as(action_mean)
    #     probs = Normal(action_mean, action_logstd.exp())
    #     return probs.sample()

    # def get_distribution(self, x):
    #     assert self.gaussian
    #     action_mean = self.mlp(x)
    #     action_logstd = self.log_std.expand_as(action_mean)
    #     return Normal(action_mean, action_logstd.exp())
