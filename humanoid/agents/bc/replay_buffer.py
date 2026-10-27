from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import gymnasium as gym

import numpy as np
import torch

from humanoid.utils.array import DictArray, recursive_slice, to_tensor


@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    actions: torch.Tensor

    def __len__(self):
        return self.obs.size(0)


class ReplayBuffer:
    def __init__(
        self,
        single_obs_space: gym.spaces.Space,
        single_act_space: gym.spaces.Box,
        buffer_size: int,
        storage_device: torch.device,
        sample_device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False

        self.storage_device = storage_device
        self.sample_device = sample_device
        self._must_transfer_devices = storage_device != sample_device

        if isinstance(single_obs_space, gym.spaces.Dict):
            self.obs = DictArray(
                (buffer_size,),
                single_obs_space,
                device=storage_device,
            )
        else:
            self.obs = torch.zeros(
                (buffer_size,) + single_obs_space.shape,
                device=storage_device,
            )
        self.actions = torch.zeros(
            (buffer_size,) + single_act_space.shape,
            device=storage_device,
        )

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ):
        if self._must_transfer_devices:
            obs = to_tensor(obs, device=self.storage_device)
            action = to_tensor(action, device=self.storage_device)

        # Get number of new samples
        n_samples = action.shape[0]

        # Handle case where new samples would exceed buffer size
        if self.pos + n_samples > self.buffer_size:
            space_left = self.buffer_size - self.pos

            self.obs[self.pos : self.buffer_size] = recursive_slice(
                obs, slice(space_left)
            )
            self.obs[: n_samples - space_left] = recursive_slice(
                obs, slice(space_left, None)
            )

            self.actions[self.pos : self.buffer_size] = action[:space_left]
            self.actions[: n_samples - space_left] = action[space_left:]

            self.pos = n_samples - space_left
            self.full = True
        else:
            self.obs[self.pos : self.pos + n_samples] = obs
            self.actions[self.pos : self.pos + n_samples] = action

            self.pos += n_samples
            if self.pos == self.buffer_size:
                self.pos = 0
                self.full = True

    def sample(self, batch_size: int):
        batch_inds = torch.randint(
            0, self.buffer_size if self.full else self.pos, size=(batch_size,)
        )

        b_obs = self.obs[batch_inds]
        b_actions = self.actions[batch_inds]

        if self._must_transfer_devices:
            b_obs = to_tensor(b_obs, device=self.sample_device, dtype=torch.float)
            b_actions = to_tensor(
                b_actions, device=self.sample_device, dtype=torch.float
            )

        return ReplayBufferSample(
            obs=b_obs,
            actions=b_actions,
        )

    def reset(self):
        self.pos = 0
        self.full = False

    def __len__(self):
        return self.buffer_size if self.full else self.pos

    @cached_property
    def device(self):
        return self.storage_device
