from typing import Any, Dict, List

import torch


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def parse_observation(
    cls: Any,
    key_list: List,
    buf_dict: Dict,
    obs_scales: Dict,
    noise_scales: Dict,
    noise_level: float = 1.0,
) -> None:
    """Parse observations for the legged_robot_base class"""

    for obs_key in key_list:
        if obs_key.endswith("_raw"):
            base_obs_key = obs_key[:-4]
            obs_noise = None
        else:
            base_obs_key = obs_key
            obs_noise = noise_scales[obs_key] * noise_level

        base_obs_key_raw = base_obs_key + "_raw"
        if base_obs_key_raw in buf_dict:
            scaled_actor_obs = buf_dict[base_obs_key_raw]
        else:
            if "history" in base_obs_key:
                actor_obs = cls._get_obs_history(obs_auxiliary_key=base_obs_key).clone()
            else:
                actor_obs = getattr(cls, f"_get_obs_{base_obs_key}")().clone()
            obs_scale = obs_scales[base_obs_key]
            scaled_actor_obs = actor_obs * obs_scale

        buf_dict[base_obs_key_raw] = scaled_actor_obs
        if obs_noise is not None:
            # use scaled_actor_obs as the shape template so noise works whether we
            # loaded a cached raw obs or just computed actor_obs above
            buf_dict[obs_key] = (
                scaled_actor_obs
                + (torch.randn_like(scaled_actor_obs) * 2.0 - 1.0) * obs_noise
            )
