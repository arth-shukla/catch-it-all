from functools import partial
from pathlib import Path

import torch


def get_actor_from_ckpt(ckpt_path, device=torch.device("cpu")):
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy_type = ckpt["policy_type"]

    if policy_type == "ppo":
        from humanoid.agents.ppo.modules.actor_critic_recurrent import ActorCritic
        from humanoid.utils.array import recursive_to_python

        actor_critic = ActorCritic(**recursive_to_python(ckpt["actor_critic_kwargs"]))
        actor_critic.load_state_dict(ckpt["agent"])
        actor_critic.to(device)
        actor_critic.eval()
        actor_critic.requires_grad_(False)

        return actor_critic
    elif policy_type == "ppo_lstm":
        from humanoid.agents.ppo.modules.actor_critic_recurrent import (
            ActorCriticRecurrent,
        )
        from humanoid.utils.array import recursive_to_python

        actor_critic = ActorCriticRecurrent(
            **recursive_to_python(ckpt["actor_critic_kwargs"])
        )
        actor_critic.load_state_dict(ckpt["agent"])
        actor_critic.to(device)
        actor_critic.eval()
        actor_critic.requires_grad_(False)

        return actor_critic
    elif policy_type == "dagger":
        from humanoid.agents.bc import Agent

        agent = Agent(
            ckpt["obs_space"], ckpt["action_space"], ckpt.get("gaussian", True)
        )
        agent.load_state_dict(ckpt["agent"])
        agent.to(device)
        agent.eval()
        agent.requires_grad_(False)

        return agent
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def get_act_fn_from_ckpt(
    ckpt_path, device=torch.device("cpu"), sample_action=False, ret_actor=False
):
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy_type = ckpt["policy_type"]

    actor = get_actor_from_ckpt(ckpt_path, device=device)

    if policy_type == "ppo":
        act_fn = actor.act if sample_action else actor.act_inference
    elif policy_type == "ppo_lstm":
        act_fn = actor.act if sample_action else actor.act_inference
    elif policy_type == "dagger":
        act_fn = partial(actor.forward, deterministic=True)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    if ret_actor:
        return act_fn, actor

    return act_fn
