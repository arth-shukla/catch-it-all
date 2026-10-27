import copy
import os
from pathlib import Path

import torch


def export(ckpt_path, policy_type=None):
    ckpt_path = Path(ckpt_path)
    output_path = ckpt_path.with_suffix(".jit.pt")
    ckpt = torch.load(ckpt_path, weights_only=False)
    policy_type = ckpt["policy_type"]

    os.makedirs(Path(output_path).parent, exist_ok=True)

    if policy_type == "ppo":
        from humanoid.agents.make import get_actor_from_ckpt

        actor_critic = get_actor_from_ckpt(ckpt_path, device=torch.device("cpu"))
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(output_path)
    elif policy_type == "ppo_lstm":
        from humanoid.agents.make import get_actor_from_ckpt
        from humanoid.agents.ppo.jit_export import export_ppo_lstm_policy_as_jit

        actor_critic = get_actor_from_ckpt(ckpt_path, device=torch.device("cpu"))
        export_ppo_lstm_policy_as_jit(actor_critic, output_path)
    elif policy_type == "dagger":
        from humanoid.agents.make import get_actor_from_ckpt

        agent = get_actor_from_ckpt(ckpt_path, device=torch.device("cpu"))
        traced_script_module = torch.jit.script(agent)
        traced_script_module.save(output_path)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


if __name__ == "__main__":
    import sys

    export(*sys.argv[1:])
