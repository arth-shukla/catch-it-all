import multiprocessing as mp
import statistics
from collections import deque
from functools import partial

import cloudpickle

import torch

from humanoid.envs.env_cfg import EnvConfig
from humanoid.envs.make import make_env
from humanoid.utils.array import recursive_to_python, to_tensor


class Evaluator:
    def __init__(self, eval_seed, env_cfg, device, call_fn_str: str):
        self.parent, self.child = mp.Pipe()
        rollout_callback = partial(eval_rollout, call_fn_str=call_fn_str)
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        self.p = mp.Process(
            target=rollout_callback,
            args=(eval_seed, env_cfg, device, self.child),
            daemon=True,
        )
        self.p.start()

    def eval_policy(self, policy, normalizer=None):
        self.parent.send([cloudpickle.dumps(policy), normalizer])
        return self.parent.recv()

    def close(self):
        # Send None to signal the child process to exit
        self.parent.send([None, None])
        # Wait for the process to terminate
        self.p.join(timeout=5)
        # If process is still alive after timeout, terminate it forcefully
        if self.p.is_alive():
            self.p.terminate()
            self.p.join()
        # Close the pipe
        self.parent.close()
        self.child.close()


def eval_rollout(
    eval_seed, eval_env_cfg: EnvConfig, device, child, call_fn_str: str = None
):
    import pickle

    eval_env_cfg.num_envs = 150
    eval_envs = make_env(eval_env_cfg)
    eval_obs, eval_infos = eval_envs.reset(seed=eval_seed)
    eval_envs.action_space.seed(eval_seed)

    with torch.inference_mode():
        while True:
            [actor, normalizer] = child.recv()
            if actor is None:
                break
            actor = pickle.loads(actor)
            actor.eval()
            normalizer

            eval_obs, eval_infos = eval_envs.reset()

            eval_rewbuffer = []
            eval_lenbuffer = []
            eval_cur_reward_sum = torch.zeros(
                eval_envs.num_envs, dtype=torch.float, device=device
            )
            eval_cur_episode_length = torch.zeros(
                eval_envs.num_envs, dtype=torch.float, device=device
            )

            if "success" in eval_infos:
                success_once_buffer = deque(maxlen=100)
                success_at_end_buffer = deque(maxlen=100)
                cur_success_once = torch.zeros(
                    eval_envs.num_envs, dtype=torch.bool, device=device
                )

            max_steps = getattr(
                eval_envs,
                "max_episode_steps",
                getattr(eval_envs.unwrapped, "max_episode_steps", None),
            )
            with torch.inference_mode():
                for _ in range(max_steps):
                    if normalizer is not None:
                        eval_actions = getattr(actor, call_fn_str)(
                            normalizer.normalize(eval_obs)
                        )
                    else:
                        eval_actions = getattr(actor, call_fn_str)(eval_obs)

                    (
                        next_eval_obs,
                        eval_rewards,
                        eval_terminations,
                        eval_truncations,
                        eval_infos,
                    ) = eval_envs.step(eval_actions)

                    # Book keeping
                    eval_cur_reward_sum += eval_rewards
                    eval_cur_episode_length += 1

                    if "success" in eval_infos:
                        success = to_tensor(eval_infos["success"], device)
                        if "final_info" in eval_infos:
                            _final_info_mask = to_tensor(
                                eval_infos["_final_info"], device
                            )
                            success[_final_info_mask] = to_tensor(
                                eval_infos["final_info"]["success"][_final_info_mask],
                                device,
                            )
                        cur_success_once |= success

                    eval_dones = eval_terminations | eval_truncations
                    if torch.any(eval_dones):
                        eval_rewbuffer.extend(
                            recursive_to_python(eval_cur_reward_sum[eval_dones])
                        )
                        eval_lenbuffer.extend(
                            recursive_to_python(eval_cur_episode_length[eval_dones])
                        )
                        eval_cur_reward_sum[eval_dones] = 0
                        eval_cur_episode_length[eval_dones] = 0
                        if "success" in eval_infos:
                            success_once_buffer.extend(
                                recursive_to_python(cur_success_once[eval_dones])
                            )
                            success_at_end_buffer.extend(
                                recursive_to_python(success[eval_dones])
                            )
                            cur_success_once[eval_dones] = False
                    eval_obs = next_eval_obs

            eval_stats = {
                "ep_return": statistics.mean(eval_rewbuffer),
                "len": statistics.mean(eval_lenbuffer),
            }
            if "success" in eval_infos:
                eval_stats.update(
                    success_once=statistics.mean(success_once_buffer),
                    success_at_end=statistics.mean(success_at_end_buffer),
                )
            child.send(eval_stats)
    child.close()
