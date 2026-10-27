import gymnasium as gym

from humanoid.envs.humanoidverse.tasks.catching.catch_upper import CatchUpperEnv


class HumanoidverseVectorEnv(gym.vector.VectorEnv):
    def __init__(
        self,
        humanoidverse_env: CatchUpperEnv,
    ):
        super().__init__()

        self.humanoidverse_env = humanoidverse_env
        self.control_freq = int(
            round(
                self.humanoidverse_env.config.simulator.sim.fps
                / self.humanoidverse_env.config.simulator.sim.control_decimation
            )
        )
        self.sim_freq = self.humanoidverse_env.config.simulator.sim.fps

    def reset(self, *args, **kwargs):
        return self.humanoidverse_env.reset()

    def step(self, action):
        return self.humanoidverse_env.step(action)

    def render(self):
        return self.humanoidverse_env.render()

    def __getattr__(self, name):
        return getattr(self.humanoidverse_env, name)

    # def close(self):
    #     self.humanoidverse_env.close()
