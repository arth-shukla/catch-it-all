import gymnasium as gym


class Wrapper(gym.vector.VectorWrapper):
    @property
    def privileged_observation_space(self) -> gym.Space:
        return getattr(
            self.env, "privileged_observation_space", self.env.observation_space
        )

    @property
    def single_privileged_observation_space(self) -> gym.Space:
        return getattr(
            self.env,
            "single_privileged_observation_space",
            self.env.single_observation_space,
        )
