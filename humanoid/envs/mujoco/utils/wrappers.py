from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv


class MujocoAsyncVectorEnv(AsyncVectorEnv):
    def __init__(self, env_fns, **kwargs):
        super().__init__(env_fns, **kwargs)
        dummy_env = env_fns[0]().unwrapped
        self.control_freq = dummy_env.control_freq
        self.sim_freq = dummy_env.sim_freq
        self.sim_timestep = dummy_env.sim_timestep
        self.control_timestep = dummy_env.control_timestep
        self.config = dummy_env.config
        dummy_env.close()

    def render(self, **kwargs):
        self.call_async("render", **kwargs)
        return self.call_wait()


class MujocoSyncVectorEnv(SyncVectorEnv):
    def __init__(self, env_fns, **kwargs):
        super().__init__(env_fns, **kwargs)
        dummy_env = env_fns[0]().unwrapped
        self.control_freq = dummy_env.control_freq
        self.sim_freq = dummy_env.sim_freq
        self.sim_timestep = dummy_env.sim_timestep
        self.control_timestep = dummy_env.control_timestep
        self.config = dummy_env.config
        dummy_env.close()

    def render(self, **kwargs):
        return self.call("render", **kwargs)
