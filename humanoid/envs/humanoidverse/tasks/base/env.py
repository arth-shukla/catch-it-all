import numpy as np
import torch

from isaacgym import gymapi, gymtorch

from humanoid.envs.humanoidverse.config.catching.catch_upper import (
    G1CatchUpperLockLegsConfig,
)
from humanoid.envs.humanoidverse.simulator.isaacgym_catching.sim import (
    IsaacGymCatchingSimulator,
)
from humanoid.utils.array import to_tensor
from humanoid.utils.dataclasses import get_class


# Base class for RL tasks
class BaseEnv:
    def __init__(self, config: G1CatchUpperLockLegsConfig, device: torch.device):
        self.config = config

        self.used_obs_keys = set()
        all_obs_keys = []
        for key_list in self.config.obs.obs_dict.__dict__.values():
            all_obs_keys += key_list
        for key_dict in self.config.obs.obs_auxiliary.__dict__.values():
            all_obs_keys += list(key_dict.keys())
        for obs_key in all_obs_keys:
            raw_suffix = "_raw"
            if obs_key.endswith(raw_suffix):
                self.used_obs_keys.add(obs_key[: -len(raw_suffix)])
            self.used_obs_keys.add(obs_key)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # self.simulator = instantiate(config=self.config.simulator, device=device)
        SimulatorClass = get_class(self.config.simulator._target_)
        self.simulator: IsaacGymCatchingSimulator = SimulatorClass(
            config=self.config, device=device
        )

        self.headless = config.env.headless
        self.simulator.set_headless(self.headless)
        self.simulator.setup()
        self.device = self.simulator.sim_device
        self.sim_dt = self.simulator.sim_dt
        self.up_axis_idx = 2

        self.dt = self.config.simulator.sim.control_decimation * self.sim_dt
        self.max_episode_steps = self.config.env.max_episode_steps
        self.max_episode_length_s = round(self.max_episode_steps * self.dt, 2)

        self.num_envs = self.config.env.num_envs
        self.dim_obs = self.config.robot.policy_obs_dim
        self.dim_critic_obs = self.config.robot.critic_obs_dim
        self.dim_teacher_obs = self.config.robot.teacher_obs_dim
        self.dim_actions = self.config.robot.actions_dim

        self.simulator.setup_terrain()

        # create envs, sim and viewer
        self._load_assets()
        self._get_env_origins()
        self._create_envs()
        self.dof_pos_limits, self.dof_vel_limits, self.torque_limits = (
            self.simulator.get_dof_limits_properties()
        )
        self._setup_robot_body_indices()
        # self._create_sim()
        self.simulator.prepare_sim()
        # if running with a viewer, set up keyboard shortcuts and camera
        self.viewer = None
        self.render_cameras = None
        if self.config.env.render_mode == "human":
            assert not self.headless
            self.debug_viz = False
            self.simulator.setup_viewer()
            self.viewer = self.simulator.viewer
        elif self.config.env.render_mode == "rgb_array":
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.config.camera.width
            camera_props.height = self.config.camera.height
            camera_props.horizontal_fov = self.config.camera.horizontal_fov
            camera_props.enable_tensors = True
            self.render_cameras = []
            for env_id, env_ptr in enumerate(self.simulator.envs):
                camera_handle = self.simulator.gym.create_camera_sensor(
                    env_ptr, camera_props
                )
                if self.config.camera.attach_body is not None:
                    local_transform = gymapi.Transform()
                    local_transform.p = gymapi.Vec3(*self.config.camera.local_cam_pos)
                    local_transform.r = gymapi.Quat(*self.config.camera.local_cam_quat)
                    self.simulator.gym.attach_camera_to_body(
                        camera_handle,
                        env_ptr,
                        self.simulator.find_rigid_body_indice(
                            self.config.camera.attach_body
                        ),
                        local_transform,
                        gymapi.FOLLOW_TRANSFORM,
                    )
                else:
                    ex, ey, ez = self.env_origins[env_id].tolist()
                    self.simulator.gym.set_camera_location(
                        camera_handle,
                        env_ptr,
                        gymapi.Vec3(1.25 + ex, 1.25 + ey, 1.6 + ez),
                        gymapi.Vec3(1.0 + ex, 1.0 + ey, 1.3 + ez),
                    )
                self.render_cameras.append(camera_handle)

        self._init_buffers()

    def _init_buffers(self):
        self.obs_buf_dict = {}
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.raw_rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.pen_rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.task_rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.term_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.trunc_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.extras = {}
        self.log_dict = {}

    def _refresh_sim_tensors(self):
        self.simulator.refresh_sim_tensors()
        return

    def reset(self):
        """Reset all robots"""
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
        self.simulator.set_actor_root_state_tensor(
            torch.arange(self.num_envs, device=self.device),
            self.simulator.all_root_states,
        )
        self.simulator.set_dof_state_tensor(
            torch.arange(self.num_envs, device=self.device), self.simulator.dof_state
        )
        # self._refresh_env_idx_tensors(torch.arange(self.num_envs, device=self.device))
        actions = torch.zeros(
            self.num_envs, self.dim_actions, device=self.device, requires_grad=False
        )
        obs_dict, _, _, _, infos = self.step(actions)
        if self.config.env.init_at_random_ep_len:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_steps)
            )
        return obs_dict, infos

    # def _refresh_env_idx_tensors(self, env_ids):
    #     env_ids_int32 = env_ids.to(dtype=torch.int32)
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                 gymtorch.unwrap_tensor(self.all_root_states),
    #                                                 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    #     self.gym.set_dof_state_tensor_indexed(self.sim,
    #                                             gymtorch.unwrap_tensor(self.dof_state),
    #                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def render(self, sync_frame_time=True):
        if self.viewer:
            self.simulator.render(sync_frame_time)
        elif self.render_cameras:
            if self.device != "cpu":
                self.simulator.gym.fetch_results(self.simulator.sim, True)
            self.simulator.gym.step_graphics(self.simulator.sim)
            self.simulator.gym.render_all_camera_sensors(self.simulator.sim)
            self.simulator.gym.start_access_image_tensors(self.simulator.sim)
            render_images = []
            for env_ptr, render_camera in zip(self.simulator.envs, self.render_cameras):
                if self.device == "cpu":
                    img = torch.from_numpy(
                        self.simulator.gym.get_camera_image(
                            self.simulator.sim,
                            env_ptr,
                            render_camera,
                            gymapi.IMAGE_COLOR,
                        )
                    )
                else:
                    img = gymtorch.wrap_tensor(
                        self.simulator.gym.get_camera_image_gpu_tensor(
                            self.simulator.sim,
                            env_ptr,
                            render_camera,
                            gymapi.IMAGE_COLOR,
                        )
                    )
                render_images.append(img)
            self.simulator.gym.end_access_image_tensors(self.simulator.sim)
            return torch.cat(render_images).view(
                self.num_envs, self.config.camera.height, self.config.camera.width, -1
            )[..., :3]

    ###########################################################################
    #### Helper functions
    ###########################################################################
    def _get_env_origins(self):
        """Sets environment origins on a grid."""
        self.custom_origins = False
        self.env_origins = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False
        )
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.config.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
        self.env_origins[:, 2] = 0.0

    def _load_assets(self):

        self.simulator.load_assets()
        self.num_dof, self.num_bodies, self.dof_names, self.body_names = (
            self.simulator.num_dof,
            self.simulator.num_bodies,
            self.simulator.dof_names,
            self.simulator.body_names,
        )

        # check dimensions
        assert (
            self.num_dof == self.dim_actions
        ), "Number of DOFs must be equal to number of actions"

        # other properties
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        base_init_state_list = (
            self.config.robot.init_state.pos
            + self.config.robot.init_state.rot
            + self.config.robot.init_state.lin_vel
            + self.config.robot.init_state.ang_vel
        )
        self.base_init_state = to_tensor(
            base_init_state_list, device=self.device, requires_grad=False
        )

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        # env_config = self.config
        self.simulator.create_envs(
            self.num_envs, self.env_origins, self.base_init_state
        )

    def _setup_robot_body_indices(self):
        penalized_contact_names = []
        for name in self.config.robot.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.config.robot.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):

            self.penalised_contact_indices[i] = self.simulator.find_rigid_body_indice(
                penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.simulator.find_rigid_body_indice(
                termination_contact_names[i]
            )

        if self.config.robot.has_torso:
            self.torso_name = self.config.robot.torso_name
            self.torso_index = self.simulator.find_rigid_body_indice(self.torso_name)
