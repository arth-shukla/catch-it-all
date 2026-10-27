import sys


if sys.version_info < (3, 9):
    # NOTE (arth): import first to avoid importing after torch:
    import isaacgym

import math

import gymnasium as gym
from gymnasium.vector.utils import batch_space

import torch

from humanoid.envs.humanoidverse.config.catching.catch_upper import (
    G1CatchUpperLockLegsConfig,
)
from humanoid.envs.humanoidverse.tasks.legged_base.env import LeggedRobotBaseEnv
from humanoid.envs.humanoidverse.utils.kinematics import Kinematics
from humanoid.envs.humanoidverse.utils.spatial_utils.rotations import wrap_to_pi
from humanoid.envs.humanoidverse.utils.torch_utils import *
from humanoid.utils.array import to_tensor


class CatchUpperEnv(LeggedRobotBaseEnv):
    def __init__(self, config: G1CatchUpperLockLegsConfig, device):
        self.init_done = False
        super().__init__(config, device)
        self.init_done = True
        self.config: G1CatchUpperLockLegsConfig

    def _load_assets(self):

        self.simulator.load_assets()
        self.num_dof, self.num_bodies, self.dof_names, self.body_names = (
            self.simulator.num_dof,
            self.simulator.num_bodies,
            self.simulator.dof_names,
            self.simulator.body_names,
        )

        # # check dimensions
        # assert (
        #     self.num_dof == self.dim_actions
        # ), "Number of DOFs must be equal to number of actions"

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

    def _setup_gym_spaces(self):

        self.action_scale = self.config.robot.control.pos_controller_config.action_scale
        gym_space_action_scale = self.action_scale

        # self.action_scale = self.config.robot.control.pos_controller_config.action_scale
        # NOTE (arth): first get the position control limits
        assert (
            self.config.robot.control.pos_controller_config.action_clip_value
            == "joint_limits"
        )
        self._act_limit_low = (
            (
                (
                    self.simulator.hard_dof_pos_limits[:, 0]
                    - self.default_dof_pos.squeeze(0)
                )
                / gym_space_action_scale
            )
            .float()
            .to(self.device)
        ) * self.config.robot.control.pos_controller_config.action_lim_scale
        self._act_limit_high = (
            (
                (
                    self.simulator.hard_dof_pos_limits[:, 1]
                    - self.default_dof_pos.squeeze(0)
                )
                / gym_space_action_scale
            )
            .float()
            .to(self.device)
        ) * self.config.robot.control.pos_controller_config.action_lim_scale

        finger_act_space_low = self._act_limit_low[self.finger_joint_indices].clone()
        finger_act_space_high = self._act_limit_high[self.finger_joint_indices].clone()

        # NOTE (arth): then get the arm control limits based on pos or target_delta_pos controller
        if self.config.robot.control.arm_control_mode == "pos":
            arm_act_space_low = self._act_limit_low[self.arm_joint_indices].clone()
            arm_act_space_high = self._act_limit_high[self.arm_joint_indices].clone()

            self.act_space_low = torch.cat([finger_act_space_low, arm_act_space_low])
            self.act_space_high = torch.cat([finger_act_space_high, arm_act_space_high])
            act_space_dim = (
                self.finger_joint_indices.numel() + self.arm_joint_indices.numel()
            )
        elif self.config.robot.control.arm_control_mode == "ik":
            self.ik_config = self.config.robot.control.ik_controller_config
            arm_act_space_high = torch.tensor(
                [self.ik_config.max_delta_pos * self.ik_config.delta_pos_action_scale]
                * 3
                + [self.ik_config.max_delta_rpy * self.ik_config.delta_rpy_action_scale]
                * 3,
                device=self.device,
                dtype=torch.float,
            )
            arm_act_space_low = -arm_act_space_high.clone()

            self.act_space_low = torch.cat([finger_act_space_low, arm_act_space_low])
            self.act_space_high = torch.cat([finger_act_space_high, arm_act_space_high])
            act_space_dim = self.finger_joint_indices.numel() + 6
        else:
            assert self.config.robot.control.arm_control_mode == "target_delta_pos"
            assert (
                self.config.robot.control.target_delta_pos_controller_config.normalize_action_space
            )
            action_lim_scale = (
                self.config.robot.control.target_delta_pos_controller_config.action_lim_scale
            )
            arm_act_space_low = (
                -torch.ones_like(self._act_limit_low[self.arm_joint_indices])
                * action_lim_scale
            )
            arm_act_space_high = (
                torch.ones_like(self._act_limit_low[self.arm_joint_indices])
                * action_lim_scale
            )

            self.act_space_low = torch.cat([finger_act_space_low, arm_act_space_low])
            self.act_space_high = torch.cat([finger_act_space_high, arm_act_space_high])
            act_space_dim = (
                self.finger_joint_indices.numel() + self.arm_joint_indices.numel()
            )

        _act_space_low = self.act_space_low.cpu().numpy()
        _act_space_high = self.act_space_high.cpu().numpy()
        self.single_action_space = gym.spaces.Box(
            low=_act_space_low,
            high=_act_space_high,
            shape=(act_space_dim,),
            dtype=np.float32,
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        assert self.config.robot.actions_dim == math.prod(
            self.single_action_space.shape
        )
        assert self.config.robot.actions_dim * self.num_envs == math.prod(
            self.action_space.shape
        )

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.dim_obs,),
            dtype=np.float32,
        )
        self.single_privileged_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.dim_critic_obs,),
            dtype=np.float32,
        )
        self.single_teacher_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.dim_teacher_obs,),
            dtype=np.float32,
        )
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.privileged_observation_space = batch_space(
            self.single_privileged_observation_space, self.num_envs
        )
        self.teacher_observation_space = batch_space(
            self.single_teacher_observation_space, self.num_envs
        )

    def _init_kinematics(self):
        self.kinematics = Kinematics(
            self.config.robot.asset.asset_root / self.config.robot.asset.urdf_file,
            "torso_link",
            f"{self.config.robot.used_hand}_wrist_yaw_link",
            dict((name, i) for i, name in enumerate(self.arm_joint_names)),
            self.arm_link_names,
            self.device,
        )
        eef_pos, eef_quat = self.kinematics.forward_kinematics(
            self.default_dof_pos[:, self.arm_joint_indices].repeat(self.num_envs, 1)
        )
        self.default_eef_pose = torch.cat([eef_pos, eef_quat], dim=1)
        self.target_eef_pose = self.default_eef_pose.clone()
        self.prev_target_eef_pose = self.target_eef_pose.clone()

        self.eef_pos_min = self.default_eef_pose[:, :3] + torch.tensor(
            list(self.ik_config.eef_pos_limits_lower),
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)
        self.eef_pos_max = self.default_eef_pose[:, :3] + torch.tensor(
            list(self.ik_config.eef_pos_limits_upper),
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)

        self._ik_dof_pos_noise_scale = self.config.obs.noise_scales[
            "dof_pos_finger_arm"
        ]
        self._ik_target_joint_angles = self.default_dof_pos[
            :, self.arm_joint_indices
        ].repeat(self.num_envs, 1)
        self._prev_ik_target_joint_angles = self._ik_target_joint_angles.clone()
        self._prev_prev_ik_target_joint_angles = self._ik_target_joint_angles.clone()

    def _init_buffers(self):
        super()._init_buffers()

        if self.config.env.toss_noise_curriculum:
            self.throw_target_noise_scale = self.config.env.toss_noise_initial
            self.log_dict["throw_target_noise_scale"] = self.throw_target_noise_scale

        if self.config.env.toss_type == "underhand":
            toss_lims = self.config.env.toss_lims[
                f"underhand_{self.config.robot.used_hand}"
            ]
            self.toss_lims_list = [
                list(toss_lims["x"]),
                list(toss_lims["y"]),
                list(toss_lims["z"]),
            ]
            self.toss_lims = torch.tensor(
                self.toss_lims_list,
                dtype=torch.float,
                device=self.device,
            ).unsqueeze(0)
            self.init_toss_shift = None
        elif self.config.env.toss_type == "overhand":
            self.overhand_dof_pos = to_tensor(
                [
                    [
                        self.config.robot.init_state.overhand_joint_angles[name]
                        for name in self.dof_names
                    ]
                ],
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            toss_lims = self.config.env.toss_lims[
                f"overhand_{self.config.robot.used_hand}"
            ]
            self.toss_lims_list = [
                list(toss_lims["x"]),
                list(toss_lims["y"]),
                list(toss_lims["z"]),
            ]
            self.toss_lims = torch.tensor(
                self.toss_lims_list,
                dtype=torch.float,
                device=self.device,
            ).unsqueeze(0)
            self.init_toss_shift = torch.tensor(
                [-0.1, 0.0, 0.1],
                dtype=torch.float,
                device=self.device,
            ).unsqueeze(0)
        elif self.config.env.toss_type == "both":
            self.overhand_dof_pos = to_tensor(
                [
                    [
                        self.config.robot.init_state.overhand_joint_angles[name]
                        for name in self.dof_names
                    ]
                ],
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )

            self.is_underhand = torch.randint(
                0, 2, (self.num_envs,), dtype=torch.bool, device=self.device
            )
            self.is_overhand = ~self.is_underhand
            self.under_env_ids = self.is_underhand.nonzero(as_tuple=False).flatten()
            self.over_env_ids = self.is_overhand.nonzero(as_tuple=False).flatten()

            utoss_lims = self.config.env.toss_lims[
                f"underhand_{self.config.robot.used_hand}"
            ]
            utoss_lims = torch.tensor(
                [
                    list(utoss_lims["x"]),
                    list(utoss_lims["y"]),
                    list(utoss_lims["z"]),
                ],
                dtype=torch.float,
                device=self.device,
            ).unsqueeze(0)

            otoss_lims = self.config.env.toss_lims[
                f"overhand_{self.config.robot.used_hand}"
            ]
            otoss_lims = torch.tensor(
                [
                    list(otoss_lims["x"]),
                    list(otoss_lims["y"]),
                    list(otoss_lims["z"]),
                ],
                dtype=torch.float,
                device=self.device,
            ).unsqueeze(0)
            oinit_toss_shift = torch.tensor(
                [-0.1, 0.0, 0.1],
                dtype=torch.float,
                device=self.device,
            ).unsqueeze(0)

            self.toss_lims = torch.empty_like(otoss_lims).repeat(self.num_envs, 1, 1)
            self.toss_lims[self.under_env_ids] = utoss_lims
            self.toss_lims[self.over_env_ids] = otoss_lims

            self.init_toss_shift = torch.zeros_like(oinit_toss_shift).repeat(
                self.num_envs, 1
            )
            self.init_toss_shift[self.over_env_ids] = oinit_toss_shift
        else:
            raise NotImplementedError(self.config.env.toss_type)

        # from isaacgym import gymapi

        # self.simulator.gym.subscribe_viewer_keyboard_event(
        #     self.simulator.viewer, gymapi.KEY_1, "inc_x"
        # )
        # self.simulator.gym.subscribe_viewer_keyboard_event(
        #     self.simulator.viewer, gymapi.KEY_3, "inc_y"
        # )
        # self.simulator.gym.subscribe_viewer_keyboard_event(
        #     self.simulator.viewer, gymapi.KEY_5, "inc_z"
        # )
        # self.simulator.gym.subscribe_viewer_keyboard_event(
        #     self.simulator.viewer, gymapi.KEY_2, "dec_x"
        # )
        # self.simulator.gym.subscribe_viewer_keyboard_event(
        #     self.simulator.viewer, gymapi.KEY_4, "dec_y"
        # )
        # self.simulator.gym.subscribe_viewer_keyboard_event(
        #     self.simulator.viewer, gymapi.KEY_6, "dec_z"
        # )

        self.action_max = (
            self.simulator.hard_dof_pos_limits[:, 1].unsqueeze(0) - self.default_dof_pos
        ) / self.action_scale
        self.action_min = (
            self.simulator.hard_dof_pos_limits[:, 0].unsqueeze(0) - self.default_dof_pos
        ) / self.action_scale

        self.obj_target_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.toss_dist_ranges = torch.tensor(
            [
                [1.0, 1.45],  # short
                [1.45, 2.0],  # med
                [2.0, 2.75],  # long
            ],
            dtype=torch.float,
            device=self.device,
        )

        if self.config.robot.control.arm_control_mode == "target_delta_pos":
            self.target_dof_pos = torch.zeros(
                (self.num_envs, self.arm_joint_indices.numel()),
                device=self.device,
                dtype=torch.float,
            )
        elif self.config.robot.control.arm_control_mode == "ik":
            self._init_kinematics()

        self._kinematics_for_hand_pos = Kinematics(
            self.config.robot.asset.asset_root / self.config.robot.asset.urdf_file,
            "pelvis",
            f"{self.config.robot.used_hand}_palm",
            dict((name, i) for i, name in enumerate(self.arm_joint_names)),
            [
                "waist_yaw_link",
                "waist_roll_link",
                "torso_link",
                *self.arm_link_names,
                f"{self.config.robot.used_hand}_palm",
            ],
            self.device,
        )

        self.episode_out_of_torque_lims = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int32
        )
        self.average_out_of_torque_lims = 0.0

        self.episode_obj_in_hand_goal_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.average_obj_in_hand_goal_once_rate = 0.0

        self.episode_obj_in_hand_goal_static_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.average_obj_in_hand_goal_static_once_rate = 0.0

        self.episode_obj_near_hand_goal_obj_static_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.average_obj_near_hand_goal_obj_static_once_rate = 0.0

        self.obj_near_hand_goal_obj_static = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.average_obj_near_hand_goal_obj_static_at_end_rate = 0.0

        self.hand_goal_point_in_hand = torch.tensor(
            [[0.05, 0.0, -0.05]],
            device=self.device,  # [0.1, -0.1, 0.0]
            # [[0.05, -0.05, 0.0]], device=self.device  # [0.1, -0.1, 0.0]
        ).repeat((self.num_envs, 1))

        self.fingertip_pos = torch.zeros(
            (self.num_envs, len(self.fingertip_indices), 3), device=self.device
        )
        self.fingertip_shift = (
            torch.tensor(
                [-0.0175, 0.03, -0.01],
                dtype=torch.float,
                device=self.device,
            )
            .unsqueeze(0)
            .unsqueeze(1)
            .repeat((self.num_envs, len(self.fingertip_indices), 1))
        )

        self.obj_init_state = torch.zeros(
            (self.num_envs, 13), device=self.device, dtype=torch.float
        )
        self.obj_goal_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.throw_wait_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.obj_thrown = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.wait_for_throw = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        # cam pos

        if self.config.domain_rand.randomize_cam_pos_bias:
            self.cam_pos_bias = torch_rand_float(
                -self.config.domain_rand.cam_pos_bias,
                self.config.domain_rand.cam_pos_bias,
                (self.num_envs, 3),
                device=self.device,
            )
        self.cam_local_quat = (
            torch.tensor(
                list(self.config.camera.local_cam_quat),
                device=self.device,
                dtype=torch.float,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self.cam_local_pos = (
            torch.tensor(
                list(self.config.camera.local_cam_pos),
                device=self.device,
                dtype=torch.float,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # setup tracking vals

        with torch.device(self.device):

            # obj
            self.obj_pos = torch.zeros((self.num_envs, 3), dtype=torch.float)
            self.obj_quat = torch.zeros((self.num_envs, 4), dtype=torch.float)
            self.obj_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float)
            self.obj_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float)

            self.obj_pos_delay = torch.zeros((self.num_envs, 3), dtype=torch.float)
            if self.config.domain_rand.randomize_obj_latency:
                self.obj_pos_queue = torch.zeros(
                    self.num_envs,
                    self.config.domain_rand.obj_latency_range[1] + 1,
                    3,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                self.obj_latency_idx = torch.randint(
                    self.config.domain_rand.obj_latency_range[0],
                    self.config.domain_rand.obj_latency_range[1] + 1,
                    (self.num_envs,),
                    device=self.device,
                    requires_grad=False,
                )

            # fingertip
            self.fingertip_pos = torch.zeros(
                (self.num_envs, len(self.fingertip_indices), 3),
                dtype=torch.float,
            )

            # head
            self.head_pos = torch.zeros((self.num_envs, 3), dtype=torch.float)
            self.head_quat = torch.zeros((self.num_envs, 4), dtype=torch.float)
            self.head_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float)
            self.head_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float)

            # torso
            self.torso_pos = torch.zeros((self.num_envs, 3), dtype=torch.float)
            self.torso_quat = torch.zeros((self.num_envs, 4), dtype=torch.float)

            # hand
            self.last_last_hand_pos = (
                None  # torch.zeros((self.num_envs, 3), dtype=torch.float)
            )
            self.last_hand_pos = (
                None  # torch.zeros((self.num_envs, 3), dtype=torch.float)
            )
            self.hand_pos = torch.zeros((self.num_envs, 3), dtype=torch.float)
            self.hand_quat = torch.zeros((self.num_envs, 4), dtype=torch.float)
            self.hand_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float)
            self.hand_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float)
            self.hand_projected_gravity = torch.zeros(
                (self.num_envs, 3), dtype=torch.float
            )
            self.hand_forward = torch.zeros((self.num_envs, 3), dtype=torch.float)

            # hand goal
            self.hand_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float)

            # hand dists
            self.hand_goal_obj_dist = torch.zeros(self.num_envs, dtype=torch.float)

            # camera
            self.camera_quat = torch.zeros((self.num_envs, 4), dtype=torch.float)
            self.camera_pos = torch.zeros((self.num_envs, 3), dtype=torch.float)
            self.camera_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float)
            self.camera_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float)

            self._camera_obs = any(
                x in self.used_obs_keys
                for x in [
                    "obj_pose_wrt_camera",
                    "obj_pos_wrt_camera_clean",
                    "obj_pos_wrt_camera_noisy",
                    "obj_lin_vel_wrt_camera",
                    "obj_ang_vel_wrt_camera",
                ]
            )
            if self._camera_obs:
                # obj wrt cam (+ custom noise)
                # NOTE (arth): isaacgym has x-forward, y-left, z-up
                # cameras w/ 0px,0px top-right use x-right, y-down, z-forward
                self.u_obj = torch.zeros(self.num_envs, dtype=torch.float)
                self.v_obj = torch.zeros(self.num_envs, dtype=torch.float)

                self.obj_in_frame = torch.zeros(self.num_envs, dtype=torch.bool)

                self._obj_pos_wrt_cam_clean = torch.zeros(
                    (self.num_envs, 3), dtype=torch.float
                )
                self._obj_pos_wrt_cam_noisy = torch.zeros(
                    (self.num_envs, 3), dtype=torch.float
                )
                self._obj_pos_wrt_cam_noisy_after_delay = torch.zeros(
                    (self.num_envs, 3), dtype=torch.float
                )
                self._obj_quat_wrt_cam_clean = torch.zeros(
                    (self.num_envs, 4), dtype=torch.float
                )

                if self.config.domain_rand.randomize_cam_latency:
                    self.cam_pos_queue = torch.zeros(
                        self.num_envs,
                        self.config.domain_rand.cam_latency_range[1] + 1,
                        3,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    self.cam_latency_idx = torch.randint(
                        self.config.domain_rand.cam_latency_range[0],
                        self.config.domain_rand.cam_latency_range[1] + 1,
                        (self.num_envs,),
                        device=self.device,
                        requires_grad=False,
                    )

            # events
            self.obj_near_hand_goal = torch.zeros(self.num_envs, dtype=torch.bool)
            self.obj_vel_mag = torch.zeros(self.num_envs, dtype=torch.float)
            self.obj_static = torch.zeros(self.num_envs, dtype=torch.bool)
            self.obj_near_hand_goal_obj_static = torch.zeros(
                self.num_envs, dtype=torch.bool
            )

            self.obj_in_hand_goal = torch.zeros(self.num_envs, dtype=torch.bool)
            self.hand_static = torch.zeros(self.num_envs, dtype=torch.bool)
            self.obj_in_hand_goal_static = torch.zeros(self.num_envs, dtype=torch.bool)

            self.obj_stability_counter = torch.zeros(self.num_envs, dtype=torch.long)

    def update_trackers_step(self):
        super().update_trackers_step()
        obj_in_hand_goal = self.obj_in_hand_goal
        self.episode_obj_in_hand_goal_once |= obj_in_hand_goal
        self.episode_obj_in_hand_goal_static_once |= self.obj_in_hand_goal_static
        self.episode_obj_near_hand_goal_obj_static_once |= (
            self.obj_near_hand_goal_obj_static
        )

        out_of_torque_limits = torch.sum(
            (
                torch.abs(self.torques)
                - self.torque_limits
                * self.config.env.termination.scales.termination_close_to_torque_limit
            )
            > 0,
            dim=1,
            dtype=torch.int32,
        )
        self.episode_out_of_torque_lims += out_of_torque_limits
        self.log_dict["episode_out_of_torque_lims"] = (
            self.episode_out_of_torque_lims.clone()
        )

    def update_trackers_reset(self, env_ids):
        super().update_trackers_reset(env_ids)
        num = len(env_ids)
        current_average_obj_in_hand_goal_once_rate = torch.mean(
            self.episode_obj_in_hand_goal_once[env_ids], dtype=torch.float
        )
        self.average_obj_in_hand_goal_once_rate = (
            self.average_obj_in_hand_goal_once_rate
            * (1 - num / self.num_compute_average_epl)
            + current_average_obj_in_hand_goal_once_rate
            * (num / self.num_compute_average_epl)
        )

        current_average_obj_in_hand_goal_static_once_rate = torch.mean(
            self.episode_obj_in_hand_goal_static_once[env_ids], dtype=torch.float
        )
        self.average_obj_in_hand_goal_static_once_rate = (
            self.average_obj_in_hand_goal_static_once_rate
            * (1 - num / self.num_compute_average_epl)
            + current_average_obj_in_hand_goal_static_once_rate
            * (num / self.num_compute_average_epl)
        )

        current_average_obj_near_hand_goal_obj_static_once_rate = torch.mean(
            self.episode_obj_near_hand_goal_obj_static_once[env_ids], dtype=torch.float
        )
        self.average_obj_near_hand_goal_obj_static_once_rate = (
            self.average_obj_near_hand_goal_obj_static_once_rate
            * (1 - num / self.num_compute_average_epl)
            + current_average_obj_near_hand_goal_obj_static_once_rate
            * (num / self.num_compute_average_epl)
        )

        self.average_obj_near_hand_goal_obj_static_at_end_rate = (
            self.average_obj_near_hand_goal_obj_static_at_end_rate
            * (1 - num / self.num_compute_average_epl)
            + torch.mean(self.obj_near_hand_goal_obj_static[env_ids], dtype=torch.float)
            * (num / self.num_compute_average_epl)
        )

        current_average_out_of_torque_lims = torch.mean(
            self.episode_out_of_torque_lims[env_ids], dtype=torch.float
        )
        self.average_out_of_torque_lims = self.average_out_of_torque_lims * (
            1 - num / self.num_compute_average_epl
        ) + current_average_out_of_torque_lims * (num / self.num_compute_average_epl)

        self.log_dict["average_obj_in_hand_goal_once_rate"] = (
            self.average_obj_in_hand_goal_once_rate
        )
        self.log_dict["average_obj_in_hand_goal_static_once_rate"] = (
            self.average_obj_in_hand_goal_static_once_rate
        )
        self.log_dict["average_obj_near_hand_goal_obj_static_once_rate"] = (
            self.average_obj_near_hand_goal_obj_static_once_rate
        )
        self.log_dict["average_obj_near_hand_goal_obj_static_at_end_rate"] = (
            self.average_obj_near_hand_goal_obj_static_at_end_rate
        )
        self.log_dict["average_out_of_torque_lims"] = self.average_out_of_torque_lims

        self.episode_obj_in_hand_goal_once[env_ids] = False
        self.episode_obj_in_hand_goal_static_once[env_ids] = False
        self.episode_obj_near_hand_goal_obj_static_once[env_ids] = False
        self.episode_out_of_torque_lims[env_ids] = 0

    def _update_clip_torque_curriculum(self):
        if (
            self.average_episode_length
            > self.config.robot.control.clip_torque_level_up_threshold
        ):
            self.clip_torque_lim_scale *= (
                1 + self.config.robot.control.clip_torque_degree
            )

        self.clip_torque_lim_scale = np.clip(
            self.clip_torque_lim_scale,
            self.config.robot.control.clip_torque_lim_min,
            self.config.robot.control.clip_torque_lim_max,
        )

    def _update_target_noise_curriculum(self):
        self.log_dict["throw_target_noise_scale"] = self.throw_target_noise_scale
        if self.average_episode_length > self.config.env.toss_noise_level_up_threshold:
            self.throw_target_noise_scale *= 1 + self.config.env.toss_noise_degree

        self.throw_target_noise_scale = np.clip(
            self.throw_target_noise_scale,
            self.config.env.toss_noise_min,
            self.config.env.toss_noise_max,
        )

    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        self._episodic_domain_randomization(env_ids)
        if self.config.robot.control.clip_torque_curriculum:
            self._update_clip_torque_curriculum()
        if self.config.env.toss_noise_curriculum:
            self._update_target_noise_curriculum()

    def randomize_projectile_start(self, env_ids: torch.Tensor):
        if self.config.env.projectiles.ensure_obj_in_frame:
            min_start_x, min_start_y, min_start_z = 0.5, -1.0, 0.25
            max_start_x, max_start_y, max_start_z = 2.75, 1.0, 2.0
        else:
            min_start_x, min_start_y, min_start_z = 0.5, -2.0, 0.25
            max_start_x, max_start_y, max_start_z = 2.75, 2.0, 2.0

        prob_short_toss = 0.4
        prob_med_toss = 0.4
        prob_long_toss = 0.2

        toss_rand_select = torch_rand_float(
            0, 1, (env_ids.numel(), 1), device=self.device
        )
        toss_type = torch.zeros_like(toss_rand_select, dtype=torch.int)
        toss_type[toss_rand_select > prob_short_toss] = 1
        toss_type[toss_rand_select > (prob_short_toss + prob_med_toss)] = 2

        toss_dist_range = self.toss_dist_ranges[toss_type]  # (num_envs, 1, 2)
        x_o = (toss_dist_range[..., 1] - toss_dist_range[..., 0]) * torch.rand(
            env_ids.numel(), 1, device=self.device
        ) + toss_dist_range[..., 0]
        y_o = torch_rand_float(
            min_start_y, max_start_y, (env_ids.numel(), 1), device=self.device
        )
        z_o = torch_rand_float(
            min_start_z, max_start_z, (env_ids.numel(), 1), device=self.device
        )

        return x_o, y_o, z_o

    def random_toss(
        self,
        env_ids: torch.Tensor,
    ):
        # get obj init pos (relative to env origins)
        x_o, y_o, z_o = self.randomize_projectile_start(env_ids)

        # NOTE (arth): separate toss curr from throwing

        # get obj end pos (relative to env origins)
        robot_init_pos = (
            self.simulator.robot_root_states[env_ids, :3].clone()
            - self.env_origins[env_ids]
        )

        if self.config.env.toss_noise_curriculum:
            robot_init_quat = self.simulator.robot_root_states[env_ids, 3:7].clone()
            arm_dof_pos_env_ids = self.simulator.dof_pos[
                env_ids[:, None], self.arm_joint_indices
            ].clone()
            hand_pos_wrt_pelvis, _ = self._kinematics_for_hand_pos.forward_kinematics(
                arm_dof_pos_env_ids
            )
            hand_init_pos = robot_init_pos + quat_apply(
                robot_init_quat, hand_pos_wrt_pelvis
            )

            toss_mean: torch.Tensor = hand_init_pos

            if self.config.env.toss_type == "both":
                _init_toss_shift = self.init_toss_shift[env_ids]
                _toss_lims = self.toss_lims[env_ids]
            else:
                _init_toss_shift = self.init_toss_shift
                _toss_lims = self.toss_lims
            if self.init_toss_shift is not None:
                toss_mean += _init_toss_shift
            toss_lims = _toss_lims + robot_init_pos.unsqueeze(2)
            toss_sample_range = torch.clip(
                toss_lims - toss_mean.unsqueeze(2),
                -self.throw_target_noise_scale,
                self.throw_target_noise_scale,
            )

            toss_targs = toss_mean + scale(
                torch_rand_float(-1.0, 1.0, toss_mean.shape, self.device),
                toss_sample_range[..., 0],
                toss_sample_range[..., 1],
            )
            x_f, y_f, z_f = (
                toss_targs[:, 0:1],
                toss_targs[:, 1:2],
                toss_targs[:, 2:3],
            )
        else:
            if self.config.env.toss_type == "both":
                robot_init_pos += scale(
                    torch_rand_float(-1.0, 1.0, robot_init_pos.shape, self.device),
                    self.toss_lims[env_ids, :, 0],
                    self.toss_lims[env_ids, :, 1],
                )
                x_f, y_f, z_f = (
                    robot_init_pos[:, 0:1],
                    robot_init_pos[:, 1:2],
                    robot_init_pos[:, 2:3],
                )
            else:
                x_f, y_f, z_f = (
                    robot_init_pos[:, 0:1],
                    robot_init_pos[:, 1:2],
                    robot_init_pos[:, 2:3],
                )
                x_f += torch_rand_float(
                    *self.toss_lims_list[0], x_f.shape, device=self.device
                )
                y_f += torch_rand_float(
                    *self.toss_lims_list[1],
                    y_f.shape,
                    device=self.device,
                )
                z_f += torch_rand_float(
                    *self.toss_lims_list[2], z_f.shape, device=self.device
                )

        delta_x = x_f - x_o
        delta_y = y_f - y_o
        delta_z = z_f - z_o

        # fmt: off
        t_impact = torch.rand((env_ids.numel(), 1), device=self.device)
        t_impact[delta_x < 1.0]                         = (0.8 - 0.4) * t_impact[delta_x < 1.0] + 0.4
        t_impact[(1.0 <= delta_x) & (delta_x <= 2.0)]   = (2.5 - 0.8) * t_impact[(1.0 <= delta_x) & (delta_x <= 2.0)] + 0.8
        t_impact[2.0 < delta_x]                         = (5.0 - 3.5) * t_impact[2.0 < delta_x] + 3.5
        # fmt: on

        # Solve velocity
        v_xo = delta_x / t_impact
        v_yo = delta_y / t_impact
        v_zo = (delta_z - 0.5 * self.gravity * torch.square(t_impact)) / t_impact

        # ang vel based on obj scale (larger scale -> less ang vel)
        obj_scale = self.simulator.projectile_scales[env_ids]
        obj_ang_vel = torch_rand_float(-1, 1, (env_ids.numel(), 3), device=self.device)
        obj_ang_vel[obj_scale < 0.85] *= 25
        obj_ang_vel[(0.85 <= obj_scale) & (obj_scale <= 1.25)] *= 15
        obj_ang_vel[1.25 < obj_scale] *= 12
        w_ox = obj_ang_vel[:, 0:1]
        w_oy = obj_ang_vel[:, 1:2]
        w_oz = obj_ang_vel[:, 2:3]

        # xyzw
        obj_quat = torch.zeros(
            (env_ids.numel(), 4), dtype=torch.float, device=self.device
        )
        obj_quat[:, 3] = 1

        # # {
        # #     "obj_init_pos": np.array([1.66807288, -0.87529668, 1.86571429]),
        # #     "obj_init_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        # #     "obj_init_lin_vel": np.array([-2.2554125, 1.85359417, 0.94586492]),
        # #     "obj_init_ang_vel": np.array([6.69787103, -9.90443602, 10.04414543]),
        # #     "obj_goal_pos": np.array([0.4019805, 0.16523203, 0.85100875]),
        # # }
        # x_o[:], y_o[:], z_o[:] = 1.66807288, -0.87529668, 1.86571429
        # # obj_quat[:] = [0.0, 0.0, 0.0, 1.0]
        # v_xo[:], v_yo[:], v_zo[:] = -2.2554125, 1.85359417, 0.94586492
        # w_ox[:], w_oy[:], w_oz[:] = 0.74854196, 0.46948389, -0.71742531
        # x_f[:], y_f[:], z_f[:] = 0.4019805, 0.16523203, 0.85100875

        obj_init_state = torch.cat(
            [
                x_o,
                y_o,
                z_o,
                obj_quat,
                v_xo,
                v_yo,
                v_zo,
                w_ox,
                w_oy,
                w_oz,
            ],
            dim=1,
        )

        obj_goal_pos = torch.cat([x_f, y_f, z_f], dim=1)

        return obj_init_state, obj_goal_pos

    def _reset_root_states(self, env_ids, target_root_states=None):
        self.simulator.robot_root_states[env_ids] = self.base_init_state
        self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]

        obj_init_state, obj_goal_pos = self.random_toss(env_ids)
        obj_init_state[:, :3] += self.env_origins[env_ids]
        self.obj_init_state[env_ids] = obj_init_state
        self.obj_goal_pos[env_ids] = obj_goal_pos

        throw_wait_s = torch_rand_float(
            0.1, 0.5, (len(env_ids), 1), self.device
        ).squeeze(1)
        self.throw_wait_steps[env_ids] = (throw_wait_s / self.dt).to(torch.long)

        self.simulator.obj_root_states[env_ids, :3] = self.env_origins[env_ids].clone()
        self.simulator.obj_root_states[env_ids, 2] = 100.0
        self.simulator.obj_root_states[env_ids, 7:] = 0.0
        self.obj_target_pos[env_ids] = 0.0

    def _sample_interpolate_arm_dof(self, ordered_arm_inits, n_envs):
        arm_init_idx_0 = torch.randint(
            0,
            ordered_arm_inits.shape[0] - 1,
            (n_envs,),
            device=self.device,
        )
        arm_init_idx_1 = arm_init_idx_0 + 1
        alpha = torch.rand((n_envs, 1), dtype=torch.float, device=self.device)
        return (
            alpha * ordered_arm_inits[arm_init_idx_0]
            + (1 - alpha) * ordered_arm_inits[arm_init_idx_1]
        )

    def _reset_dofs(self, env_ids, target_state=None):
        if self.config.robot.init_state.use_manual_arm_inits:

            if self.config.env.toss_type == "underhand":
                arm_inits = self._sample_interpolate_arm_dof(
                    self.ordered_underhand_arm_inits, len(env_ids)
                )
            elif self.config.env.toss_type == "overhand":
                arm_inits = self._sample_interpolate_arm_dof(
                    self.ordered_overhand_arm_inits, len(env_ids)
                )
            elif self.config.env.toss_type == "both":
                n_under = self.is_underhand[env_ids].sum()
                n_over = len(env_ids) - n_under

                uarm_inits = self._sample_interpolate_arm_dof(
                    self.ordered_underhand_arm_inits, n_under
                )
                oarm_inits = self._sample_interpolate_arm_dof(
                    self.ordered_overhand_arm_inits, n_over
                )

                arm_inits = torch.empty_like(
                    self.simulator.dof_pos[env_ids[:, None], self.arm_joint_indices]
                )
                arm_inits[self.is_underhand[env_ids]] = uarm_inits
                arm_inits[self.is_overhand[env_ids]] = oarm_inits
            else:
                raise NotImplementedError()

            init_dof_pos = self.default_dof_pos.repeat(len(env_ids), 1)
            init_dof_pos[:, self.arm_joint_indices] = arm_inits

            if self.config.domain_rand.randomize_initial_joint_pos:
                init_dof_pos += torch_rand_float(
                    self.config.domain_rand.initial_joint_pos_offset[0],
                    self.config.domain_rand.initial_joint_pos_offset[1],
                    (len(env_ids), self.num_dof),
                    device=self.device,
                )

            dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
            dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
            self.simulator.dof_pos[env_ids] = torch.clip(
                init_dof_pos, dof_lower, dof_upper
            )
            self.simulator.dof_vel[env_ids] = 0.0
        else:
            super()._reset_dofs(env_ids)

        if self.config.robot.control.arm_control_mode == "target_delta_pos":
            self.target_dof_pos[env_ids] = self.simulator.dof_pos[
                env_ids[:, None], self.arm_joint_indices
            ]
        elif self.config.robot.control.arm_control_mode == "ik":
            curr_dof_pos_env_ids = self.simulator.dof_pos[
                env_ids[:, None], self.arm_joint_indices
            ].clone()
            eef_pos, eef_quat = self.kinematics.forward_kinematics(curr_dof_pos_env_ids)
            curr_eef_pose_env_ids = torch.cat([eef_pos, eef_quat], dim=1)
            self.target_eef_pose[env_ids] = curr_eef_pose_env_ids
            self.prev_target_eef_pose[env_ids] = curr_eef_pose_env_ids
            self._ik_target_joint_angles[env_ids] = curr_dof_pos_env_ids
            self._prev_ik_target_joint_angles[env_ids] = curr_dof_pos_env_ids
            self._prev_prev_ik_target_joint_angles[env_ids] = curr_dof_pos_env_ids

    def _episodic_domain_randomization(self, env_ids):
        super()._episodic_domain_randomization(env_ids)
        if self.config.domain_rand.randomize_cam_pos_bias:
            self.cam_pos_bias[env_ids] = torch_rand_float(
                -self.config.domain_rand.cam_pos_bias,
                self.config.domain_rand.cam_pos_bias,
                (len(env_ids), 3),
                device=self.device,
            )

        if self.config.domain_rand.randomize_obj_latency:
            self.obj_pos_queue[env_ids] *= 0.0
            self.obj_latency_idx[env_ids] = torch.randint(
                self.config.domain_rand.obj_latency_range[0],
                self.config.domain_rand.obj_latency_range[1] + 1,
                (len(env_ids),),
                device=self.device,
                requires_grad=False,
            )
            self.log_dict["obj_latency_idx"] = self.obj_latency_idx

        if self._camera_obs and self.config.domain_rand.randomize_cam_latency:
            self.cam_pos_queue[env_ids] *= 0.0
            self.cam_latency_idx[env_ids] = torch.randint(
                self.config.domain_rand.cam_latency_range[0],
                self.config.domain_rand.cam_latency_range[1] + 1,
                (len(env_ids),),
                device=self.device,
                requires_grad=False,
            )
            self.log_dict["cam_latency_idx"] = self.cam_latency_idx

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        assert target_buf is None
        super()._reset_buffers_callback(env_ids, target_buf)
        self.obj_stability_counter[env_ids] = 0

    def control_ik(self, dpose, robot_dof_pos, damping=0.05, only_pos=False):
        j_eef = self.kinematics._compute_jac(robot_dof_pos, eps=1e-3)

        if j_eef is None:
            raise ValueError(f"{j_eef=}")
            return None

        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping**2)
        if only_pos:
            u = (
                (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda))[:, :, :3]
                @ dpose[:, :3].unsqueeze(-1)
            ).squeeze()
        else:
            u = (
                j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose.unsqueeze(-1)
            ).squeeze()
        return u

    def _compute_arm_actions(self, arm_actions: torch.Tensor):
        if self.config.robot.control.arm_control_mode == "pos":
            return arm_actions

        if self.config.robot.control.arm_control_mode == "ik":
            self.prev_target_eef_pose[:] = self.target_eef_pose[:]
            self._prev_prev_ik_target_joint_angles[:] = (
                self._prev_ik_target_joint_angles[:]
            )
            self._prev_ik_target_joint_angles[:] = self._ik_target_joint_angles[:]

            pos_delta = arm_actions[:, :3] / self.ik_config.delta_pos_action_scale
            orn_rpy_delta = arm_actions[:, 3:] / self.ik_config.delta_rpy_action_scale

            # pos_delta = arm_actions[:, :3] * 0
            # orn_rpy_delta = arm_actions[:, 3:] * 0

            # for evt in self.simulator.gym.query_viewer_action_events(
            #     self.simulator.viewer
            # ):
            #     if evt.value <= 0:
            #         continue  # only act on key-down

            #     if evt.action == "inc_x":
            #         pos_delta[:, 0] = 0.1
            #     elif evt.action == "inc_y":
            #         pos_delta[:, 1] = 0.1
            #     elif evt.action == "inc_z":
            #         pos_delta[:, 2] = 0.1
            #     elif evt.action == "dec_x":
            #         pos_delta[:, 0] = -0.1
            #     elif evt.action == "dec_y":
            #         pos_delta[:, 1] = -0.1
            #     elif evt.action == "dec_z":
            #         pos_delta[:, 2] = -0.1

            orn_quat_delta = quat_from_euler_xyz(
                orn_rpy_delta[:, 0], orn_rpy_delta[:, 1], orn_rpy_delta[:, 2]
            )

            self.target_eef_pose[:, :3] = torch.clip(
                self.target_eef_pose[:, :3] + pos_delta,
                min=self.eef_pos_min,
                max=self.eef_pos_max,
            )
            self.target_eef_pose[:, 3:] = quat_mul(
                orn_quat_delta, self.target_eef_pose[:, 3:]
            )
            if self.ik_config.target_eef_pose_ema is not None:
                ema = self.ik_config.target_eef_pose_ema
                self.target_eef_pose[:, :3] = (
                    ema * self.prev_target_eef_pose[:, :3]
                    + (1.0 - ema) * self.target_eef_pose[:, :3]
                )
                self.target_eef_pose[:, 3:] = quat_weighted_average(
                    self.prev_target_eef_pose[:, 3:],
                    self.target_eef_pose[:, 3:],
                    ema,
                    1.0 - ema,
                )

            self._ik_dof_pos_arm_noisy = self.simulator.dof_pos[
                :, self.arm_joint_indices
            ].clone()
            if self.ik_config.noised_dof_pos:
                self._ik_dof_pos_arm_noisy += torch_rand_float(
                    -self._ik_dof_pos_noise_scale,
                    self._ik_dof_pos_noise_scale,
                    self._ik_dof_pos_arm_noisy.shape,
                    self.device,
                )

            eef_pos, eef_quat = self.kinematics.forward_kinematics(
                self._ik_dof_pos_arm_noisy
            )

            ik_pos_delta = self.target_eef_pose[:, :3] - eef_pos
            ik_orn_delta = orientation_error(self.target_eef_pose[:, 3:], eef_quat)

            if self.ik_config.ik_version == "v1":
                self._ik_target_joint_angles[:] = self.control_ik(
                    torch.cat([ik_pos_delta, ik_orn_delta], dim=1),
                    self._ik_dof_pos_arm_noisy,
                    damping=self.ik_config.damping,
                )
            elif self.ik_config.ik_version == "v2":
                self._ik_target_joint_angles[:], error = (
                    self.kinematics.inverse_kinematics(
                        torch.cat([ik_pos_delta, ik_orn_delta], dim=1),
                        self._ik_dof_pos_arm_noisy,
                        tol=self.ik_config.tol,  # 0.01,
                        max_iters=self.ik_config.max_iters,  # 100,
                        damping=self.ik_config.damping,  # 0.05
                    )
                )
                self.log_dict["ik_error"] = error

            if self.ik_config.joint_target_ema is not None:
                ema = self.ik_config.joint_target_ema
                self._ik_target_joint_angles[:] = (
                    ema * self._prev_ik_target_joint_angles
                    + (1.0 - ema) * self._ik_target_joint_angles
                )

            return torch.clip(
                (
                    self._ik_target_joint_angles
                    - self.default_dof_pos[:, self.arm_joint_indices]
                )
                / self.action_scale,
                self._act_limit_low[self.arm_joint_indices],
                self._act_limit_high[self.arm_joint_indices],
            )

        if (
            self.config.robot.control.target_delta_pos_controller_config.normalize_action_space
        ):
            action_lim_scale = (
                self.config.robot.control.target_delta_pos_controller_config.action_lim_scale
            )
            action_clip_value = (
                self.config.robot.control.target_delta_pos_controller_config.action_clip_value
            )
            arm_actions = arm_actions / action_lim_scale * action_clip_value

        self.target_dof_pos[:] = torch.clip(
            self.target_dof_pos + arm_actions,
            self.simulator.hard_dof_pos_limits[self.arm_joint_indices, 0],
            self.simulator.hard_dof_pos_limits[self.arm_joint_indices, 1],
        )

        return torch.clip(
            (self.target_dof_pos - self.default_dof_pos[:, self.arm_joint_indices])
            / self.action_scale,
            self._act_limit_low[self.arm_joint_indices],
            self._act_limit_high[self.arm_joint_indices],
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        actions_clipped = torch.clip(
            actions.to(self.device), self.act_space_low, self.act_space_high
        )

        n_finger_joints = len(self.finger_joint_indices)
        # if self.config.env.termination.processed_actions_lock_arm_post_term:
        #     term_not_reset = (
        #         (self.term_buf & ~self.reset_buf.bool())
        #         .nonzero(as_tuple=False)
        #         .flatten()
        #     )
        #     self.log_dict["term_not_reset"] = self.term_buf & ~self.reset_buf.bool()
        #     if len(term_not_reset) > 0:
        #         actions_clipped[term_not_reset, :n_finger_joints] = self.actions[
        #             term_not_reset[:, None], self.finger_joint_indices
        #         ].clone()
        #         actions_clipped[term_not_reset, n_finger_joints:] = 0.0
        hand_actions = actions_clipped[:, :n_finger_joints]
        arm_actions = actions_clipped[:, n_finger_joints:]
        self.actions[:] = 0.0
        self.actions[:, self.finger_joint_indices] = hand_actions
        self.actions[:, self.arm_joint_indices] = self._compute_arm_actions(
            arm_actions
        )

        # self.log_dict["action_clip_frac"] = (
        #     (actions < self.act_space_low) | (actions > self.act_space_high)
        # ).sum() / self.actions.numel()
        self.log_dict["processed_actions"] = actions_clipped.clone()
        # self.log_dict["noise_level"] = self.config.obs.noise_level

        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = self.actions.clone()
            self.actions_after_delay = self.action_queue[
                torch.arange(self.num_envs), self.action_delay_idx
            ].clone()
        else:
            self.actions_after_delay = self.actions.clone()

    def _setup_robot_body_indices(self):
        super()._setup_robot_body_indices()

        avoid_contact_body_names = self.config.robot.avoid_contact_body_names
        self.avoid_contact_body_indices = torch.zeros(
            len(avoid_contact_body_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(avoid_contact_body_names)):
            self.avoid_contact_body_indices[i] = self.simulator.find_rigid_body_indice(
                avoid_contact_body_names[i]
            )

        ### HEAD ###
        self.head_idx = torch.tensor(
            [self.simulator.find_rigid_body_indice("head_link")],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        ### TORSO ###
        self.torso_idk = torch.tensor(
            [self.simulator.find_rigid_body_indice("torso_link")],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        ### HANDS/ARMS LINKS/JOINTS ###
        used_hand = self.config.robot.used_hand

        # PALM LINK
        hand_names = [
            s
            for s in self.body_names
            if getattr(self.config.robot, f"{used_hand}_hand_name") in s
            and not s.endswith("_fsr")
        ]
        self.hand_indices = torch.zeros(
            len(hand_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(hand_names)):
            self.hand_indices[i] = self.simulator.find_rigid_body_indice(hand_names[i])

        # FINGER LINK/JOINT
        fingertip_names = getattr(self.config.robot, f"{used_hand}_fingertip_names")
        self.fingertip_indices = torch.zeros(
            len(fingertip_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(fingertip_names)):
            self.fingertip_indices[i] = self.simulator.find_rigid_body_indice(
                fingertip_names[i]
            )

        contact_rew_body_names = fingertip_names + [
            f"{self.config.robot.used_hand}_palm"
        ]
        self.contact_rew_indices = torch.zeros(
            len(contact_rew_body_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(contact_rew_body_names)):
            self.contact_rew_indices[i] = self.simulator.find_rigid_body_indice(
                contact_rew_body_names[i]
            )

        finger_joint_names = self.config.robot.finger_joint_names
        self.finger_joint_indices = torch.zeros(
            len(finger_joint_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(finger_joint_names)):
            self.finger_joint_indices[i] = self.dof_names.index(finger_joint_names[i])

        # ARM LINK/JOINT
        arm_link_names = getattr(self.config.robot, f"{used_hand}_arm_link_names")
        self.arm_link_indices = torch.zeros(
            len(arm_link_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(arm_link_names)):
            self.arm_link_indices[i] = self.simulator.find_rigid_body_indice(
                arm_link_names[i]
            )

        arm_joint_names = getattr(self.config.robot, f"{used_hand}_arm_joint_names")
        self.arm_joint_indices = torch.zeros(
            len(arm_joint_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(arm_joint_names)):
            self.arm_joint_indices[i] = self.dof_names.index(arm_joint_names[i])

        # SHOULDER JOINT
        shoulder_joint_names = getattr(
            self.config.robot, f"{used_hand}_shoulder_joint_names"
        )
        self.shoulder_joint_indices = torch.zeros(
            len(shoulder_joint_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(shoulder_joint_names)):
            self.shoulder_joint_indices[i] = self.dof_names.index(
                shoulder_joint_names[i]
            )

        # WRIST JOINT
        wrist_joint_names = getattr(self.config.robot, f"{used_hand}_wrist_joint_names")
        self.wrist_joint_indices = torch.zeros(
            len(wrist_joint_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(wrist_joint_names)):
            self.wrist_joint_indices[i] = self.dof_names.index(wrist_joint_names[i])

        # EXTRAS
        self.finger_arm_joint_indices = torch.cat(
            [self.finger_joint_indices, self.arm_joint_indices]
        )
        self.arm_joint_names = getattr(
            self.config.robot, f"{used_hand}_arm_joint_names"
        )
        self.arm_link_names = getattr(self.config.robot, f"{used_hand}_arm_link_names")

        if self.config.robot.init_state.use_manual_arm_inits:

            def setup_ordered_arm_inits(ordered_arm_init_dict):
                ordered_arm_inits = torch.zeros(
                    (len(ordered_arm_init_dict), len(self.arm_joint_names)),
                    dtype=torch.float,
                    device=self.device,
                )

                for arm_init_num in range(len(ordered_arm_init_dict)):
                    for i, arm_joint_name in enumerate(self.arm_joint_names):
                        ordered_arm_inits[arm_init_num, i] = ordered_arm_init_dict[
                            arm_init_num
                        ][arm_joint_name]

                return ordered_arm_inits

            self.ordered_underhand_arm_inits = setup_ordered_arm_inits(
                self.config.robot.init_state.ordered_underhand_arm_inits
            )
            self.ordered_overhand_arm_inits = setup_ordered_arm_inits(
                self.config.robot.init_state.ordered_overhand_arm_inits
            )

    ### ENV PROPERTIES ###

    def render(self, sync_frame_time=True):
        render_images = super().render(sync_frame_time)
        if self._camera_obs and self.render_cameras:
            obj_pos = (
                # self._obj_pos_wrt_cam_clean
                # self._obj_pos_wrt_cam_seg_after_delay
                # if self._real_camera_obs
                # else self._obj_pos_wrt_cam_noisy_after_delay
                self._obj_pos_wrt_cam_noisy_after_delay
            )
            x_cam = -obj_pos[:, 1].clone()
            y_cam = -obj_pos[:, 2].clone()
            z_cam = obj_pos[:, 0].clone()
            u = (
                (self.config.camera.fx * (x_cam / z_cam) + self.config.camera.cx)
                .to(torch.long)
                .clip(min=0, max=self.config.camera.width - 1)
            )
            v = (
                (self.config.camera.fy * (y_cam / z_cam) + self.config.camera.cy)
                .to(torch.long)
                .clip(min=0, max=self.config.camera.height - 1)
            )

            color = torch.full(
                (self.num_envs, 3),
                0,
                dtype=render_images.dtype,
                device=self.device,
            )
            color[:, 1:] = 200
            # color = torch.full(
            #     (self.num_envs, 3),
            #     0,
            #     dtype=render_images.dtype,
            #     device=self.device,
            # )
            # color[torch.arange(self.num_envs), self.action_delay_idx] += 100
            # color[torch.arange(self.num_envs), self._fake_cam_latency_idx] += 100

            k = 7

            H = self.config.camera.height
            W = self.config.camera.width
            xs = torch.arange(W, device=self.device)
            ys = torch.arange(H, device=self.device)
            X = xs[None, :].expand(H, W)
            Y = ys[:, None].expand(H, W)

            dx = X[None, :, :].float() - u[:, None, None].float()
            dy = Y[None, :, :].float() - v[:, None, None].float()
            mask = (dx * dx + dy * dy) <= (k * k)

            if render_images.ndim == 4:
                color_full = color[:, None, None, :].expand(-1, H, W, -1)
                render_images[mask] = color_full[mask]
            else:
                render_images[mask] = 255
            return render_images
        return render_images

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        self.throw_wait_steps -= 1

        self.obj_thrown[:] = self.throw_wait_steps < 0
        self.wait_for_throw[:] = self.throw_wait_steps > 0
        start_throw = self.throw_wait_steps == 0
        if start_throw.any():
            self.need_to_refresh_envs[start_throw] = True
            self.simulator.obj_root_states[start_throw] = self.obj_init_state[
                start_throw
            ].clone()
            self.obj_target_pos[start_throw] = self.obj_goal_pos[start_throw].clone()

        self._update_attrs()

    def _update_attrs(self, env_ids=None, resetting=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.obj_thrown[env_ids] = self.throw_wait_steps[env_ids] < 0
        self.wait_for_throw[env_ids] = self.throw_wait_steps[env_ids] > 0

        # obj
        self.obj_pos[env_ids] = (
            self.simulator.obj_root_states[env_ids, :3].clone()
            - self.env_origins[env_ids]
        )
        self.obj_quat[env_ids] = self.simulator.obj_root_states[env_ids, 3:7].clone()
        self.obj_lin_vel[env_ids] = self.simulator.obj_root_states[
            env_ids, 7:10
        ].clone()
        self.obj_ang_vel[env_ids] = self.simulator.obj_root_states[
            env_ids, 10:13
        ].clone()

        if self.wait_for_throw[env_ids].any():
            wait_for_throw_env_ids = torch.zeros_like(self.wait_for_throw)
            wait_for_throw_env_ids[env_ids] = self.wait_for_throw[env_ids]
            wait_for_throw_env_ids = wait_for_throw_env_ids.nonzero(
                as_tuple=False
            ).flatten()
            self.obj_pos[wait_for_throw_env_ids] = (
                self.obj_init_state[wait_for_throw_env_ids, 0:3].clone()
                - self.env_origins[wait_for_throw_env_ids]
            )
            self.obj_quat[wait_for_throw_env_ids] = self.obj_init_state[
                wait_for_throw_env_ids, 3:7
            ].clone()
            self.obj_lin_vel[wait_for_throw_env_ids] = 0.0
            self.obj_ang_vel[wait_for_throw_env_ids] = 0.0

        # fingertip
        finger_link_pos = (
            self.simulator._rigid_body_pos[
                env_ids[:, None], self.fingertip_indices
            ].clone()
            - self.env_origins[env_ids].unsqueeze(1)
        ).view(len(env_ids) * len(self.fingertip_indices), 3)
        finger_link_quat = (
            self.simulator._rigid_body_rot[env_ids[:, None], self.fingertip_indices]
            .clone()
            .view(len(env_ids) * len(self.fingertip_indices), 4)
        )
        self.fingertip_pos[env_ids] = tf_apply(
            finger_link_quat,
            finger_link_pos,
            self.fingertip_shift[env_ids].view(
                len(env_ids) * len(self.fingertip_indices), 3
            ),
        ).reshape(len(env_ids), len(self.fingertip_indices), 3)

        # head
        self.head_pos[env_ids] = (
            self.simulator._rigid_body_pos[env_ids[:, None], self.head_idx]
            .squeeze(1)
            .clone()
            - self.env_origins[env_ids]
        )
        self.head_quat[env_ids] = (
            self.simulator._rigid_body_rot[env_ids[:, None], self.head_idx]
            .squeeze(1)
            .clone()
        )
        self.head_lin_vel[env_ids] = (
            self.simulator._rigid_body_vel[env_ids[:, None], self.head_idx]
            .squeeze(1)
            .clone()
        )
        self.head_ang_vel[env_ids] = (
            self.simulator._rigid_body_ang_vel[env_ids[:, None], self.head_idx]
            .squeeze(1)
            .clone()
        )

        # torso
        self.torso_pos[env_ids] = (
            self.simulator._rigid_body_pos[env_ids[:, None], self.torso_idk]
            .squeeze(1)
            .clone()
            - self.env_origins[env_ids]
        )
        self.torso_quat[env_ids] = (
            self.simulator._rigid_body_rot[env_ids[:, None], self.torso_idk]
            .squeeze(1)
            .clone()
        )

        # hand
        if self.last_hand_pos is None:
            self.hand_pos[env_ids] = (
                self.simulator._rigid_body_pos[env_ids[:, None], self.hand_indices]
                .squeeze(1)
                .clone()
                - self.env_origins[env_ids]
            )
            self.last_hand_pos = self.hand_pos.clone()
            self.last_last_hand_pos = self.hand_pos.clone()
        else:
            if not resetting:
                self.last_last_hand_pos[env_ids] = self.last_hand_pos[env_ids]
                self.last_hand_pos[env_ids] = self.hand_pos[env_ids]
            self.hand_pos[env_ids] = (
                self.simulator._rigid_body_pos[env_ids[:, None], self.hand_indices]
                .squeeze(1)
                .clone()
                - self.env_origins[env_ids]
            )
            if resetting:
                self.last_last_hand_pos[env_ids] = self.hand_pos[env_ids]
                self.last_hand_pos[env_ids] = self.hand_pos[env_ids]
        self.hand_quat[env_ids] = (
            self.simulator._rigid_body_rot[env_ids[:, None], self.hand_indices]
            .squeeze(1)
            .clone()
        )
        self.hand_lin_vel[env_ids] = (
            self.simulator._rigid_body_vel[env_ids[:, None], self.hand_indices]
            .squeeze(1)
            .clone()
        )
        self.hand_ang_vel[env_ids] = (
            self.simulator._rigid_body_ang_vel[env_ids[:, None], self.hand_indices]
            .squeeze(1)
            .clone()
        )
        self.hand_projected_gravity[env_ids] = quat_rotate_inverse(
            self.hand_quat[env_ids], self.gravity_vec[env_ids]
        )
        self.hand_forward[env_ids] = quat_rotate_inverse(
            self.hand_quat[env_ids], self.forward_vec[env_ids]
        )

        # hand goal
        self.hand_goal_pos[env_ids] = tf_apply(
            self.hand_quat[env_ids],
            self.hand_pos[env_ids],
            self.hand_goal_point_in_hand[env_ids],
        )

        # hand dists
        self.hand_goal_obj_dist[env_ids] = torch.norm(
            self.hand_goal_pos[env_ids] - self.obj_pos[env_ids], dim=1
        )

        # camera
        self.camera_quat[env_ids], self.camera_pos[env_ids] = tf_combine(
            self.torso_quat[env_ids],
            self.torso_pos[env_ids],
            self.cam_local_quat[env_ids],
            self.cam_local_pos[env_ids],
        )
        self.camera_lin_vel[env_ids] = self.head_lin_vel[env_ids] + torch.cross(
            self.head_ang_vel[env_ids],
            quat_rotate(self.head_quat[env_ids], self.cam_local_pos[env_ids]),
            dim=-1,
        )
        self.camera_ang_vel[env_ids] = self.head_ang_vel[env_ids]

        if self._camera_obs:
            # obj wrt cam (+ custom noise)
            cam_quat_inv, cam_pos_inv = tf_inverse(
                self.camera_quat[env_ids], self.camera_pos[env_ids]
            )
            obj_pos_cam_clean: torch.Tensor = tf_apply(
                cam_quat_inv, cam_pos_inv, self.obj_pos[env_ids]
            )
            obj_quat_cam_clean = quat_mul(cam_quat_inv, self.obj_quat[env_ids])

            # NOTE (arth): isaacgym has x-forward, y-left, z-up
            # cameras w/ 0px,0px top-right use x-right, y-down, z-forward
            x_cam = -obj_pos_cam_clean[:, 1].clone()
            y_cam = -obj_pos_cam_clean[:, 2].clone()
            z_cam = obj_pos_cam_clean[:, 0].clone()
            self.u_obj[env_ids] = (
                self.config.camera.fx * (x_cam / z_cam) + self.config.camera.cx
            )
            self.v_obj[env_ids] = (
                self.config.camera.fy * (y_cam / z_cam) + self.config.camera.cy
            )

            valid_obj = z_cam > 0.0
            # NOTE (arth): we will mask out objs not in frame later
            self.obj_in_frame[env_ids] = valid_obj & (
                (self.u_obj[env_ids] >= 0.0)
                & (self.u_obj[env_ids] < self.config.camera.width)
                & (self.v_obj[env_ids] >= 0.0)
                & (self.v_obj[env_ids] < self.config.camera.height)
            )
            obj_near_cam_edge = (
                (self.u_obj[env_ids] < self.config.camera.edge_margin_px)
                | (
                    self.u_obj[env_ids]
                    >= (self.config.camera.width - self.config.camera.edge_margin_px)
                )
                | (self.v_obj[env_ids] < self.config.camera.edge_margin_px)
                | (
                    self.v_obj[env_ids]
                    >= (self.config.camera.height - self.config.camera.edge_margin_px)
                )
            ) & self.obj_in_frame[env_ids]
            obj_near_hand = (
                torch.norm(self.obj_pos[env_ids] - self.hand_pos[env_ids], dim=1) < 0.2
            )

            # Region: 0 clear, 1 edge, 2 near hand
            region_obj = torch.zeros_like(z_cam, dtype=torch.long)
            region_obj[obj_near_cam_edge] = 1
            region_obj[obj_near_hand] = 2

            base_sigma_obj = torch.empty(
                (len(env_ids), 3), device=self.device, dtype=obj_pos_cam_clean.dtype
            )
            base_sigma_obj[:] = self.config.camera.noise_clear
            base_sigma_obj[region_obj == 1] = self.config.camera.noise_edge
            base_sigma_obj[region_obj == 2] = self.config.camera.noise_near_hand

            # add noise for cam dist/obj speed
            camera_dist = torch.norm(
                self.obj_pos[env_ids] - self.camera_pos[env_ids], dim=1
            )
            obj_speed = torch.norm(self.obj_lin_vel[env_ids], dim=1)
            sigma_obj = (
                base_sigma_obj
                + camera_dist.view(-1, 1) * self.config.camera.noise_distance_factor
                + obj_speed.view(-1, 1) * self.config.camera.noise_velocity_factor
            )

            obj_pos_cam_noisy = (
                obj_pos_cam_clean
                + (2 * torch.rand_like(obj_pos_cam_clean) - 1) * sigma_obj
            )
            if self.config.domain_rand.randomize_cam_pos_bias:
                obj_pos_cam_noisy += self.cam_pos_bias[env_ids]
            obj_pos_cam_noisy = (
                obj_pos_cam_noisy
                * self.obj_in_frame[env_ids].unsqueeze(1)
            )
            if self.config.camera.no_pred_prob > 0.0:
                obj_pos_cam_noisy *= (
                    torch.rand_like(camera_dist) >= self.config.camera.no_pred_prob
                ).unsqueeze(1)

            self._obj_pos_wrt_cam_clean[env_ids] = obj_pos_cam_clean
            self._obj_pos_wrt_cam_noisy[env_ids] = obj_pos_cam_noisy
            self._obj_quat_wrt_cam_clean[env_ids] = obj_quat_cam_clean

        # events
        self.obj_near_hand_goal[env_ids] = (
            self.hand_goal_obj_dist[env_ids]
            < self.config.env.projectiles.obj_near_hand_goal_dist_threshold
        )
        self.obj_vel_mag[env_ids] = torch.norm(self.obj_lin_vel[env_ids], dim=1)
        self.obj_static[env_ids] = (
            self.obj_vel_mag[env_ids] < self.config.env.projectiles.obj_static_threshold
        )
        self.obj_near_hand_goal_obj_static[env_ids] = (
            self.obj_static[env_ids] & self.obj_near_hand_goal[env_ids]
        )

        self.obj_in_hand_goal[env_ids] = (
            self.hand_goal_obj_dist[env_ids]
            < self.config.env.projectiles.obj_in_hand_goal_dist_threshold
        )
        self.hand_static[env_ids] = torch.all(
            self.hand_lin_vel[env_ids].abs()
            < self.config.env.projectiles.hand_static_threshold,
            dim=1,
        )
        self.obj_in_hand_goal_static[env_ids] = (
            self.hand_static[env_ids] & self.obj_in_hand_goal[env_ids]
        )
        if not resetting:
            self.obj_stability_counter[env_ids] *= self.obj_in_hand_goal_static[env_ids]
            self.obj_stability_counter[env_ids] += self.obj_in_hand_goal_static[env_ids]

    ### TERMINATION ###

    def _update_reset_buf(self):
        super()._update_reset_buf()

        if self.config.env.termination.terminate_by_obj_in_hand_goal:
            term_obj_in_hand_goal = self.obj_in_hand_goal
            self.term_buf |= term_obj_in_hand_goal
            # self.log_dict["term_obj_in_hand_goal"] = term_obj_in_hand_goal

        if self.config.env.termination.terminate_by_obj_in_hand_goal_static:
            term_obj_in_hand_goal_static = self.obj_in_hand_goal_static
            self.term_buf |= term_obj_in_hand_goal_static
            # self.log_dict["term_obj_in_hand_goal_static"] = term_obj_in_hand_goal_static

        if self.config.env.termination.terminate_by_obj_near_hand_goal_obj_static:
            term_obj_near_hand_goal_obj_static = self.obj_near_hand_goal_obj_static
            self.term_buf |= term_obj_near_hand_goal_obj_static
            # self.log_dict["term_obj_near_hand_goal_obj_static"] = (
            # term_obj_near_hand_goal_obj_static
            # )

        if self.config.env.termination.terminate_by_hand_goal_obj_dist:
            term_hand_goal_obj_dist = (
                self.hand_goal_obj_dist
                >= self.config.env.termination.scales.termination_hand_goal_obj_dist
            ) & ~self.wait_for_throw
            self.term_buf |= term_hand_goal_obj_dist
            # self.log_dict["term_hand_goal_obj_dist"] = term_hand_goal_obj_dist

        if self.config.env.termination.terminate_by_obj_height:
            term_obj_height = (
                self.obj_pos[:, 2]
                <= self.config.env.termination.scales.termination_obj_height
            ) & ~self.wait_for_throw
            self.term_buf |= term_obj_height
            # self.log_dict["term_obj_height"] = term_obj_height

    ### REWARDS ###

    def _compute_reward(self):
        retval = super()._compute_reward()

        self.extras.pop("final_info", None)
        self.extras.pop("_final_info", None)
        self.extras.pop("success", None)

        success = self.obj_near_hand_goal_obj_static.clone()
        reset = self.reset_buf.clone()
        self.extras["success"] = success
        self.extras["_final_info"] = reset
        self.extras["final_info"] = {"success": success}

        if self.config.env.toss_type == "both":
            self.extras["underhand"] = self.is_underhand.clone()
            self.extras["overhand"] = self.is_overhand.clone()

        return retval

    ### task rewards ###

    def _reward_obj_stability(self):
        return self.obj_stability_counter.float() / 50.0

    def _reward_finger_palm_touch_sparse(self):
        # NOTE (arth): pairwise not supported by isaacgym, but generally self-collisions are a) rare and b) low magnitude
        contact_forces = self.simulator.contact_forces[
            :, self.contact_rew_indices
        ].norm(dim=2)
        force_threshold = 0.4
        return torch.any(contact_forces > force_threshold, dim=1)

    def _reward_finger_palm_touch(self):
        # NOTE (arth): pairwise not supported by isaacgym, but generally self-collisions are a) rare and b) low magnitude
        contact_forces = self.simulator.contact_forces[
            :, self.contact_rew_indices
        ].norm(dim=2)
        force_threshold = 0.4
        binary = contact_forces > force_threshold
        finger_contacts = binary[:, :5].sum(dim=1)
        palm_contact = binary[:, 5]

        # Base reward for any contact
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Progressive finger contact
        reward += (finger_contacts >= 1).float() * 0.1
        reward += (finger_contacts >= 2).float() * 0.2
        reward += (finger_contacts >= 3).float() * 0.4

        # Bonus for palm contact when fingers engaged
        reward += ((finger_contacts >= 2) & palm_contact).float() * 0.3
        return reward

    def _reward_obj_hand_goal_dist(self):
        # self.log_dict["hand_goal_obj_dist"] = self.hand_goal_obj_dist
        return torch.exp(-self.hand_goal_obj_dist * 2.0)

    def _reward_obj_fingertip_dist(self):
        fingertip_dist = torch.norm(
            self.fingertip_pos - self.obj_pos[:, None, :], dim=2
        )
        max_fingertip_dist = fingertip_dist.max(dim=1).values
        # self.log_dict["max_fingertip_dist"] = max_fingertip_dist
        # self.log_dict["fingertip_dist_x"] = fingertip_dist[:, 0]
        # self.log_dict["fingertip_dist_y"] = fingertip_dist[:, 1]
        # self.log_dict["fingertip_dist_z"] = fingertip_dist[:, 2]
        return torch.exp(-max_fingertip_dist * 20.0)  # - 0.05

    def _reward_approx_obj_surface_fingertip_dist(self):
        fingertip_dist = torch.norm(
            self.fingertip_pos - self.obj_pos[:, None, :], dim=2
        )
        obj_radius = self.simulator.projectile_dims_scaled.min(dim=1).values / 2
        max_fingertip_dist = fingertip_dist.max(dim=1).values
        approx_obj_surface_fingertip_dist = max_fingertip_dist - obj_radius
        # self.log_dict["obj_radius"] = obj_radius
        # self.log_dict["max_fingertip_dist"] = max_fingertip_dist
        # self.log_dict["approx_obj_surface_fingertip_dist"] = (
        #     approx_obj_surface_fingertip_dist
        # )
        return torch.exp(
            -approx_obj_surface_fingertip_dist.clip(min=0.0) * 10.0
        )  # - 0.05

    def _reward_obj_height(self):
        # self.log_dict["obj_z"] = self.obj_pos[:, 2]
        min_thresh = self.config.env.projectiles.obj_on_floor_threshold
        max_thresh = self.config.env.projectiles.obj_height_max_threshold
        reward = (
            torch.clip(
                self.obj_pos[:, 2],
                min=min_thresh,
                max=max_thresh,
            )
            - min_thresh
        ) / (max_thresh - min_thresh)
        reward[self.obj_near_hand_goal] = 1.0
        return reward

    def _reward_avoid_contact_body(self):  # penalty
        arm_contact = torch.norm(
            self.simulator.contact_forces[:, self.avoid_contact_body_indices, :], dim=-1
        )  # > 0.1
        reward = torch.mean(arm_contact.float(), dim=-1)
        return reward

    def _reward_avoid_contact_arm(self):  # penalty
        arm_contact = torch.norm(
            self.simulator.contact_forces[:, self.arm_link_indices, :], dim=-1
        )  # > 0.1
        reward = torch.mean(arm_contact.float(), dim=-1)
        return reward

    def _reward_hand_grasp_pose(self):
        grasp_diff = torch.mean(
            torch.abs(
                self.default_grasp_pos[:, self.finger_joint_indices]
                - self.simulator.dof_pos[:, self.finger_joint_indices]
            ),
            dim=1,
        )
        # self.log_dict["grasp_diff"] = grasp_diff
        return self.obj_near_hand_goal.float() * (
            1 - torch.tanh((grasp_diff - 0.6).clip(min=0.0))
        )

    def _reward_hand_default_pos(self):  # penalty
        arm_diff = torch.mean(
            torch.abs(
                self.default_dof_pos[:, self.finger_joint_indices]
                - self.simulator.dof_pos[:, self.finger_joint_indices]
            ),
            dim=1,
        )
        # self.log_dict["arm_diff"] = arm_diff
        return arm_diff

    def _reward_arm_default_pos(self):  # penalty
        if self.config.env.toss_type == "underhand":
            default_dof_pos = self.default_dof_pos[:, self.arm_joint_indices]
        elif self.config.env.toss_type == "overhand":
            default_dof_pos = self.overhand_dof_pos[:, self.arm_joint_indices]
        elif self.config.env.toss_type == "both":
            default_dof_pos = torch.empty_like(
                self.simulator.dof_pos[:, self.arm_joint_indices]
            )
            default_dof_pos[self.under_env_ids] = self.default_dof_pos[
                :, self.arm_joint_indices
            ]
            default_dof_pos[self.over_env_ids] = self.overhand_dof_pos[
                :, self.arm_joint_indices
            ]

        pen = torch.mean(
            torch.abs(
                default_dof_pos - self.simulator.dof_pos[:, self.arm_joint_indices]
            ),
            dim=1,
        )
        if self.config.env.toss_type == "underhand":
            pen[self.wait_for_throw] *= 2
        elif self.config.env.toss_type == "both":
            raise NotImplementedError()
        return pen

    ### sim2real rewards ###
    def _reward_arm_action_rate(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.last_actions[:, self.arm_joint_indices]
                - self.actions[:, self.arm_joint_indices]
            ),
            dim=1,
        )

    def _reward_finger_action_rate(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.last_actions[:, self.finger_joint_indices]
                - self.actions[:, self.finger_joint_indices]
            ),
            dim=1,
        )

    def _reward_arm_smoothness(self):
        # second order smoothness
        return torch.sum(
            torch.square(
                self.actions[:, self.arm_joint_indices]
                - self.last_actions[:, self.arm_joint_indices]
                - self.last_actions[:, self.arm_joint_indices]
                + self.last_last_actions[:, self.arm_joint_indices]
            ),
            dim=1,
        )

    def _reward_finger_smoothness(self):
        # second order smoothness
        return torch.sum(
            torch.square(
                self.actions[:, self.finger_joint_indices]
                - self.last_actions[:, self.finger_joint_indices]
                - self.last_actions[:, self.finger_joint_indices]
                + self.last_last_actions[:, self.finger_joint_indices]
            ),
            dim=1,
        )

    def _reward_joint_power_arm(self):
        # Penalize high power
        return torch.sum(
            torch.abs(self.simulator.dof_vel[:, self.arm_joint_indices])
            * torch.abs(self.torques[:, self.arm_joint_indices]),
            dim=1,
        )

    def _reward_joint_power_hand(self):
        # Penalize high power
        return torch.sum(
            torch.abs(self.simulator.dof_vel[:, self.finger_joint_indices])
            * torch.abs(self.torques[:, self.finger_joint_indices]),
            dim=1,
        )

    def _reward_arm_joint_tracking_error(self):  # penalty
        return torch.sum(
            torch.square(
                self.joint_pos_target[:, self.arm_joint_indices]
                - self.simulator.dof_pos[:, self.arm_joint_indices]
            ),
            dim=-1,
        )

    def _reward_finger_joint_tracking_error(self):  # penalty
        return torch.sum(
            torch.square(
                self.joint_pos_target[:, self.finger_joint_indices]
                - self.simulator.dof_pos[:, self.finger_joint_indices]
            ),
            dim=-1,
        )

    def _reward_dof_vel_arm(self):  # penalty
        # Penalize dof velocities
        return torch.sum(
            torch.square(self.simulator.dof_vel[:, self.arm_joint_indices]),
            dim=1,
        )

    def _reward_dof_vel_hand(self):  # penalty
        # Penalize dof velocities
        return torch.sum(
            torch.square(self.simulator.dof_vel[:, self.finger_joint_indices]),
            dim=1,
        )

    def _reward_dof_acc_arm(self):  # penalty
        # Penalize dof accelerations
        return torch.sum(
            torch.square(
                (
                    self.last_dof_vel[:, self.arm_joint_indices]
                    - self.simulator.dof_vel[:, self.arm_joint_indices]
                )
                / self.dt
            ),
            dim=1,
        )

    def _reward_dof_acc_hand(self):  # penalty
        # Penalize dof accelerations
        return torch.sum(
            torch.square(
                (
                    self.last_dof_vel[:, self.finger_joint_indices]
                    - self.simulator.dof_vel[:, self.finger_joint_indices]
                )
                / self.dt
            ),
            dim=1,
        )

    def _reward_torques_arm(self):  # penalty
        # Penalize torques
        return torch.sum(
            torch.square(
                (
                    self.torques[:, self.arm_joint_indices]
                    / self.p_gains[self.arm_joint_indices].unsqueeze(0)
                )
            ),
            dim=1,
        )

    def _reward_torques_hand(self):  # penalty
        # Penalize torques
        return torch.sum(
            torch.square(
                (
                    self.torques[:, self.finger_joint_indices]
                    / self.p_gains[self.finger_joint_indices].unsqueeze(0)
                )
            ),
            dim=1,
        )

    def _reward_dof_pos_arm_limits(self):  # penalty
        # Penalize dof positions too close to the limit
        out_of_limits = -(
            self.simulator.dof_pos[:, self.arm_joint_indices]
            - self.dof_pos_limits[self.arm_joint_indices, 0]
        ).clip(max=0.0)
        out_of_limits += (
            self.simulator.dof_pos[:, self.arm_joint_indices]
            - self.dof_pos_limits[self.arm_joint_indices, 1]
        ).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_pos_hand_limits(self):  # penalty
        # Penalize dof positions too close to the limit
        out_of_limits = -(
            self.simulator.dof_pos[:, self.finger_joint_indices]
            - self.dof_pos_limits[self.finger_joint_indices, 0]
        ).clip(max=0.0)
        out_of_limits += (
            self.simulator.dof_pos[:, self.finger_joint_indices]
            - self.dof_pos_limits[self.finger_joint_indices, 1]
        ).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_arm_limits(self):  # penalty
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (
                torch.abs(self.simulator.dof_vel[:, self.arm_joint_indices])
                - self.dof_vel_limits[self.arm_joint_indices]
                * self.config.rewards.reward_limit.soft_dof_vel_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_dof_vel_hand_limits(self):  # penalty
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (
                torch.abs(self.simulator.dof_vel[:, self.finger_joint_indices])
                - self.dof_vel_limits[self.finger_joint_indices]
                * self.config.rewards.reward_limit.soft_dof_vel_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_torque_arm_limits(self):  # penalty
        # penalize torques too close to the limit
        return torch.sum(
            (
                torch.abs(self.torques[:, self.arm_joint_indices])
                - self.torque_limits[self.arm_joint_indices]
                * self.config.rewards.reward_limit.soft_torque_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_torque_hand_limits(self):  # penalty
        # penalize torques too close to the limit
        return torch.sum(
            (
                torch.abs(self.torques[:, self.finger_joint_indices])
                - self.torque_limits[self.finger_joint_indices]
                * self.config.rewards.reward_limit.soft_torque_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_action_vanish_arm(self):
        upper_error = torch.clip(
            self.actions[:, self.arm_joint_indices]
            - self.action_max[:, self.arm_joint_indices],
            min=0.0,
        )
        lower_error = torch.clip(
            self.action_min[:, self.arm_joint_indices]
            - self.actions[:, self.arm_joint_indices],
            min=0.0,
        )
        return torch.sum(upper_error + lower_error, dim=-1)

    def _reward_action_vanish_hand(self):
        upper_error = torch.clip(
            self.actions[:, self.finger_joint_indices]
            - self.action_max[:, self.finger_joint_indices],
            min=0.0,
        )
        lower_error = torch.clip(
            self.action_min[:, self.finger_joint_indices]
            - self.actions[:, self.finger_joint_indices],
            min=0.0,
        )
        return torch.sum(upper_error + lower_error, dim=-1)

    def _reward_hand_pos_rate(self):
        pen = torch.norm(self.last_hand_pos - self.hand_pos, dim=1)
        if self.config.env.toss_type == "underhand":
            pen[self.wait_for_throw] *= 2
        elif self.config.env.toss_type == "both":
            raise NotImplementedError()
        return pen

    def _reward_hand_pos_smoothness(self):
        # second order smoothness
        return torch.norm(
            self.hand_pos
            - self.last_hand_pos
            - self.last_hand_pos
            + self.last_last_hand_pos,
            dim=1,
        )

    def _reward_hand_orientation(self):
        if self.config.env.toss_type == "underhand":
            dir_pen = self.hand_projected_gravity[:, :2]
        elif self.config.env.toss_type == "overhand":
            dir_pen = self.hand_forward[:, :2]
        elif self.config.env.toss_type == "both":
            dir_pen = torch.empty_like(self.hand_forward[:, :2])
            dir_pen[self.under_env_ids] = self.hand_projected_gravity[
                self.under_env_ids, :2
            ]
            dir_pen[self.over_env_ids] = self.hand_forward[self.over_env_ids, :2]
        return torch.sum(torch.square(dir_pen), dim=1)

    ### OBSERVATIONS ###

    def _compute_observations(self):
        # NOTE (arth): order is
        #   [1] simulate
        #   [2] update tasks (+ update attrs)
        #   [3] compute termination
        #   [4] compute reward
        #   [5] reset envs
        #   [6] gen obs
        # attrs must be computed before reward, so we can give rewards based on s_{t+1}
        # but since we reset after updating attrs, o_0 for the new episode
        #   will use attrs based s_{t+1} of the last episode, generating incorrect o_0
        # so, we update again when resetting
        if self.reset_buf.any():
            self._update_attrs(
                self.reset_buf.nonzero(as_tuple=False).flatten(), resetting=True
            )
        if self.config.domain_rand.randomize_obj_latency:
            self.obj_pos_queue[:, 1:] = self.obj_pos_queue[:, :-1].clone()
            self.obj_pos_queue[:, 0] = self.obj_pos.clone()
            self.obj_pos_delay[:] = self.obj_pos_queue[
                torch.arange(self.num_envs), self.obj_latency_idx
            ].clone()
        else:
            self.obj_pos_delay[:] = self.obj_pos
        if self._camera_obs:
            if self.config.domain_rand.randomize_cam_latency:
                self.cam_pos_queue[:, 1:] = self.cam_pos_queue[:, :-1].clone()
                self.cam_pos_queue[:, 0] = self._obj_pos_wrt_cam_noisy.clone()
                self._obj_pos_wrt_cam_noisy_after_delay[:] = self.cam_pos_queue[
                    torch.arange(self.num_envs), self.cam_latency_idx
                ].clone()
            else:
                self._obj_pos_wrt_cam_noisy_after_delay[:] = (
                    self._obj_pos_wrt_cam_noisy.clone()
                )
        self.log_dict["obj_scale"] = self.simulator.projectile_scales
        return super()._compute_observations()

    # eef

    def _get_obs_eef_pose_fk_clean(self):
        eef_pos, eef_quat = self.kinematics.forward_kinematics(
            self.simulator.dof_pos[:, self.arm_joint_indices].clone()
        )
        return torch.cat([eef_pos, eef_quat], dim=1)

    # dof

    def _get_obs_dof_pos_finger_arm(self):
        return (
            self.simulator.dof_pos[:, self.finger_arm_joint_indices]
            - self.default_dof_pos[:, self.finger_arm_joint_indices]
        )

    def _get_obs_dof_vel_finger_arm(self):
        return self.simulator.dof_vel[:, self.finger_arm_joint_indices]

    def _get_obs_dof_vel_arm(self):
        return self.simulator.dof_vel[:, self.arm_joint_indices]

    def _get_obs_actions_finger_arm(self):
        return self.actions[:, self.finger_arm_joint_indices]

    # obj

    def _get_obs_obj_scale(self):
        return self.simulator.projectile_scales.unsqueeze(1)

    def _get_obs_obj_dims(self):
        return self.simulator.projectile_dims

    def _get_obs_obj_dims_scaled(self):
        return self.simulator.projectile_dims_scaled

    # wrt cam

    def _get_obs_obj_target_pos_wrt_camera(self):
        cam_quat_inv, cam_pos_inv = tf_inverse(self.camera_quat, self.camera_pos)
        obj_target_pos_wrt_cam = tf_apply(
            cam_quat_inv, cam_pos_inv, self.obj_target_pos
        )
        return obj_target_pos_wrt_cam

    def _get_obs_obj_pose_wrt_camera(self):
        assert self._camera_obs
        pos = self._obj_pos_wrt_cam_clean
        quat = self._obj_quat_wrt_cam_clean
        return torch.cat([pos, quat], dim=1)

    def _get_obs_obj_pos_wrt_camera_clean(self):
        assert self._camera_obs
        return self._obj_pos_wrt_cam_clean

    def _get_obs_obj_pos_wrt_camera_noisy(self):
        assert self._camera_obs
        return self._obj_pos_wrt_cam_noisy_after_delay

    def _get_obs_obj_lin_vel_wrt_camera(self):
        assert self._camera_obs
        return quat_rotate_inverse(
            self.camera_quat, self.obj_lin_vel - self.camera_lin_vel
        )

    def _get_obs_obj_ang_vel_wrt_camera(self):
        return quat_rotate_inverse(
            self.camera_quat, self.obj_ang_vel - self.camera_ang_vel
        )

    # wrt head

    def _get_obs_obj_target_pos_wrt_head(self):
        head_quat_inv, head_pos_inv = tf_inverse(self.head_quat, self.head_pos)
        obj_target_pos_wrt_head = tf_apply(
            head_quat_inv, head_pos_inv, self.obj_target_pos
        )
        return obj_target_pos_wrt_head

    def _get_obs_obj_pose_wrt_head(self):
        head_quat_inv, head_pos_inv = tf_inverse(self.head_quat, self.head_pos)
        obj_pos_in_head = tf_apply(head_quat_inv, head_pos_inv, self.obj_pos_delay)
        obj_quat_in_head = quat_mul(head_quat_inv, self.obj_quat)
        return torch.cat([obj_pos_in_head, obj_quat_in_head], dim=1)

    def _get_obs_obj_pos_wrt_head(self):
        head_quat_inv, head_pos_inv = tf_inverse(self.head_quat, self.head_pos)
        obj_pos_in_head = tf_apply(head_quat_inv, head_pos_inv, self.obj_pos_delay)
        return obj_pos_in_head

    def _get_obs_obj_lin_vel_wrt_head(self):
        return quat_rotate_inverse(self.head_quat, self.obj_lin_vel - self.head_lin_vel)

    def _get_obs_obj_ang_vel_wrt_head(self):
        return quat_rotate_inverse(self.head_quat, self.obj_ang_vel - self.head_ang_vel)

    # wrt hand

    def _get_obs_obj_target_pos_wrt_hand(self):
        hand_quat_inv, hand_pos_inv = tf_inverse(self.hand_quat, self.hand_pos)
        obj_target_pos_wrt_hand = tf_apply(
            hand_quat_inv, hand_pos_inv, self.obj_target_pos
        )
        return obj_target_pos_wrt_hand

    def _get_obs_obj_pose_wrt_hand(self):
        hand_quat_inv, hand_pos_inv = tf_inverse(self.hand_quat, self.hand_pos)
        obj_pos_in_hand = tf_apply(hand_quat_inv, hand_pos_inv, self.obj_pos_delay)
        obj_quat_in_hand = quat_mul(hand_quat_inv, self.obj_quat)
        return torch.cat([obj_pos_in_hand, obj_quat_in_hand], dim=1)

    def _get_obs_obj_lin_vel_wrt_hand(self):
        return quat_rotate_inverse(self.hand_quat, self.obj_lin_vel - self.hand_lin_vel)

    def _get_obs_obj_ang_vel_wrt_hand(self):
        return quat_rotate_inverse(self.hand_quat, self.obj_ang_vel - self.hand_ang_vel)

    # global frame

    def _get_obs_obj_world_target_pos(self):
        return self.obj_target_pos

    def _get_obs_obj_world_pose(self):
        return torch.cat([self.obj_pos_delay, self.obj_quat], dim=1)

    def _get_obs_obj_lin_vel(self):
        return self.obj_lin_vel

    def _get_obs_obj_ang_vel(self):
        return self.obj_ang_vel

    def _get_obs_hand_lin_vel(self):
        return self.hand_lin_vel

    def _get_obs_hand_world_pose(self):
        return torch.cat([self.hand_pos, self.hand_quat], dim=1)
