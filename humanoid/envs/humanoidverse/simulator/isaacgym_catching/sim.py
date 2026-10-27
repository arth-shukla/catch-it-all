import os
import random
import sys
from typing import List, Tuple

import numpy as np
import torch

from isaacgym import gymapi, gymtorch, gymutil

from humanoid import HUMANOID_RESOURCES_DIR
from humanoid.envs.humanoidverse.config.catching.catch_upper import (
    G1CatchUpperLockLegsConfig,
)
from humanoid.envs.humanoidverse.simulator.base import BaseSimulator
from humanoid.envs.humanoidverse.utils.helpers import class_to_dict
from humanoid.envs.humanoidverse.utils.torch_utils import torch_rand_float


# from loguru import logger


class IsaacGymCatchingSimulator(BaseSimulator):
    def __init__(self, config: G1CatchUpperLockLegsConfig, device):
        super().__init__(config, device)
        self.simulator_config = config.simulator
        self.robot_config = config.robot
        self.visualize_viewer = False

    def set_headless(self, headless):
        # call super
        super().set_headless(headless)

    def setup(self):
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(
            str(self.sim_device)
        )
        self.device = self.sim_device

        self.sim_params = self._parse_sim_params(sim_device_type == "cuda")
        self.sim_dt = self.sim_params.dt

        self.physics_engine = gymapi.SIM_PHYSX
        self.gym = gymapi.acquire_gym()

        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        if sim is None:
            # logger.error("*** Failed to create sim")
            quit()
        # logger.info("Creating Sim...", "green")

        self.sim = sim

    def _parse_sim_params(self, use_gpu=False):
        # initialize sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / self.simulator_config.sim.fps
        sim_params.up_axis = gymapi.UP_AXIS_Z
        # sim_params.up_axis = 1  # 0 is y, 1 is z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.num_client_threads = 0

        sim_params.physx.solver_type = self.simulator_config.sim.physx.solver_type
        sim_params.physx.num_position_iterations = (
            self.simulator_config.sim.physx.num_position_iterations
        )
        sim_params.physx.num_velocity_iterations = (
            self.simulator_config.sim.physx.num_velocity_iterations
        )
        sim_params.physx.num_threads = self.simulator_config.sim.physx.num_threads
        sim_params.physx.use_gpu = use_gpu
        sim_params.physx.num_subscenes = 0
        sim_params.physx.max_gpu_contact_pairs = (
            self.simulator_config.sim.physx.max_gpu_contact_pairs
        )
        sim_params.use_gpu_pipeline = use_gpu

        gymutil.parse_sim_config(class_to_dict(self.simulator_config.sim), sim_params)
        return sim_params

    def setup_terrain(self):
        self._create_ground_plane()

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        # logger.info("Creating plane terrain")
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.simulator_config.plane.static_friction
        plane_params.dynamic_friction = self.simulator_config.plane.dynamic_friction
        plane_params.restitution = self.simulator_config.plane.restitution
        self.gym.add_ground(self.sim, plane_params)
        # logger.info("Created plane terrain")

    def load_assets(self):
        asset_root = self.robot_config.asset.asset_root
        asset_file = self.robot_config.asset.urdf_file
        self.robot_asset = self._setup_robot_asset_when_env_created(
            asset_root, asset_file, self.robot_config.asset
        )
        self.num_dof, self.num_bodies, self.dof_names, self.body_names = (
            self._setup_robot_props_when_env_created()
        )

        print(f"{self.dof_names=}")
        print(f"{self.body_names=}")

        # assert if  aligns with config
        assert self.num_dof == len(
            self.robot_config.dof_names
        ), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(
            self.robot_config.body_names
        ), "Number of bodies must be equal to number of body names"
        assert (
            self.dof_names == self.robot_config.dof_names
        ), "DOF names must match the config"
        assert (
            self.body_names == self.robot_config.body_names
        ), "Body names must match the config"

    def _setup_robot_asset_when_env_created(self, asset_root, asset_file, asset_cfg):
        asset_path = os.path.join(asset_root, asset_file)
        gym_asset_root = os.path.dirname(asset_path)
        gym_asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()

        def set_value_if_not_none(prev_value, new_value):
            return new_value if new_value is not None else prev_value

        asset_config_options = [
            "default_dof_drive_mode",
            "collapse_fixed_joints",
            "replace_cylinder_with_capsule",
            "flip_visual_attachments",
            "fix_base_link",
            "density",
            "angular_damping",
            "linear_damping",
            "max_angular_velocity",
            "max_linear_velocity",
            "armature",
            "thickness",
            "disable_gravity",
        ]
        for option in asset_config_options:
            option_value = set_value_if_not_none(
                getattr(asset_options, option), getattr(asset_cfg, option)
            )
            setattr(asset_options, option, option_value)

        self.robot_asset = self.gym.load_asset(
            self.sim, gym_asset_root, gym_asset_file, asset_options
        )
        return self.robot_asset

    def _setup_robot_props_when_env_created(
        self,
    ) -> Tuple[int, int, List[str], List[str]]:
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)

        # save body names from the asset
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)

        return self.num_dof, self.num_bodies, self.dof_names, self.body_names

    def create_envs(self, num_envs, env_origins, base_init_state):
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.num_envs = num_envs
        self.env_config: G1CatchUpperLockLegsConfig = self.config
        self.env_origins = env_origins
        self.base_init_state = base_init_state
        self.envs = []
        self.robot_handles = []

        self.obj_metadata_dict = dict(
            (k, self.env_config.env.projectiles.obj_metadata_dict[k])
            for k in self.env_config.env.projectiles.included_objs
        )

        self.projectile_types = []
        self.projectile_handles = []
        self.projectile_scales = []
        self.projectile_dims = []
        self.projectile_dims_scaled = []
        self.projectile_coms = []

        self.env_handles = []

        self.asset_cache = {}
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            self._build_each_env(i, env_handle)
            self.env_handles.append(env_handle)
        self.asset_cache.clear()

        self.projectile_scales = torch.tensor(
            self.projectile_scales, dtype=torch.float, device=self.device
        )
        self.projectile_dims = torch.tensor(
            self.projectile_dims, dtype=torch.float, device=self.device
        )
        self.projectile_dims_scaled = torch.tensor(
            self.projectile_dims_scaled, dtype=torch.float, device=self.device
        )
        self.projectile_coms = torch.tensor(
            self.projectile_coms, dtype=torch.float, device=self.device
        )

        print(f"{self.projectile_types=}")

        return self.envs, self.robot_handles, self.projectile_handles

    def _build_each_env(self, env_id, env_ptr):
        # Robot
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        pos = self.env_origins[env_id].clone()
        pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=str(self.device)).squeeze(
            1
        )
        start_pose.p = gymapi.Vec3(*pos)

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(
            self.robot_asset
        )
        rigid_shape_props = self._process_rigid_shape_props(
            rigid_shape_props_asset, env_id
        )
        self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)

        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)

        robot_handle = self.gym.create_actor(
            env_ptr,
            self.robot_asset,
            start_pose,
            self.env_config.robot.asset.robot_type,
            env_id,
            self.env_config.robot.asset.self_collisions,
            0,
        )
        self._body_list = self.gym.get_actor_rigid_body_names(env_ptr, robot_handle)
        dof_props = self._process_dof_props(dof_props_asset, env_id)
        self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)
        body_props = self.gym.get_actor_rigid_body_properties(env_ptr, robot_handle)
        body_props = self._process_rigid_body_props(body_props, env_id)
        self.gym.set_actor_rigid_body_properties(
            env_ptr, robot_handle, body_props, recomputeInertia=True
        )
        self.robot_handles.append(robot_handle)

        # Projectile

        projectile_name, projectile_metadata = random.choice(
            list(self.obj_metadata_dict.items())
        )

        if projectile_name in self.asset_cache:
            asset = self.asset_cache[projectile_name]
        else:
            asset_options = gymapi.AssetOptions()
            asset_options.density = 100.0
            asset_options.disable_gravity = False
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.fix_base_link = False
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset = self.gym.load_asset(
                self.sim,
                str(HUMANOID_RESOURCES_DIR / "projectiles"),
                projectile_metadata.asset_path,
                asset_options,
            )
            self.asset_cache[projectile_name] = asset

        pose = gymapi.Transform()
        pos = self.env_origins[env_id].clone()
        pos[2] = 2
        pose.p = gymapi.Vec3(*pos)

        projectile_handle = self.gym.create_actor(
            env_ptr, asset, pose, f"{projectile_name}_{env_id}", env_id, 1
        )

        shape_props = self.gym.get_actor_rigid_shape_properties(
            env_ptr, projectile_handle
        )
        shape_props = self._process_obj_rigid_shape_props(shape_props, env_id)
        self.gym.set_actor_rigid_shape_properties(
            env_ptr, projectile_handle, shape_props
        )

        obj_body_props = self.gym.get_actor_rigid_body_properties(
            env_ptr, projectile_handle
        )
        obj_body_props = self._process_obj_rigid_body_props(obj_body_props, env_id)
        self.gym.set_actor_rigid_body_properties(
            env_ptr, projectile_handle, obj_body_props, recomputeInertia=True
        )

        if self.env_config.domain_rand.randomize_obj_scale:
            if env_id == 0:
                self.obj_scales = torch.zeros(
                    (self.num_envs, 1),
                    device="cpu",
                )
            obj_scale_ranges = self.env_config.domain_rand.obj_scale_ranges
            scale_range = obj_scale_ranges.get(
                projectile_name, obj_scale_ranges["default"]
            )
            scale = random.uniform(*scale_range)
            self.obj_scales[env_id] = scale
        else:
            scale = 1.0
        self.gym.set_actor_scale(env_ptr, projectile_handle, scale)

        self.projectile_types.append(projectile_name)
        self.projectile_handles.append(projectile_handle)
        self.projectile_scales.append(scale)
        self.projectile_dims.append(list(projectile_metadata.dims))
        self.projectile_dims_scaled.append(
            (np.array(list(projectile_metadata.dims)) * scale).tolist()
        )
        self.projectile_coms.append(list(projectile_metadata.center_of_mass))

        #

        self.envs.append(env_ptr)

    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.env_config.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.env_config.domain_rand.friction_range
                self.num_friction_buckets = 250
                bucket_ids = torch.randint(
                    0, self.num_friction_buckets, (self.num_envs, 1)
                )
                self.friction_buckets = torch_rand_float(
                    friction_range[0],
                    friction_range[1],
                    (self.num_friction_buckets, 1),
                    device="cpu",
                )
                self.friction_coeffs = self.friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.env_config.domain_rand.randomize_restitution:
            if env_id == 0:
                # prepare restitution randomization
                restitution_range = self.env_config.domain_rand.restitution_range
                self.num_restitution_buckets = 250
                bucket_ids = torch.randint(
                    0, self.num_restitution_buckets, (self.num_envs, 1)
                )
                self.restitution_buckets = torch_rand_float(
                    restitution_range[0],
                    restitution_range[1],
                    (self.num_restitution_buckets, 1),
                    device="cpu",
                )
                self.restitution_coeffs = self.restitution_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props

    def _process_obj_rigid_shape_props(self, props, env_id):
        if self.env_config.domain_rand.randomize_obj_friction:
            if env_id == 0:
                # prepare friction randomization
                obj_friction_range = self.env_config.domain_rand.obj_friction_range
                self.num_obj_friction_buckets = 250
                bucket_ids = torch.randint(
                    0, self.num_obj_friction_buckets, (self.num_envs, 1)
                )
                self.obj_friction_buckets = torch_rand_float(
                    obj_friction_range[0],
                    obj_friction_range[1],
                    (self.num_obj_friction_buckets, 1),
                    device="cpu",
                )
                self.obj_friction_coeffs = self.obj_friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.obj_friction_coeffs[env_id]

        if self.env_config.domain_rand.randomize_obj_restitution:
            if env_id == 0:
                # prepare restitution randomization
                obj_restitution_range = (
                    self.env_config.domain_rand.obj_restitution_range
                )
                self.num_obj_restitution_buckets = 250
                bucket_ids = torch.randint(
                    0, self.num_obj_restitution_buckets, (self.num_envs, 1)
                )
                self.obj_restitution_buckets = torch_rand_float(
                    obj_restitution_range[0],
                    obj_restitution_range[1],
                    (self.num_obj_restitution_buckets, 1),
                    device="cpu",
                )
                self.obj_restitution_coeffs = self.obj_restitution_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].restitution = self.obj_restitution_coeffs[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.hard_dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )

            self.dof_pos_limits_termination = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )

            for i in range(len(props)):

                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = (
                    m
                    - 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
                )
                self.dof_pos_limits[i, 1] = (
                    m
                    + 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
                )

                self.dof_pos_limits_termination[i, 0] = (
                    m
                    - 0.5
                    * r
                    * self.env_config.env.termination.scales.termination_close_to_dof_pos_limit
                )
                self.dof_pos_limits_termination[i, 1] = (
                    m
                    + 0.5
                    * r
                    * self.env_config.env.termination.scales.termination_close_to_dof_pos_limit
                )
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id < 3:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                # logger.debug(f"Mass of body {i}: {p.mass} (before randomization)")
            # logger.debug(f"Total mass {sum} (before randomization)")

        # randomize base com
        if self.env_config.domain_rand.randomize_base_com:
            self._base_com_bias = torch.zeros(
                self.num_envs,
                3,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            # if env_id < 3:
            #     logger.debug("randomizing base com")
            if self.env_config.robot.has_torso:
                torso_index = self._body_list.index("torso_link")
            else:
                torso_index = self._body_list.index(
                    "pelvis"
                )  # for fixed upper URDF we only have pelvis link
            assert torso_index != -1

            com_x_bias = np.random.uniform(
                self.env_config.domain_rand.base_com_range.x[0],
                self.env_config.domain_rand.base_com_range.x[1],
            )
            com_y_bias = np.random.uniform(
                self.env_config.domain_rand.base_com_range.y[0],
                self.env_config.domain_rand.base_com_range.y[1],
            )
            com_z_bias = np.random.uniform(
                self.env_config.domain_rand.base_com_range.z[0],
                self.env_config.domain_rand.base_com_range.z[1],
            )

            self._base_com_bias[env_id, 0] += com_x_bias
            self._base_com_bias[env_id, 1] += com_y_bias
            self._base_com_bias[env_id, 2] += com_z_bias

            props[torso_index].com.x += com_x_bias
            props[torso_index].com.y += com_y_bias
            props[torso_index].com.z += com_z_bias

        # randomize link mass
        if self.env_config.domain_rand.randomize_link_mass:
            self._link_mass_scale = torch.ones(
                self.num_envs,
                len(self.env_config.robot.randomize_link_body_names),
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            # if env_id < 3:
            #     logger.debug("randomizing link mass")
            for i, body_name in enumerate(
                self.env_config.robot.randomize_link_body_names
            ):
                body_index = self._body_list.index(body_name)
                assert body_index != -1

                mass_scale = np.random.uniform(
                    self.env_config.domain_rand.link_mass_range[0],
                    self.env_config.domain_rand.link_mass_range[1],
                )
                props[body_index].mass *= mass_scale

                self._link_mass_scale[env_id, i] *= mass_scale

        if self.env_config.domain_rand.randomize_payload_mass:
            if env_id == 0:
                self.payload = torch_rand_float(
                    self.env_config.domain_rand.payload_mass_range[0],
                    self.env_config.domain_rand.payload_mass_range[1],
                    (self.num_envs, 1),
                    device=self.device,
                )
                self.hand_payload = torch_rand_float(
                    self.env_config.domain_rand.hand_payload_mass_range[0],
                    self.env_config.domain_rand.hand_payload_mass_range[1],
                    (self.num_envs, 2),
                    device=self.device,
                )
            if self.env_config.robot.has_torso:
                torso_index = self._body_list.index("torso_link")
            else:
                torso_index = self._body_list.index(
                    "pelvis"
                )  # for fixed upper URDF we only have pelvis link
            assert torso_index != -1

            left_hand_index = self._body_list.index("left_hand_palm_link")
            right_hand_index = self._body_list.index("right_hand_palm_link")

            assert torso_index != -1
            assert left_hand_index != -1
            assert right_hand_index != -1

            props[torso_index].mass += self.payload[env_id, 0]
            props[left_hand_index].mass += self.hand_payload[env_id, 0]
            props[right_hand_index].mass += self.hand_payload[env_id, 1]

        # # randomize base mass
        # if self.env_config.domain_rand.randomize_base_mass:
        #     raise Exception("index 0 is for world, 13 is for torso!")
        #     raise NotImplementedError
        #     rng = self.env_config.domain_rand.added_mass_range
        #     props[0].mass += np.random.uniform(rng[0], rng[1])

        if env_id < 3:
            sum_mass = 0
            for i in range(len(props)):
                # logger.debug(f"Mass of body {i}: {props[i].mass} (after randomization)")
                sum_mass += props[i].mass
            # logger.debug(f"Total mass {sum_mass} (afters randomization)")
        return props

    def _process_obj_rigid_body_props(self, props, env_id):
        if self.env_config.domain_rand.randomize_obj_mass:
            if env_id == 0:
                obj_mass_scale_range = self.env_config.domain_rand.obj_mass_scale_range
                self._obj_mass_scale = torch_rand_float(
                    obj_mass_scale_range[0],
                    obj_mass_scale_range[1],
                    (self.num_envs, len(props)),
                    device="cpu",
                )
            for s in range(len(props)):
                props[s].mass *= self._obj_mass_scale[env_id, s]
        return props

    def get_dof_limits_properties(self):
        # assert the isaacgym dof limits are the same as the config
        assert self._body_list == self.body_names
        for i in range(self.num_dof):
            # import pdb; pdb.set_trace()
            assert (
                abs(
                    self.hard_dof_pos_limits[i, 0].item()
                    - self.robot_config.dof_pos_lower_limit_list[i]
                )
                < 1e-5
            ), f"DOF {i} lower limit does not match"
            assert (
                abs(
                    self.hard_dof_pos_limits[i, 1].item()
                    - self.robot_config.dof_pos_upper_limit_list[i]
                )
                < 1e-5
            ), f"DOF {i} upper limit does not match"
            assert (
                abs(
                    self.dof_vel_limits[i].item()
                    - self.robot_config.dof_vel_limit_list[i]
                )
                < 1e-5
            ), f"DOF {i} velocity limit does not match"
            assert (
                abs(
                    self.torque_limits[i].item()
                    - self.robot_config.dof_effort_limit_list[i]
                )
                < 1e-5
            ), f"DOF {i} effort limit does not match"
            # assert self.dof_pos_hard_dof_pos_limitslimits[i, 1].item() == self.robot_config.dof_pos_upper_limit_list[i], f"DOF {i} upper limit does not match"
            # assert self.dof_vel_limits[i].item() == self.robot_config.dof_vel_limit_list[i], f"DOF {i} velocity limit does not match"
            # assert self.torque_limits[i].item() == self.robot_config.dof_effort_limit_list[i], f"DOF {i} effort limit does not match"

        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name):
        return self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.robot_handles[0], body_name
        )

    def prepare_sim(self):
        self.gym.prepare_sim(self.sim)
        # Refresh tensors BEFORE we acquire them https://forums.developer.nvidia.com/t/isaacgym-preview-4-actor-root-state-returns-nans-with-isaacgymenvs-style-task/223738/4
        self.refresh_sim_tensors()

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)

        # jacobian and mass matrix
        robot_name = self.env_config.robot.asset.robot_type
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, robot_name)
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, robot_name)

        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        self.massmatrix = gymtorch.wrap_tensor(_massmatrix)

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(
            self.num_envs, bodies_per_env, 13
        )
        self._rigid_body_pos = self._rigid_body_state_reshaped[
            ..., : self.num_bodies, 0:3
        ]
        self._rigid_body_rot = self._rigid_body_state_reshaped[
            ..., : self.num_bodies, 3:7
        ]
        self._rigid_body_vel = self._rigid_body_state_reshaped[
            ..., : self.num_bodies, 7:10
        ]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[
            ..., : self.num_bodies, 10:13
        ]

        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        self.refresh_sim_tensors()

        self.all_root_states: torch.Tensor = gymtorch.wrap_tensor(actor_root_state)
        assert (
            self.num_actors == 2
        ), f"Catching env currently only suppoerts robot + 1 obj, got {self.num_actors=}"
        all_root_states_reshaped = self.all_root_states.view(
            self.num_envs, self.num_actors, actor_root_state.shape[-1]
        )
        self.robot_root_states = all_root_states_reshaped[..., 0, :]
        self.base_quat = self.robot_root_states[..., 3:7]  # isaacgym uses xyzws

        self.obj_root_states = all_root_states_reshaped[..., 1, :]

        self.dof_state: torch.Tensor = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos: torch.Tensor = self.dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.dof_vel: torch.Tensor = self.dof_state.view(self.num_envs, -1, 2)[..., 1]
        self.contact_forces: torch.Tensor = gymtorch.wrap_tensor(
            net_contact_forces
        ).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

    def refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    @property
    def num_actors(self):
        return self.all_root_states.shape[0] // self.num_envs

    def apply_torques_at_dof(self, torques):
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(torques)
        )

    ################################################################################################
    # NOTE (arth): for each of the below, we assume user is sending "env_id to reset"
    #   rather than "obj_id to reset", and change the set_env_ids accordingly

    def _env_ids_to_actor_ids(self, env_ids: torch.Tensor):
        return (
            env_ids.unsqueeze(1) * self.num_actors
            + torch.arange(self.num_actors, device=env_ids.device, dtype=env_ids.dtype)
        ).flatten()

    def _env_ids_to_robot_ids(self, env_ids: torch.Tensor):
        return env_ids * self.num_actors

    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        set_env_ids_int32 = self._env_ids_to_actor_ids(set_env_ids).to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(set_env_ids_int32),
            len(set_env_ids_int32),
        )

    def apply_rigid_body_force_at_pos_tensor(self, force_tensor, pos_tensor):
        self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim,
            gymtorch.unwrap_tensor(force_tensor),
            gymtorch.unwrap_tensor(pos_tensor),
            gymapi.ENV_SPACE,
        )

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        set_env_ids_int32 = self._env_ids_to_robot_ids(set_env_ids).to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(dof_states),
            gymtorch.unwrap_tensor(set_env_ids_int32),
            len(set_env_ids_int32),
        )

    ################################################################################################

    def simulate_at_each_physics_step(self):
        self.gym.simulate(self.sim)
        if self.sim_device == "cpu":
            self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)

    def setup_viewer(self):
        self.enable_viewer_sync = True
        self.visualize_viewer = True
        viewer_camera_props = gymapi.CameraProperties()
        viewer_camera_props.horizontal_fov = 45
        self.viewer = self.gym.create_viewer(self.sim, viewer_camera_props)
        # subscribe to keyboard shortcuts
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_W, "forward_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_S, "backward_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_A, "left_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_D, "right_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_Q, "heading_left_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_E, "heading_right_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_Z, "zero_command"
        )

        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_P, "push_robots"
        )

        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_N, "next_task"
        # )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_R, "toggle_video_record"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_SEMICOLON, "cancel_video_record"
        )

        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_UP, "force_left_up"
        # )
        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_DOWN, "force_left_down"
        # )

        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_LEFT, "force_right_down"
        # )
        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_RIGHT, "force_right_up"
        # )

        sim_params = self.sim_params
        if sim_params.up_axis == gymapi.UP_AXIS_Z:
            cam_pos = gymapi.Vec3(5.0, 5.0, 3.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 3.0)
        else:
            cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 15.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def render(self, sync_frame_time=True):
        # check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()
        delete_user_viewer_recordings = False
        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()

        # fetch results
        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)

        # step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)

    # debug visualization
    def clear_lines(self):
        self.gym.clear_lines(self.viewer)

    def draw_sphere(self, pos, radius, color, env_id):
        sphere_geom_marker = gymutil.WireframeSphereGeometry(
            radius, 20, 20, None, color=color
        )
        sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
        gymutil.draw_lines(
            sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose
        )

    def draw_line(self, start_point, end_point, color, env_id):
        gymutil.draw_line(
            start_point, end_point, color, self.gym, self.viewer, self.envs[env_id]
        )
