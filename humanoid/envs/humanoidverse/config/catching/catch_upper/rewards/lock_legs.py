from dataclasses import dataclass, field
from typing import List, Literal

from humanoid.utils.dataclasses import class_factory, default_field


@dataclass
class G1CatchUpperLockLegsRewardsConfig:
    reward_mode: Literal["dense", "normalized_dense"] = "dense"
    reward_min: float = -10.0
    reward_max: float = 0.01

    @dataclass
    class RewardScales:
        obj_hand_goal_dist: float = 4.0
        approx_obj_surface_fingertip_dist: float = 4.0
        obj_height: float = 1.0

        # NOTE (arth): reward ablations, off by default. contact: finger_palm_touch=4,
        #   approx_obj_surface_fingertip_dist=0; distance + contact: both 2; stability:
        #   obj_stability=4, approx_obj_surface_fingertip_dist=0; object-palm distance:
        #   obj_hand_goal_dist=8, approx_obj_surface_fingertip_dist=0
        finger_palm_touch: float = 0.0
        obj_stability: float = 0.0

        hand_default_pos: float = -1.0
        arm_default_pos: float = -0.25
        avoid_contact_body: float = -2.0
        avoid_contact_arm: float = (
            -1.0  # -0.5
        )  # NOTE (arth): scales much less bc we so mean() across more links

        finger_joint_tracking_error: float = -0.1
        finger_action_rate: float = -0.01
        finger_smoothness: float = -0.05
        joint_power_hand: float = -2e-5
        action_vanish_hand: float = -1.0
        dof_vel_hand: float = -1e-4
        dof_acc_hand: float = -2.5e-7
        torques_hand: float = -2.5e-6
        dof_pos_hand_limits: float = -2.0
        dof_vel_hand_limits: float = -2e-3
        torque_hand_limits: float = -0.1

        arm_joint_tracking_error: float = -0.1
        arm_action_rate: float = -0.01
        arm_smoothness: float = -0.05
        joint_power_arm: float = -2e-5
        # action_vanish_arm: float = -1.0
        dof_vel_arm: float = -1e-4
        dof_acc_arm: float = -2.5e-7
        torques_arm: float = -2.5e-6
        dof_pos_arm_limits: float = -2.0
        dof_vel_arm_limits: float = -2e-3
        torque_arm_limits: float = -0.1

        hand_pos_rate: float = -0.05  # -0.25
        hand_pos_smoothness: float = -0.05  # -0.15
        hand_orientation: float = -0.025  # -0.05

        #

        # # finger_joint_tracking_error: float = 0.0
        # arm_joint_tracking_error: float = -0.1

        # # finger_action_rate: float = 0.0
        # arm_action_rate: float = -0.1  # -0.01

        # # finger_smoothness: float = 0.0
        # arm_smoothness: float = -0.5  # -0.05

        # # hand_action_vanish: float = 0.0

        # joint_power: float = -2e-5 * 10  # -2e-4
        # torques_arm: float = -2.5e-6 * 10  # -1.0
        # # torques_hand: float = 0.0

        # dof_pos_limits: float = -2.0
        # torque_arm_limits: float = -0.1
        # # torque_hand_limits: float = 0.0

        #

        # finger_joint_tracking_error: float = (
        #     -1.0
        # )  # -0.5  # -0.1  # -0.005  # -0.01  # -0.005
        # arm_joint_tracking_error: float = -1.0  # -0.5  # -0.5  # -1.0  #  -0.5

        # # finger_action_rate: float = 0.0  # -0.5
        # arm_action_rate: float = -2.0  # -1.0  # -0.7  # -0.3  # -0.2  # -0.01

        # # finger_smoothness: float = 0.0  # -0.5
        # arm_smoothness: float = -2.0  # -1.0  # -0.5  # -0.2  # -0.05

        # joint_power: float = -2e-4  # -2e-5

        # # hand_action_vanish: float = -1.0
        # # arm_action_vanish: float = -1.0
        # action_vanish: float = -1.0

        # dof_vel: float = -5e-4  # -2e-4  # -5e-4  # -2e-4  # -1e-4
        # dof_acc: float = -5e-7  # -5e-7  # -5e-7  # -5e-7  # -2.5e-7
        # torques_arm: float = -1.0  # -5e-1  # -5e-2  # -5e-5
        # torques_hand: float = -5e-2  # -5e-3  # -6e-4

        # dof_pos_limits: float = -2.0
        # dof_vel_limits: float = -0.1  # -0.05  # -0.1  # -2e-3
        # torque_arm_limits: float = -0.1  # -0.1
        # torque_hand_limits: float = -1.0  # -0.1

    @dataclass
    class RewardLimit:
        soft_dof_pos_limit: float = 0.975
        soft_dof_vel_limit: float = 0.80
        soft_torque_limit: float = 0.95

    only_positive_rewards: bool = False
    tracking_sigma: float = 0.25

    max_contact_force: float = 400.0

    reward_penalty_curriculum: bool = True
    reward_initial_penalty_scale: float = 1
    reward_min_penalty_scale: float = 1
    reward_max_penalty_scale: float = 20.0
    reward_penalty_level_down_threshold: int = 40
    reward_penalty_level_up_threshold: int = 80
    reward_penalty_degree: float = 0.00005
    num_compute_average_epl: int = 10000
    reward_penalty_reward_names: List[str] = default_field(
        [
            "arm_default_pos",
            # "avoid_contact_body",
            # "avoid_contact_arm",
            #
            "finger_joint_tracking_error",
            "finger_action_rate",
            "finger_smoothness",
            "joint_power_hand",
            "action_vanish_hand",
            "dof_vel_hand",
            "dof_acc_hand",
            "torques_hand",
            "dof_pos_hand_limits",
            "dof_vel_hand_limits",
            "torque_hand_limits",
            #
            "arm_joint_tracking_error",
            "arm_action_rate",
            "arm_smoothness",
            "joint_power_arm",
            "action_vanish_arm",
            "dof_vel_arm",
            "dof_acc_arm",
            "torques_arm",
            "dof_pos_arm_limits",
            "dof_vel_arm_limits",
            "torque_arm_limits",
            #
            "hand_pos_rate",
            "hand_pos_smoothness",
            "hand_orientation",
        ]
    )

    reward_scales: RewardScales = class_factory(RewardScales)
    reward_limit: RewardLimit = class_factory(RewardLimit)
