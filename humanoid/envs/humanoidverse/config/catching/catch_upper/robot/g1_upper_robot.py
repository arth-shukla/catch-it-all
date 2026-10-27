import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from humanoid import HUMANOID_RESOURCES_DIR
from humanoid.utils.dataclasses import class_factory, default_field


@dataclass
class G1CatchUpperLockLegsRobotConfig:
    name: str = "g1_29dof_lock_legs_lock_waist"

    dof_obs_size: int = 27
    arm_dof_obs_size: int = 27
    actions_dim: int = field(
        init=False
    )  # NOTE (arth): 26 for ik, 27 for dof pos / target dof pos

    policy_obs_dim: int = field(init=False)
    critic_obs_dim: int = field(init=False)
    teacher_obs_dim: int = field(init=False)
    algo_obs_dim_dict: Dict[str, int] = field(init=False)

    contact_pairs_multiplier: int = 8
    mimic_small_marker_bodies: Optional[Any] = None
    # hand/arm
    used_hand: str = "right"

    finger_joint_names: List[str] = default_field(
        [
            "f1_finger_palm_joint",
            "f1_finger_base_joint",
            "f1_finger_middle_joint",
            "f1_finger_end_joint",
            "f2_finger_palm_joint",
            "f2_finger_base_joint",
            "f2_finger_middle_joint",
            "f2_finger_end_joint",
            "f3_finger_palm_joint",
            "f3_finger_base_joint",
            "f3_finger_middle_joint",
            "f3_finger_end_joint",
            "f4_finger_palm_joint",
            "f4_finger_base_joint",
            "f4_finger_middle_joint",
            "f4_finger_end_joint",
            "f5_finger_palm_joint",
            "f5_finger_base_joint",
            "f5_finger_middle_joint",
            "f5_finger_end_joint",
        ]
    )
    finger_link_names: List[str] = default_field(
        [
            "f1_finger_base",
            "f1_finger_base_lever",
            "f1_finger_middle_lever",
            "f1_finger_end_lever",
            "f1_finger_tip",
            "f2_finger_base",
            "f2_finger_base_lever",
            "f2_finger_middle_lever",
            "f2_finger_end_lever",
            "f2_finger_tip",
            "f3_finger_base",
            "f3_finger_base_lever",
            "f3_finger_middle_lever",
            "f3_finger_end_lever",
            "f3_finger_tip",
            "f4_finger_base",
            "f4_finger_base_lever",
            "f4_finger_middle_lever",
            "f4_finger_end_lever",
            "f4_finger_tip",
            "f5_finger_base",
            "f5_finger_base_lever",
            "f5_finger_middle_lever",
            "f5_finger_end_lever",
            "f5_finger_tip",
        ]
    )

    right_hand_name: str = "right_palm"
    right_arm_link_names: List[str] = default_field(
        [
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_roll_link",
            "right_wrist_pitch_link",
            "right_wrist_yaw_link",
        ]
    )
    right_arm_joint_names: List[str] = default_field(
        [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
    )
    right_fingertip_names: List[str] = default_field(
        [
            "f1_finger_tip",
            "f2_finger_tip",
            "f3_finger_tip",
            "f4_finger_tip",
            "f5_finger_tip",
        ]
    )
    right_shoulder_joint_names: List[str] = default_field(
        [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
        ]
    )
    right_wrist_joint_names: List[str] = default_field(
        [
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
    )

    dof_names: List[str] = field(init=False)
    has_torso: bool = True
    torso_name: str = "torso_link"

    # fmt: off
    right_arm_dof_pos_lower_limit_list: List[float] = default_field([
        -3.089200019836426, -2.251499891281128, -2.618000030517578, -1.0471999645233154, 
        -1.972222089767456, -1.6144295930862427, -1.6144295930862427,
    ])
    right_arm_dof_pos_upper_limit_list: List[float] = default_field([
        2.6703999042510986, 1.5881999731063843, 2.618000030517578, 2.094399929046631,
        1.972222089767456, 1.6144295930862427, 1.6144295930862427,
    ])

    finger_dof_pos_lower_limit_list: List[float] = default_field([
        -0.6980000138282776, 0.0, 0.0, 0.0,
        -0.6980000138282776, 0.0, 0.0, 0.0,
        -0.6980000138282776, 0.0, 0.0, 0.0,
        -0.6980000138282776, 0.0, 0.0, 0.0,
        -0.6980000138282776, 0.0, 0.0, 0.0,
    ])
    finger_dof_pos_upper_limit_list: List[float] = default_field([
        0.6980000138282776, 1.5707999467849731, 1.5707999467849731, 1.5707999467849731,
        0.6980000138282776, 1.5707999467849731, 1.5707999467849731, 1.5707999467849731,
        0.6980000138282776, 1.5707999467849731, 1.5707999467849731, 1.5707999467849731,
        0.6980000138282776, 1.5707999467849731, 1.5707999467849731, 1.5707999467849731,
        0.6980000138282776, 1.5707999467849731, 1.5707999467849731, 1.5707999467849731,
    ])

    dof_pos_lower_limit_list: List[float] = field(init=False)
    dof_pos_upper_limit_list: List[float] = field(init=False)
    dof_vel_limit_list: List[float] = default_field(
        [
            # # 32.0, 20.0, 32.0, 20.0, 37.0, 37.0,
            # # 32.0, 20.0, 32.0, 20.0, 37.0, 37.0,
            # # 32.0, # 37.0, 37.0,
            # 37.0, 37.0, 37.0, 37.0,
            # 37.0, 22.0, 22.0,
            # 37.0, 37.0, 37.0, 37.0,
            # 37.0, 22.0, 22.0
            # arm
            37.0, 37.0, 37.0, 37.0,
            37.0, 22.0, 22.0,
            # hand
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ]
    )
    dof_effort_limit_list: List[float] = default_field(
        [
            # # 88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
            # # 88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
            # # 88.0, # 50.0, 50.0,
            # arm
            25.0, 25.0, 25.0, 25.0,
            25.0, 5.0, 5.0,
            # hand
            10.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0,
        ]
    )
    # fmt: on

    terminate_after_contacts_on: List[str] = default_field(["torso"])
    penalize_contacts_on: List[str] = default_field(["hip", "knee"])

    randomize_link_body_names: Union[List[str], str] = "${robot.body_names}"
    avoid_contact_body_names: List[str] = default_field(
        ["pelvis", "torso_link", "head_link"]
    )

    @dataclass
    class InitState:
        pos: List[float] = default_field([0.0, 0.0, 0.8])  # x,y,z [m]
        rot: List[float] = default_field([0.0, 0.0, 0.0, 1.0])  # x,y,z,w [quat]
        lin_vel: List[float] = default_field([0.0, 0.0, 0.0])  # x,y,z [m/s]
        ang_vel: List[float] = default_field([0.0, 0.0, 0.0])  # x,y,z [rad/s]

        default_joint_angles: Dict[str, float] = (
            "${robot.init_state.underhand_joint_angles}"
            # NOTE (arth): ik uses the normal default dof pos when computing eef lims, keep constant across policies
            # "${robot.init_state.overhand_joint_angles} if ${env.toss_overhand} else ${robot.init_state.underhand_joint_angles}"
        )

        underhand_joint_angles: Dict[str, float] = default_field(
            {
                "left_hip_yaw_joint": 0.0,
                "left_hip_roll_joint": 0.0,
                "left_hip_pitch_joint": -0.1,
                "left_knee_joint": 0.3,
                "left_ankle_pitch_joint": -0.2,
                "left_ankle_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_pitch_joint": -0.1,
                "right_knee_joint": 0.3,
                "right_ankle_pitch_joint": -0.2,
                "right_ankle_roll_joint": 0.0,
                # "waist_yaw_joint": 0.0,
                # "waist_roll_joint": 0.0,
                # "waist_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "left_wrist_roll_joint": -math.pi / 2,
                "left_wrist_pitch_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                #
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,
                "right_wrist_roll_joint": math.pi / 2,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                #
                "f1_finger_palm_joint": 0.0,
                "f1_finger_base_joint": math.pi / 6,
                "f1_finger_middle_joint": math.pi / 6,
                "f1_finger_end_joint": math.pi / 6,
                "f2_finger_palm_joint": 0.0,
                "f2_finger_base_joint": math.pi / 6,
                "f2_finger_middle_joint": math.pi / 6,
                "f2_finger_end_joint": math.pi / 6,
                "f3_finger_palm_joint": 0.0,
                "f3_finger_base_joint": math.pi / 6,
                "f3_finger_middle_joint": math.pi / 6,
                "f3_finger_end_joint": math.pi / 6,
                "f4_finger_palm_joint": 0.0,
                "f4_finger_base_joint": math.pi / 6,
                "f4_finger_middle_joint": math.pi / 6,
                "f4_finger_end_joint": math.pi / 6,
                "f5_finger_palm_joint": 0.0,
                "f5_finger_base_joint": math.pi / 6,
                "f5_finger_middle_joint": math.pi / 6,
                "f5_finger_end_joint": math.pi / 6,
                #
                # "right_hand_index_0_joint": 0.0,
                # "right_hand_index_1_joint": 0.0,  # 0.8,
                # "right_hand_middle_0_joint": 0.0,
                # "right_hand_middle_1_joint": 0.0,  # 0.8,
                # "right_hand_thumb_0_joint": 0.0,
                # "right_hand_thumb_1_joint": 0.0,
                # "right_hand_thumb_2_joint": 0.0,  # -0.8,
            }
        )

        overhand_joint_angles: Dict[str, float] = default_field(
            {
                "left_hip_yaw_joint": 0.0,
                "left_hip_roll_joint": 0.0,
                "left_hip_pitch_joint": -0.1,
                "left_knee_joint": 0.3,
                "left_ankle_pitch_joint": -0.2,
                "left_ankle_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_pitch_joint": -0.1,
                "right_knee_joint": 0.3,
                "right_ankle_pitch_joint": -0.2,
                "right_ankle_roll_joint": 0.0,
                # "waist_yaw_joint": 0.0,
                # "waist_roll_joint": 0.0,
                # "waist_pitch_joint": 0.0,
                #
                "left_shoulder_pitch_joint": -0.038,
                "left_shoulder_roll_joint": 0.72,
                "left_shoulder_yaw_joint": 0.17,
                "left_elbow_joint": -0.85,
                "left_wrist_roll_joint": 0.97,
                "left_wrist_pitch_joint": 0.0,
                "left_wrist_yaw_joint": 0.82,
                #
                "right_shoulder_pitch_joint": -0.038,
                "right_shoulder_roll_joint": -0.72,
                "right_shoulder_yaw_joint": -0.17,
                "right_elbow_joint": -0.85,
                "right_wrist_roll_joint": -0.97,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": -0.82,
                #
                "f1_finger_palm_joint": 0.0,
                "f1_finger_base_joint": math.pi / 6,
                "f1_finger_middle_joint": math.pi / 6,
                "f1_finger_end_joint": math.pi / 6,
                "f2_finger_palm_joint": 0.0,
                "f2_finger_base_joint": math.pi / 6,
                "f2_finger_middle_joint": math.pi / 6,
                "f2_finger_end_joint": math.pi / 6,
                "f3_finger_palm_joint": 0.0,
                "f3_finger_base_joint": math.pi / 6,
                "f3_finger_middle_joint": math.pi / 6,
                "f3_finger_end_joint": math.pi / 6,
                "f4_finger_palm_joint": 0.0,
                "f4_finger_base_joint": math.pi / 6,
                "f4_finger_middle_joint": math.pi / 6,
                "f4_finger_end_joint": math.pi / 6,
                "f5_finger_palm_joint": 0.0,
                "f5_finger_base_joint": math.pi / 6,
                "f5_finger_middle_joint": math.pi / 6,
                "f5_finger_end_joint": math.pi / 6,
                #
                # "right_hand_index_0_joint": 0.0,
                # "right_hand_index_1_joint": 0.0,  # 0.8,
                # "right_hand_middle_0_joint": 0.0,
                # "right_hand_middle_1_joint": 0.0,  # 0.8,
                # "right_hand_thumb_0_joint": 0.0,
                # "right_hand_thumb_1_joint": 0.0,
                # "right_hand_thumb_2_joint": 0.0,  # -0.8,
            }
        )

        use_manual_arm_inits: bool = False
        ordered_underhand_arm_inits: List[Dict[str, float]] = default_field(
            [
                {  # underhand left slight tilt
                    "left_elbow_joint": 0.13670447505586086,
                    "left_shoulder_pitch_joint": -0.017891922329100623,
                    "left_shoulder_roll_joint": 0.3516829501868694,
                    "left_shoulder_yaw_joint": 0.10240597960330046,
                    "left_wrist_pitch_joint": 0.09449151454870797,
                    "left_wrist_roll_joint": -1.489610829070408,
                    "left_wrist_yaw_joint": -0.12618763923203033,
                    "right_elbow_joint": 0.13670447505586086,
                    "right_shoulder_pitch_joint": -0.017891922329100623,
                    "right_shoulder_roll_joint": -0.3516829501868694,
                    "right_shoulder_yaw_joint": -0.10240597960330046,
                    "right_wrist_pitch_joint": 0.09449151454870797,
                    "right_wrist_roll_joint": 1.489610829070408,
                    "right_wrist_yaw_joint": 0.12618763923203033,
                },
                {  # underhand left
                    "left_elbow_joint": 0.05241134176191974,
                    "left_shoulder_pitch_joint": -0.009800583599418934,
                    "left_shoulder_roll_joint": 0.1594783845929413,
                    "left_shoulder_yaw_joint": 0.12352798860014183,
                    "left_wrist_pitch_joint": 0.1274697092589659,
                    "left_wrist_roll_joint": -1.7249922643662479,
                    "left_wrist_yaw_joint": -0.0230322923890603,
                    "right_elbow_joint": 0.05241134176191974,
                    "right_shoulder_pitch_joint": -0.009800583599418934,
                    "right_shoulder_roll_joint": -0.1594783845929413,
                    "right_shoulder_yaw_joint": -0.12352798860014183,
                    "right_wrist_pitch_joint": 0.1274697092589659,
                    "right_wrist_roll_joint": 1.7249922643662479,
                    "right_wrist_yaw_joint": 0.0230322923890603,
                },
                {  # underhand
                    "left_elbow_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_roll_joint": -math.pi / 2,
                    "left_wrist_yaw_joint": 0.0,
                    "right_elbow_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": -0.0,
                    "right_shoulder_yaw_joint": -0.0,
                    "right_wrist_pitch_joint": 0.0,
                    "right_wrist_roll_joint": math.pi / 2,
                    "right_wrist_yaw_joint": -0.0,
                },
                {  # underhand lower
                    "left_elbow_joint": 0.8581101233903955,
                    "left_shoulder_pitch_joint": -0.22484454880442206,
                    "left_shoulder_roll_joint": 0.009357340192165867,
                    "left_shoulder_yaw_joint": 0.054536575548134285,
                    "left_wrist_pitch_joint": -0.002528293417193155,
                    "left_wrist_roll_joint": -1.5778057893754127,
                    "left_wrist_yaw_joint": -0.6415547166996505,
                    "right_elbow_joint": 0.8581101233903955,
                    "right_shoulder_pitch_joint": -0.22484454880442206,
                    "right_shoulder_roll_joint": -0.009357340192165867,
                    "right_shoulder_yaw_joint": -0.054536575548134285,
                    "right_wrist_pitch_joint": -0.002528293417193155,
                    "right_wrist_roll_joint": 1.5778057893754127,
                    "right_wrist_yaw_joint": 0.6415547166996505,
                },
                {  # underhand lower left
                    "left_elbow_joint": 0.6044493446429693,
                    "left_shoulder_pitch_joint": 0.03612545594246233,
                    "left_shoulder_roll_joint": 0.17267030839106629,
                    "left_shoulder_yaw_joint": 0.10271930059407926,
                    "left_wrist_pitch_joint": 0.18994406126908323,
                    "left_wrist_roll_joint": -1.6505024871864613,
                    "left_wrist_yaw_joint": -0.6235493644230597,
                    "right_elbow_joint": 0.6044493446429693,
                    "right_shoulder_pitch_joint": 0.03612545594246233,
                    "right_shoulder_roll_joint": -0.17267030839106629,
                    "right_shoulder_yaw_joint": -0.10271930059407926,
                    "right_wrist_pitch_joint": 0.18994406126908323,
                    "right_wrist_roll_joint": 1.6505024871864613,
                    "right_wrist_yaw_joint": 0.6235493644230597,
                },
                {  # sideways-ish
                    "left_elbow_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.63,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_roll_joint": -1.0,
                    "left_wrist_yaw_joint": 0.0,
                    "right_elbow_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": -0.63,
                    "right_shoulder_yaw_joint": -0.0,
                    "right_wrist_pitch_joint": 0.0,
                    "right_wrist_roll_joint": 1.0,
                    "right_wrist_yaw_joint": -0.0,
                },
                # {  # sideways
                #     "left_elbow_joint": 0.0,
                #     "left_shoulder_pitch_joint": 0.0,
                #     "left_shoulder_roll_joint": 1.2,
                #     "left_shoulder_yaw_joint": 0.0,
                #     "left_wrist_pitch_joint": 0.0,
                #     "left_wrist_roll_joint": -1.4,
                #     "left_wrist_yaw_joint": 0.0,
                # },
            ]
        )
        ordered_overhand_arm_inits: List[Dict[str, float]] = default_field(
            [
                # {  # underhand
                #     "left_elbow_joint": 0.0,
                #     "left_shoulder_pitch_joint": 0.0,
                #     "left_shoulder_roll_joint": 0.0,
                #     "left_shoulder_yaw_joint": 0.0,
                #     "left_wrist_pitch_joint": 0.0,
                #     "left_wrist_roll_joint": -math.pi / 2,
                #     "left_wrist_yaw_joint": 0.0,
                # },
                {  # sideways-ish
                    "left_elbow_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.63,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_roll_joint": -1.0,
                    "left_wrist_yaw_joint": 0.0,
                    "right_elbow_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": -0.63,
                    "right_shoulder_yaw_joint": -0.0,
                    "right_wrist_pitch_joint": 0.0,
                    "right_wrist_roll_joint": 1.0,
                    "right_wrist_yaw_joint": -0.0,
                },
                {  # sideways
                    "left_elbow_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 1.2,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_roll_joint": -1.4,
                    "left_wrist_yaw_joint": 0.0,
                    "right_elbow_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": -1.2,
                    "right_shoulder_yaw_joint": -0.0,
                    "right_wrist_pitch_joint": 0.0,
                    "right_wrist_roll_joint": 1.4,
                    "right_wrist_yaw_joint": -0.0,
                },
                {  # overhand (low)
                    "left_elbow_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 1.2,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_wrist_pitch_joint": -0.57,
                    "left_wrist_roll_joint": 0.0,
                    "left_wrist_yaw_joint": 1.5,
                    "right_elbow_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": -1.2,
                    "right_shoulder_yaw_joint": -0.0,
                    "right_wrist_pitch_joint": -0.57,
                    "right_wrist_roll_joint": -0.0,
                    "right_wrist_yaw_joint": -1.5,
                },
                {  # overhand (low, tiled left)
                    "left_elbow_joint": -0.61,
                    "left_shoulder_pitch_joint": 0.38,
                    "left_shoulder_roll_joint": 0.93,
                    "left_shoulder_yaw_joint": 0.17,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_roll_joint": -0.17,
                    "left_wrist_yaw_joint": 1.5,
                    "right_elbow_joint": -0.61,
                    "right_shoulder_pitch_joint": 0.38,
                    "right_shoulder_roll_joint": -0.93,
                    "right_shoulder_yaw_joint": -0.17,
                    "right_wrist_pitch_joint": 0.0,
                    "right_wrist_roll_joint": 0.17,
                    "right_wrist_yaw_joint": -1.5,
                },
                {  # overhand
                    "left_elbow_joint": -0.85,
                    "left_shoulder_pitch_joint": -0.038,
                    "left_shoulder_roll_joint": 0.72,
                    "left_shoulder_yaw_joint": 0.17,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_roll_joint": 0.97,
                    "left_wrist_yaw_joint": 0.82,
                    "right_elbow_joint": -0.85,
                    "right_shoulder_pitch_joint": -0.038,
                    "right_shoulder_roll_joint": -0.72,
                    "right_shoulder_yaw_joint": -0.17,
                    "right_wrist_pitch_joint": 0.0,
                    "right_wrist_roll_joint": -0.97,
                    "right_wrist_yaw_joint": -0.82,
                },
                {  # overhand
                    "left_elbow_joint": -0.85,
                    "left_shoulder_pitch_joint": -0.038,
                    "left_shoulder_roll_joint": 0.72,
                    "left_shoulder_yaw_joint": 0.17,
                    "left_wrist_pitch_joint": 0.3,
                    "left_wrist_roll_joint": 1.4,
                    "left_wrist_yaw_joint": 0.8,
                    "right_elbow_joint": -0.85,
                    "right_shoulder_pitch_joint": -0.038,
                    "right_shoulder_roll_joint": -0.72,
                    "right_shoulder_yaw_joint": -0.17,
                    "right_wrist_pitch_joint": 0.3,
                    "right_wrist_roll_joint": -1.4,
                    "right_wrist_yaw_joint": -0.8,
                },
                {  # overhand
                    "left_elbow_joint": -0.85,
                    "left_shoulder_pitch_joint": -0.038,
                    "left_shoulder_roll_joint": 0.92,
                    "left_shoulder_yaw_joint": 0.18,
                    "left_wrist_pitch_joint": 0.012,
                    "left_wrist_roll_joint": 1.7,
                    "left_wrist_yaw_joint": 0.8,
                    "right_elbow_joint": -0.85,
                    "right_shoulder_pitch_joint": -0.038,
                    "right_shoulder_roll_joint": -0.92,
                    "right_shoulder_yaw_joint": -0.18,
                    "right_wrist_pitch_joint": 0.012,
                    "right_wrist_roll_joint": -1.7,
                    "right_wrist_yaw_joint": -0.8,
                },
                {  # overhand
                    "left_elbow_joint": -0.85,
                    "left_shoulder_pitch_joint": -0.038,
                    "left_shoulder_roll_joint": 0.92,
                    "left_shoulder_yaw_joint": 0.18,
                    "left_wrist_pitch_joint": 0.73,
                    "left_wrist_roll_joint": 1.1,
                    "left_wrist_yaw_joint": 0.8,
                    "right_elbow_joint": -0.85,
                    "right_shoulder_pitch_joint": -0.038,
                    "right_shoulder_roll_joint": -0.92,
                    "right_shoulder_yaw_joint": -0.18,
                    "right_wrist_pitch_joint": 0.73,
                    "right_wrist_roll_joint": -1.1,
                    "right_wrist_yaw_joint": -0.8,
                },
                {  # overhand (low-ish, tilted left)
                    "left_elbow_joint": -0.65,
                    "left_shoulder_pitch_joint": -0.053,
                    "left_shoulder_roll_joint": 0.4,
                    "left_shoulder_yaw_joint": 0.48,
                    "left_wrist_pitch_joint": 0.085,
                    "left_wrist_roll_joint": 0.2,
                    "left_wrist_yaw_joint": 0.72,
                    "right_elbow_joint": -0.65,
                    "right_shoulder_pitch_joint": -0.053,
                    "right_shoulder_roll_joint": -0.4,
                    "right_shoulder_yaw_joint": -0.48,
                    "right_wrist_pitch_joint": 0.085,
                    "right_wrist_roll_joint": -0.2,
                    "right_wrist_yaw_joint": -0.72,
                },
                {  # sideways, facing forward
                    "left_elbow_joint": -0.3,
                    "left_shoulder_pitch_joint": 0.12,
                    "left_shoulder_roll_joint": 0.4,
                    "left_shoulder_yaw_joint": 0.48,
                    "left_wrist_pitch_joint": 0.085,
                    "left_wrist_roll_joint": -0.89,
                    "left_wrist_yaw_joint": 0.68,
                    "right_elbow_joint": -0.3,
                    "right_shoulder_pitch_joint": 0.12,
                    "right_shoulder_roll_joint": -0.4,
                    "right_shoulder_yaw_joint": -0.48,
                    "right_wrist_pitch_joint": 0.085,
                    "right_wrist_roll_joint": 0.89,
                    "right_wrist_yaw_joint": -0.68,
                },
            ]
        )

    body_names: List[str] = field(init=False)

    @dataclass
    class Motion:
        upper_body_link: List[str] = field(init=False)

        pelvis_link: str = "pelvis"
        base_link: str = "torso_link"
        hips_link: List[str] = default_field(
            [
                "left_hip_pitch_link",
                "left_hip_roll_link",
                "left_hip_yaw_link",
                "right_hip_pitch_link",
                "right_hip_roll_link",
                "right_hip_yaw_link",
            ]
        )

    @dataclass
    class Control:
        control_type: str = "P"
        # PD Drive parameters:
        stiffness: Dict[str, float] = default_field(
            {  # [N*m/rad]
                "finger": 3.0,
                "hand": 1.5,
                #
                "hip_yaw": 100.0,
                "hip_roll": 100.0,
                "hip_pitch": 100.0,
                "knee": 200.0,
                "ankle_pitch": 20.0,
                "ankle_roll": 20.0,
                "waist_yaw": 300.0,
                "shoulder_pitch": 100.0,
                "shoulder_roll": 75.0,
                "shoulder_yaw": 50.0,
                "elbow": 100.0,
                "wrist_roll": 20.0,
                "wrist_pitch": 20.0,
                "wrist_yaw": 20.0,
            }
        )

        damping: Dict[str, float] = default_field(
            {  # [N*m/rad]  # [N*m*s/rad]
                "finger": 0.1,
                "hand": 0.2,
                #
                "hip_yaw": 2.5,
                "hip_roll": 2.5,
                "hip_pitch": 2.5,
                "knee": 5.0,
                "ankle_pitch": 0.2,
                "ankle_roll": 0.1,
                "waist_yaw": 5.0,
                "shoulder_pitch": 2.0,
                "shoulder_roll": 1.5,
                "shoulder_yaw": 1.0,
                "elbow": 1.0,
                "wrist_roll": 0.5,
                "wrist_pitch": 0.5,
                "wrist_yaw": 0.5,
            }
        )

        # action scale: target angle = actionScale * action + defaultAngle
        # isaac_pd_scale: bool = False
        arm_control_mode: Literal["pos", "target_delta_pos", "ik"] = "ik"

        @dataclass
        class IKControllerConfig:
            max_delta_pos: float = 1.5
            max_delta_rpy: float = math.pi

            delta_pos_action_scale: float = 30.0
            delta_rpy_action_scale: float = 30.0

            right_eef_pos_limits_lower: Tuple[float, float, float] = (-0.1, -0.4, -0.15)
            right_eef_pos_limits_upper: Tuple[float, float, float] = (0.17, 0.3, 0.35)

            eef_pos_limits_lower: Tuple[float, float, float] = field(init=False)
            eef_pos_limits_upper: Tuple[float, float, float] = field(init=False)

            ik_version: Literal["v1", "v2"] = "v2"
            tol: float = 0.01
            max_iters: int = 2
            damping: float = 0.05

            noised_dof_pos: bool = False

            target_eef_pose_ema: Optional[float] = 0.5
            joint_target_ema: Optional[float] = 0.1

            def __post_init__(self):
                if self.target_eef_pose_ema is not None:
                    assert 0.0 <= self.target_eef_pose_ema <= 1.0

        ik_controller_config: IKControllerConfig = class_factory(IKControllerConfig)

        @dataclass
        class TargetDeltaPosControllerConfig:
            action_clip_value: float = 0.1
            normalize_action_space: bool = True
            action_lim_scale: float = 10.0

        target_delta_pos_controller_config: TargetDeltaPosControllerConfig = (
            class_factory(TargetDeltaPosControllerConfig)
        )

        @dataclass
        class PosControllerConfig:
            action_scale: float = 1.0
            action_lim_scale: float = 1.0
            action_clip_value: Union[float, Literal["joint_limits"]] = (
                "joint_limits"  # 100.0
            )

        pos_controller_config: PosControllerConfig = class_factory(PosControllerConfig)

        clip_torques: bool = True
        clip_torque_lim_scale: float = 1.0

        clip_torque_curriculum: bool = False
        clip_torque_lim_min: float = 1.0
        clip_torque_lim_max: float = 2.0
        clip_torque_level_up_threshold: int = 80
        clip_torque_degree: float = 0.000001

    @dataclass
    class Asset:
        robot_type: str = "g1_29dof_lock_legs_lock_waist"

        # NOTE (arth): essential to keep torso_link separate
        collapse_fixed_joints: bool = (
            True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        )
        replace_cylinder_with_capsule: bool = True
        flip_visual_attachments: bool = False
        armature: float = 0.001  # isaacgym armature is set by this
        thickness: float = 0.01
        max_angular_velocity: float = 1000.0
        max_linear_velocity: float = 1000.0
        density: float = 0.001
        angular_damping: float = 0.0
        linear_damping: float = 0.0

        asset_root: str = HUMANOID_RESOURCES_DIR / "robots/g1_new_hand"
        urdf_file: str = field(init=False)
        xml_file: str = field(init=False)
        self_collisions: int = 0  # 1 to disable, 0 to enable self collisions
        default_dof_drive_mode: int = 3

        ankle_sole_distance: float = 0.02

        fix_base_link: Optional[bool] = True
        disable_gravity: Optional[bool] = None
        filter_ints: Optional[List[int]] = None

    init_state: InitState = class_factory(InitState)
    motion: Motion = class_factory(Motion)
    control: Control = class_factory(Control)
    asset: Asset = class_factory(Asset)

    def __post_init__(self):
        assert self.name == "g1_29dof_lock_legs_lock_waist", "Wrong robot cfg"
        self.actions_dim = {"pos": 27, "target_delta_pos": 27, "ik": 26}[
            self.control.arm_control_mode
        ]
        self.asset.urdf_file = f"{self.name}_{self.used_hand}.urdf"
        self.asset.xml_file = f"{self.name}_{self.used_hand}.xml"
        assert self.used_hand == "right", f"{self.used_hand=} not supported"
        self.dof_names = self.right_arm_joint_names + self.finger_joint_names
        self.dof_pos_lower_limit_list = (
            self.right_arm_dof_pos_lower_limit_list
            + self.finger_dof_pos_lower_limit_list
        )
        self.dof_pos_upper_limit_list = (
            self.right_arm_dof_pos_upper_limit_list
            + self.finger_dof_pos_upper_limit_list
        )
        self.motion.upper_body_link = [
            "torso_link",
            "head_link",
            *self.right_arm_link_names,
            self.right_hand_name,
            *self.finger_link_names,
        ]
        self.body_names = [
            "pelvis",
            "torso_link",
            "head_link",
            *self.right_arm_link_names,
            self.right_hand_name,
            *self.finger_link_names,
        ]
        self.control.ik_controller_config.eef_pos_limits_lower = (
            self.control.ik_controller_config.right_eef_pos_limits_lower
        )
        self.control.ik_controller_config.eef_pos_limits_upper = (
            self.control.ik_controller_config.right_eef_pos_limits_upper
        )
