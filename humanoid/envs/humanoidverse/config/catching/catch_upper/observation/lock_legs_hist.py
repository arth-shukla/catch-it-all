from dataclasses import dataclass
from typing import Dict, List

from humanoid.utils.dataclasses import class_factory, default_field


@dataclass
class G1CatchUpperLockLegsObservationHistConfig:
    @dataclass
    class ObservationDict:
        actor_obs: List[str] = default_field(
            [
                "dof_pos_finger_arm",
                "dof_vel_arm",
                "actions_finger_arm",
                "obj_dims_scaled",
                #
                # 1) 7D full pose
                # "obj_pose_wrt_camera",
                # 2) 3D clean position
                # "obj_pos_wrt_camera_clean",
                # 3) 3D noisy position
                "obj_pos_wrt_camera_noisy",
                #
                # wrt head obs (for teacher?)
                # "obj_pose_wrt_head",
                # "obj_pos_wrt_head",
                #
                "history_actor",
            ]
        )

        critic_obs: List[str] = default_field(
            [
                "dof_pos_finger_arm_raw",
                "dof_vel_arm_raw",
                "actions_finger_arm_raw",
                "obj_dims_scaled_raw",
                #
                # NOTE (arth): camera mounted to head, easier to get vels wrt head, fine for critic obs
                "obj_lin_vel_wrt_camera_raw",
                # "obj_ang_vel_wrt_camera_raw",
                #
                # 1) 7D full pose raw
                # "obj_pose_wrt_camera_raw",
                # 2) 3D clean position
                "obj_pos_wrt_camera_clean_raw",
                # 3) 3D noisy position
                # "obj_pos_wrt_camera_noisy_raw",
                #
                # "ctrl_delay_onehot_raw",
                "history_critic",
            ]
        )

        teacher_obs: List[str] = default_field(
            [
                "dof_pos_finger_arm",
                "dof_vel_finger_arm",
                "actions_finger_arm",
                "obj_dims_scaled",
                #
                # 1) 7D full pose
                # "obj_pose_wrt_camera",
                # 2) 3D clean position
                # "obj_pos_wrt_camera_clean",
                # 3) 3D noisy position
                # "obj_pos_wrt_camera_noisy",
                #
                # wrt head obs (for teacher?)
                # "obj_pose_wrt_head",
                "obj_pos_wrt_head",
            ]
        )

    # define those coumpounds in obs_dict, for example, you can define different long/short history with different length
    @dataclass
    class ObsAuxiliary:
        history_actor: Dict[str, int] = default_field(
            {
                "dof_pos_finger_arm": 10,
                "dof_vel_arm": 10,
                "actions_finger_arm": 10,
                # "obj_dims_scaled": 15,
                #
                # 1) 7D full pose
                # "obj_pose_wrt_camera": 15,
                # 2) 3D clean position
                # "obj_pos_wrt_camera_clean": 15,
                # 3) 3D noisy position
                "obj_pos_wrt_camera_noisy": 50,
                #
                # wrt head obs (for teacher?)
                # "obj_pose_wrt_head": 15,
                # "obj_pos_wrt_head": 15,
            }
        )

        history_critic: Dict[str, int] = default_field(
            {
                "dof_pos_finger_arm_raw": 10,
                "dof_vel_arm_raw": 10,
                "actions_finger_arm_raw": 10,
                # "obj_dims_scaled_raw": 15,
                #
                # NOTE (arth): camera mounted to head, easier to get vels wrt head, fine for critic obs
                "obj_lin_vel_wrt_camera_raw": 10,
                # "obj_ang_vel_wrt_camera_raw": 15,
                #
                # 1) 7D full pose raw
                # "obj_pose_wrt_camera_raw": 15,
                # 2) 3D clean position
                "obj_pos_wrt_camera_clean_raw": 50,
                # 3) 3D noisy position
                # "obj_pos_wrt_camera_noisy_raw": 15,
                #
                # "ctrl_delay_onehot_raw": 15,
            }
        )

    obs_scales: Dict[str, float] = default_field(
        {
            "base_lin_vel": 2.0,
            "base_ang_vel": 0.25,
            "projected_gravity": 1.0,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "actions": 1.0,
            #
            "history_actor": 1.0,
            "history_critic": 1.0,
            "history_teacher": 1.0,
            #
            "ctrl_delay_int": 1.0,
            "ctrl_delay_onehot": 1.0,
            #
            "dof_pos_finger_arm": 1.0,
            "dof_vel_finger_arm": 0.05,
            "dof_vel_arm": 0.05,
            "actions_finger_arm": 1.0,
            "obj_dims_scaled": 1.0,
            #
            "obj_pose_wrt_head": 1.0,
            "obj_pos_wrt_head": 1.0,
            "obj_lin_vel_wrt_head": 1.0,
            "obj_ang_vel_wrt_head": 1.0,
            #
            "obj_pose_wrt_camera": 1.0,
            "obj_pos_wrt_camera_clean": 1.0,
            "obj_pos_wrt_camera_noisy": 1.0,
            "obj_pos_wrt_camera_noisy_sam2": 1.0,
            "obj_lin_vel_wrt_camera": 1.0,
            "obj_ang_vel_wrt_camera": 1.0,
        }
    )

    noise_level: float = 1.0
    noise_scales: Dict[str, float] = default_field(
        {
            "base_lin_vel": 0.2,
            "base_ang_vel": 0.25,
            "projected_gravity": 0.05,
            "dof_pos": 0.02,
            "dof_vel": 0.1,
            "actions": 0.0,
            "history_actor": 0.0,
            "history_critic": 0.0,
            "history_teacher": 0.0,
            #
            "ctrl_delay_int": 0.0,
            "ctrl_delay_onehot": 0.0,
            #
            "dof_pos_finger_arm": 0.02,
            "dof_vel_finger_arm": 0.1,
            "dof_vel_arm": 0.1,
            "actions_finger_arm": 0.0,
            "obj_dims_scaled": 0.0,
            #
            "obj_pose_wrt_head": 0.0,
            "obj_pos_wrt_head": 0.0,
            "obj_lin_vel_wrt_head": 0.0,
            "obj_ang_vel_wrt_head": 0.0,
            #
            "obj_pose_wrt_camera": 0.0,
            "obj_pos_wrt_camera_clean": 0.0,
            "obj_pos_wrt_camera_noisy": 0.0,
            "obj_pos_wrt_camera_noisy_sam2": 0.0,
            "obj_lin_vel_wrt_camera": 0.0,
            "obj_ang_vel_wrt_camera": 0.0,
        }
    )

    # obs_dims should use list instead of dict
    # will be converted to dict in `pre_process_config`
    obs_dims: Dict[str, int] = default_field(
        {
            "base_lin_vel": 3,
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "dof_pos": "${robot.dof_obs_size}",
            "dof_vel": "${robot.dof_obs_size}",
            "actions": "${robot.dof_obs_size}",
            #
            "ctrl_delay_int": 1,
            "ctrl_delay_onehot": "${domain_rand.ctrl_delay_step_range}[1] + 1",
            #
            "dof_pos_finger_arm": "${robot.arm_dof_obs_size}",
            "dof_vel_finger_arm": "${robot.arm_dof_obs_size}",
            "dof_vel_arm": 7,
            "actions_finger_arm": "${robot.arm_dof_obs_size}",
            "obj_dims_scaled": 3,
            #
            "obj_pose_wrt_head": 7,
            "obj_pos_wrt_head": 3,
            "obj_lin_vel_wrt_head": 3,
            "obj_ang_vel_wrt_head": 3,
            #
            "obj_pose_wrt_camera": 7,
            "obj_pos_wrt_camera_clean": 3,
            "obj_pos_wrt_camera_noisy": 3,
            "obj_pos_wrt_camera_noisy_sam2": 3,
            "obj_lin_vel_wrt_camera": 3,
            "obj_ang_vel_wrt_camera": 3,
        }
    )

    obs_dict: ObservationDict = class_factory(ObservationDict)
    obs_auxiliary: ObsAuxiliary = class_factory(ObsAuxiliary)

    def __post_init__(self):
        for k in list(self.obs_dims.keys()):
            self.obs_dims[k + "_raw"] = self.obs_dims[k]
