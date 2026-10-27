#!/usr/bin/bash

CKPT_PATH="humanoid_checkpoints/humanoidverse-isaacgym-catch_upper_lock_legs-g1_27dof-teacher-overhand-right-taucurr.pt"

SEED=2024
ALLOW_RESUME=True

# Task args
ENV_ID=humanoidverse-isaacgym-catch_upper_lock_legs_hist-g1_27dof
WORKSPACE=humanoid_exps
EXP=student-overhand
GROUP=dagger
ENTITY=YOUR_WANDB_ENTITY
PROJECT_NAME=Humanoidverse-Catch-Upper-Lock-Legs-Real-Obs
EXP_NAME="$ENV_ID/$GROUP/$EXP"

LOG_ONLINE=True
WANDB=$LOG_ONLINE
NEPTUNE=False
TENSORBOARD=True

NUM_ENVS=4096
MAX_EPISODE_STEPS=100

EXTRA_STAT_KEYS="[episode_task_reward, episode_pen_reward, average_episode_length, penalty_scale, clip_torque_lim_scale, throw_target_noise_scale, average_obj_in_hand_goal_once_rate, average_obj_in_hand_goal_static_once_rate, average_obj_near_hand_goal_obj_static_once_rate, average_obj_near_hand_goal_obj_static_at_end_rate, average_out_of_torque_lims, action_clip_frac, cum_num_below_min_rew, cum_num_above_max_rew]"

#############################################

resume_config="$WORKSPACE/$EXP_NAME/config.yml"
model_ckpt="$WORKSPACE/$EXP_NAME/models/latest.pt"
if [[ -f $resume_config && $ALLOW_RESUME = "True" ]]; then
    SEED=$((SEED+1))
fi
echo "SEED=$SEED"

STEPS_PER_ITER=$MAX_EPISODE_STEPS
UPDATES_PER_ITER=$MAX_EPISODE_STEPS
args=(
    "seed=$SEED"
    #
    "env.env_id=$ENV_ID"
    "env.suite=humanoidverse"
    "env.num_envs=$NUM_ENVS"
    "env.max_episode_steps=$MAX_EPISODE_STEPS"
    "env.render_mode=null"
    "env.reward_mode=dense"
    #
    "algo.loss=rmse"
    "algo.max_grad_norm=0.2"
    "algo.data_path=$CKPT_PATH"
    "algo.num_iterations=100_000"
    "algo.buffer_size=5_000_000"
    "algo.buffer_device=cuda"
    "algo.extra_stat_keys=$EXTRA_STAT_KEYS"
    #
    "algo.muon_lr=0.02"
    "algo.adam_lr=1e-3"
    "algo.scheduler_gamma=0.995"
    "algo.batch_size=$((NUM_ENVS*2))"
    "algo.collect_steps_per_iter=$STEPS_PER_ITER"
    "algo.updates_per_iter=$UPDATES_PER_ITER"
    "algo.loss_lower_bound=0.0"
    #
    "algo.log_freq=$((NUM_ENVS*STEPS_PER_ITER*10))"
    "algo.save_freq=$((NUM_ENVS*STEPS_PER_ITER*10))"
    "algo.save_backup_ckpts=True"
    # env simplifications
    "env.options_config.env.projectiles.included_objs=[apple, block, cuboid, cuboid22, cylinder, egg, tennisBall]"
    #
    "env.options_config.robot.control.ik_controller_config.noised_dof_pos=False"
    #
    "env.options_config.rewards.reward_scales.arm_default_pos=-2.5"
    "env.options_config.rewards.reward_penalty_reward_names=[finger_joint_tracking_error, finger_action_rate, finger_smoothness, joint_power_hand, action_vanish_hand, dof_vel_hand, dof_acc_hand, torques_hand, dof_pos_hand_limits, dof_vel_hand_limits, torque_hand_limits, arm_joint_tracking_error, arm_action_rate, arm_smoothness, joint_power_arm, action_vanish_arm, dof_vel_arm, dof_acc_arm, torques_arm, dof_pos_arm_limits, dof_vel_arm_limits, torque_arm_limits, hand_pos_rate, hand_pos_smoothness, hand_orientation]"
    "env.options_config.env.projectiles.ensure_obj_in_frame=True"
    #
    "env.options_config.robot.control.pos_controller_config.action_scale=1.0"
    "env.options_config.robot.control.pos_controller_config.action_lim_scale=1.0"
    "env.options_config.robot.control.pos_controller_config.action_clip_value=joint_limits"
    #
    "env.options_config.robot.control.clip_torques=False"
    "env.options_config.env.toss_type=overhand"
    "env.options_config.robot.init_state.use_manual_arm_inits=True"
    "env.options_config.env.toss_noise_curriculum=False"
    "env.options_config.env.headless=True"
    #
    "env.options_config.domain_rand.cam_latency_range=\${as_tuple:1,2}"
    #
    # NOTE (arth): no curriculums for dagger
    "env.options_config.rewards.reward_penalty_curriculum=False"
    "env.options_config.domain_rand.init_upper_ratio=1.0"
    "env.options_config.env.action_curriculum=False"
    "env.options_config.obs.noise_curriculum=False"
    #
    "logger.best_stats_cfg={train/average_episode_length: 1, train/penalty_scale: 1, train/clip_torque_lim_scale: 1, train/throw_target_noise_scale: 1, train/raw_ep_return: 1, train/ep_return: 1, train/len: 1, train/average_obj_in_hand_goal_once_rate: 1, train/average_obj_in_hand_goal_static_once_rate: 1, train/average_obj_near_hand_goal_obj_static_once_rate: 1, train/average_obj_near_hand_goal_obj_static_at_end_rate: 1, train/average_out_of_torque_lims: -1}"
    "logger.group_name=$GROUP"
    "logger.exp_name=$EXP_NAME"
    "logger.entity_name=$ENTITY"
    "logger.project_name=$PROJECT_NAME"
    "logger.workspace=$WORKSPACE"
    "logger.wandb=$WANDB"
    "logger.tensorboard=$TENSORBOARD"
    "logger.neptune=$NEPTUNE"
    "logger.force_flush_internal_prints=True"
)

if [[ -f $resume_config && $ALLOW_RESUME = "True" ]]; then
    echo "RESUMING $resume_config..."
    SAPIEN_NO_DISPLAY=1 python -m humanoid.train_dagger "${resume_config}" \
        model_ckpt="${model_ckpt}" \
        logger.clear_out="False" \
        "${args[@]}"
else
    echo "STARTING..."
    SAPIEN_NO_DISPLAY=1 python -m humanoid.train_dagger configs/dagger.yml \
        logger.clear_out="True" \
        "${args[@]}"
fi
