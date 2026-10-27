#!/usr/bin/bash

TOSS_TYPE=overhand
CKPT_PATH="humanoid_checkpoints/humanoidverse-isaacgym-catch_upper_lock_legs-g1_27dof-teacher-$TOSS_TYPE-right-taucurr.pt"
CLIP_TORQUES=False

ENV_ID=humanoidverse-isaacgym-catch_upper_lock_legs-g1_27dof
MAX_EPISODE_STEPS=100

SEED=0

# Task args
EXP=evaluate
PROJECT_NAME="humanoidverse"

GROUP=evaluate
ENTITY=YOUR_WANDB_ENTITY
EXP_NAME="$ENV_ID/$GROUP/$EXP"
WORKSPACE=humanoid_exps

TENSORBOARD=True

RECORD_VIDEO=True
if [ "$RECORD_VIDEO" = "True" ]; then
    NUM_ENVS=4
    INFO_ON_VIDEO=True
    RENDER_MODE=rgb_array
else
    NUM_ENVS=32_768
    INFO_ON_VIDEO=False
    RENDER_MODE=null
fi

NUM_EVAL_STEPS=$((MAX_EPISODE_STEPS))

EXTRA_STAT_KEYS="[episode_out_of_torque_lims, episode_task_reward, episode_pen_reward, average_episode_length, penalty_scale, clip_torque_lim_scale, throw_target_noise_scale, average_obj_in_hand_goal_once_rate, average_obj_in_hand_goal_static_once_rate, average_obj_near_hand_goal_obj_static_once_rate, average_obj_near_hand_goal_obj_static_at_end_rate, average_out_of_torque_lims, action_clip_frac, cum_num_below_min_rew, cum_num_above_max_rew]"

#############################################

args=(
    "ckpt_path=$CKPT_PATH"
    "seed=$SEED"
    #
    "num_eval_steps=$NUM_EVAL_STEPS"
    "extra_stat_keys=$EXTRA_STAT_KEYS"
    "env.reward_mode=dense"
    #
    # "env.sim_backend=cpu"
    "env.suite=humanoidverse"
    "env.env_id=$ENV_ID"
    "env.num_envs=$NUM_ENVS"
    "env.max_episode_steps=$MAX_EPISODE_STEPS"
    "env.render_mode=$RENDER_MODE"
    "env.save_video=$RECORD_VIDEO"
    "env.info_on_video=$INFO_ON_VIDEO"
    #
    "env.options_config.env.init_at_random_ep_len=False"
    "env.options_config.env.termination.ignore_termination=True"
    "env.options_config.env.projectiles.included_objs=[apple, banana, binoculars, block, book, cuboid, cuboid22, cylinder, egg, flashlight, flute, hammer, lightbulb, mug, phone, pyramidmedium, rubberduck, tennisBall, waterbottle, wristwatch]"
    #
    "env.options_config.robot.control.ik_controller_config.noised_dof_pos=True"
    #
    "env.options_config.robot.control.pos_controller_config.action_scale=1.0"
    "env.options_config.robot.control.pos_controller_config.action_lim_scale=1.0"
    "env.options_config.robot.control.pos_controller_config.action_clip_value=joint_limits"
    #
    "env.options_config.robot.control.clip_torques=$CLIP_TORQUES"
    "env.options_config.env.toss_type=$TOSS_TYPE"
    "env.options_config.robot.init_state.use_manual_arm_inits=True"
    "env.options_config.env.toss_noise_curriculum=False"
    "env.options_config.env.headless=True"
    #
    # "env.options_config.env.termination.terminate_when_close_to_torque_limit=True"
    # "env.options_config.env.termination.scales.termination_close_to_torque_limit=0.999"
    # "env.options_config.env.termination.terminate_by_obj_height=False"
    # "env.options_config.env.termination.ignore_termination=True"
    # "env.options_config.env.termination.processed_actions_lock_arm_post_term=True"
    # "env.options_config.env.termination.scales.termination_obj_height=0.6"
    #
    "logger.best_stats_cfg={train/average_episode_length: 1, train/penalty_scale: 1, train/clip_torque_lim_scale: 1, train/throw_target_noise_scale: 1, train/raw_ep_return: 1, train/ep_return: 1, train/len: 1, train/average_obj_in_hand_goal_once_rate: 1, train/average_obj_in_hand_goal_static_once_rate: 1, train/average_obj_near_hand_goal_obj_static_once_rate: 1, train/average_obj_near_hand_goal_obj_static_at_end_rate: 1, train/average_out_of_torque_lims: -1}"
    "logger.group_name=$GROUP"
    "logger.exp_name=EVAL--$EXP_NAME"
    "logger.entity_name=$ENTITY"
    "logger.project_name=$PROJECT_NAME"
    "logger.workspace=$WORKSPACE"
    "logger.wandb=False"
    "logger.tensorboard=$TENSORBOARD"
    "logger.neptune=False"
    "logger.force_flush_internal_prints=True"
    "logger.clear_out=False"
)

SAPIEN_NO_DISPLAY=1 python -m humanoid.evaluate configs/evaluate.yml \
    "${args[@]}"
