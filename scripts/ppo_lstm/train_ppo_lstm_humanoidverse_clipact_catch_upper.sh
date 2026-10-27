#!/usr/bin/bash

SEED=68403
ALLOW_RESUME=True

# Task args
ENV_ID=humanoidverse-isaacgym-catch_upper_lock_legs-g1_27dof
WORKSPACE=humanoid_exps
EXP=teacher-underhand
GROUP=ppo_lstm
ENTITY=YOUR_WANDB_ENTITY
PROJECT_NAME=Humanoidverse-Catch-Upper-Lock-Legs-Real-Obs
EXP_NAME="$ENV_ID/$GROUP/$EXP"

LOG_ONLINE=True
WANDB=$LOG_ONLINE
NEPTUNE=False
TENSORBOARD=True

NUM_ENVS=16_384
MAX_EPISODE_STEPS=100

EXTRA_STAT_KEYS="[episode_task_reward, episode_pen_reward, average_episode_length, penalty_scale, clip_torque_lim_scale, throw_target_noise_scale, average_obj_in_hand_goal_once_rate, average_obj_in_hand_goal_static_once_rate, average_obj_near_hand_goal_obj_static_once_rate, average_obj_near_hand_goal_obj_static_at_end_rate, average_out_of_torque_lims, action_clip_frac, cum_num_below_min_rew, cum_num_above_max_rew]"

#############################################

resume_config="$WORKSPACE/$EXP_NAME/config.yml"
model_ckpt="$WORKSPACE/$EXP_NAME/models/latest.pt"
logger_state="$WORKSPACE/$EXP_NAME/logs/logger_state_dict.pkl"
logval() { python -c 'import pickle,sys; print(pickle.load(open(sys.argv[1],"rb"))["stats"][sys.argv[2]])' "$logger_state" "$1"; }
bestlogval() { python -c 'import pickle,sys; print(pickle.load(open(sys.argv[1],"rb"))["best_stats"][sys.argv[2]]["val"])' "$logger_state" "$1"; }
if [[ -f $resume_config && $ALLOW_RESUME = "True" ]]; then
    SEED=$((SEED+1))
fi
echo "SEED=$SEED"

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
    "algo.total_timesteps=100_000_000_000"
    "algo.extra_stat_keys=$EXTRA_STAT_KEYS"
    #
    "algo.rollout_steps=24"
    "algo.algorithm.desired_kl=0.01"
    "algo.algorithm.entropy_coef=0.0004"
    "algo.policy.init_noise_std=0.9"
    "algo.algorithm.bootstrap_on_trunc=True"
    "algo.algorithm.num_mini_batches=8"
    #
    "algo.frequency.log_freq=5_000_000"
    "algo.frequency.save_freq=10_000_000"
    "algo.frequency.save_backup_ckpts=True"
    # env simplifications
    "env.options_config.env.projectiles.included_objs=[apple, block, cuboid, cuboid22, cylinder, egg, tennisBall]"
    "env.options_config.robot.control.ik_controller_config.noised_dof_pos=False"
    #
    "env.options_config.env.toss_type=underhand"
    "env.options_config.robot.init_state.use_manual_arm_inits=True"
    "env.options_config.env.toss_noise_curriculum=False"
    "env.options_config.env.toss_noise_initial=0.1"
    "env.options_config.env.toss_noise_max=1.0"
    "env.options_config.env.toss_noise_degree=1e-5"
    "env.options_config.env.toss_noise_level_up_threshold=80"
    #
    "env.options_config.rewards.reward_penalty_curriculum=True"
    "env.options_config.rewards.reward_penalty_degree=5e-5"
    "env.options_config.rewards.reward_max_penalty_scale=12.5"
    "env.options_config.rewards.reward_penalty_level_up_threshold=80"
    #
    "env.options_config.rewards.reward_scales.hand_pos_rate=-0.1"
    "env.options_config.rewards.reward_scales.hand_pos_smoothness=-0.05"
    "env.options_config.rewards.reward_scales.hand_orientation=-0.1"
    #
    "env.options_config.robot.control.clip_torque_curriculum=True"
    "env.options_config.robot.control.clip_torque_degree=1e-5"
    "env.options_config.robot.control.clip_torque_level_up_threshold=70"
    #
    "env.options_config.robot.control.pos_controller_config.action_scale=1.0"
    "env.options_config.robot.control.pos_controller_config.action_lim_scale=1.0"
    "env.options_config.robot.control.pos_controller_config.action_clip_value=joint_limits"
    #
    "logger.best_stats_cfg={train/success_once: 1, train/success_at_end: 1, train/average_episode_length: 1, train/penalty_scale: 1, train/clip_torque_lim_scale: 1, train/throw_target_noise_scale: 1, train/raw_ep_return: 1, train/ep_return: 1, train/len: 1, train/average_obj_in_hand_goal_once_rate: 1, train/average_obj_in_hand_goal_static_once_rate: 1, train/average_obj_near_hand_goal_obj_static_once_rate: 1, train/average_obj_near_hand_goal_obj_static_at_end_rate: 1, train/average_out_of_torque_lims: -1}"
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
    SAPIEN_NO_DISPLAY=1 python -m humanoid.train_ppo_lstm "${resume_config}" \
        model_ckpt="${model_ckpt}" \
        logger.clear_out="False" \
        "${args[@]}" \
        env.options_config.rewards.reward_initial_penalty_scale="$(bestlogval "train/penalty_scale")" \
        env.options_config.robot.control.clip_torque_lim_scale="$(bestlogval "train/clip_torque_lim_scale")" \
        algo.algorithm.learning_rate="$(logval "losses_and_charts/learning_rate")"
else
    echo "STARTING..."
    SAPIEN_NO_DISPLAY=1 python -m humanoid.train_ppo_lstm configs/ppo_lstm.yml \
        logger.clear_out="True" \
        "${args[@]}"
fi

