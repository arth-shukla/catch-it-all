#!/usr/bin/bash

TOSS_TYPE=overhand
CKPT_PATH="humanoid_checkpoints/humanoidverse-isaacgym-catch_upper_lock_legs_hist-g1_27dof-student-$TOSS_TYPE-right-taucurr.pt"
CLIP_TORQUES=False

MAX_EPISODE_STEPS=100

SEED=0

# Task args
ENV_ID=mujoco-catch-upper-lock-legs-hist-v0
PROJECT_NAME="HumanoidVerse-PPO-mujoco"

GROUP=evaluate
ENTITY=YOUR_WANDB_ENTITY
EXP_NAME="$ENV_ID/$GROUP/$EXP"
WORKSPACE=humanoid_exps
# 
TENSORBOARD=True

RECORD_VIDEO=True
if [ "$RECORD_VIDEO" = "True" ]; then
    NUM_ENVS=4
    INFO_ON_VIDEO=True
    RENDER_MODE=rgb_array
else
    NUM_ENVS=64
    INFO_ON_VIDEO=False
    RENDER_MODE=null
fi

NUM_EVAL_STEPS=$((MAX_EPISODE_STEPS*4))

#############################################

args=(
    "ckpt_path=$CKPT_PATH"
    "seed=$SEED"
    #
    "num_eval_steps=$NUM_EVAL_STEPS"
    #
    "env.suite=mujoco"
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
    "env.options_config.env.projectiles.ensure_obj_in_frame=True"
    #
    "env.options_config.domain_rand.cam_latency_range=\${as_tuple:1,2}"
    #
    "env.options_config.robot.control.pos_controller_config.action_scale=1.0"
    "env.options_config.robot.control.pos_controller_config.action_lim_scale=1.0"
    "env.options_config.robot.control.pos_controller_config.action_clip_value=joint_limits"
    #
    "env.options_config.robot.control.clip_torques=$CLIP_TORQUES"
    "env.options_config.env.toss_type=$TOSS_TYPE"
    #
    "logger.best_stats_cfg={train/ep_return: 1, train/len: 1}" \
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

MUJOCO_GL=egl SAPIEN_NO_DISPLAY=1 python -m humanoid.evaluate configs/evaluate.yml \
    "${args[@]}"
