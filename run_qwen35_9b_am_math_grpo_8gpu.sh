#!/usr/bin/env bash
set -xeuo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=${WANDB_API_KEY:-4e5b56432bc5eddfc57e326678c9dded10bbeedc}
export WANDB_MODE=${WANDB_MODE:-online}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export CUSTOM_REWARD_DEBUG_MAX_LINES=${CUSTOM_REWARD_DEBUG_MAX_LINES:-200}
export ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-/opt/tiger/workspace/rollout_traces/qwen35_9b_grpo_moredata_8gpu}

source /opt/tiger/workspace/venvs/verl-torch210/bin/activate

export DEVICE=gpu
export INFER_BACKEND=vllm
export MODEL_PATH=/opt/tiger/workspace/models/Qwen3.5-9B
export NGPUS_PER_NODE=8
export NNODES=1

export TRAIN_BATCH_SIZE=32
export PPO_MINI_BATCH_SIZE=32
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=32768
export PPO_MAX_TOKEN_LEN_PER_GPU=33024
export ACTOR_LR=1e-6
export KL_LOSS_COEF=0.001
export ENTROPY_COEFF=0
export ROLLOUT_TP=8
export ROLLOUT_GPU_MEM_UTIL=0.45
export ROLLOUT_N=8
export TOTAL_EPOCHS=1
export SAVE_FREQ=1000
export TEST_FREQ=1000
export PROJECT_NAME=verl_grpo_moredata
export EXPERIMENT_NAME=qwen35_9b_grpo_moredata_8gpu
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  reward.custom_reward_function.path=/opt/tiger/workspace/custom_reward_schema.py \
  reward.custom_reward_function.name=compute_score \
  data.train_files="['/opt/tiger/workspace/data_community/am_math_filtered/train.parquet']" \
  data.val_files="['/opt/tiger/workspace/data_community/am_math_filtered/val.parquet']" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=${INFER_BACKEND} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.balance_batch=True \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
  trainer.nnodes=${NNODES} \
  trainer.rollout_data_dir=${ROLLOUT_DATA_DIR} \
  trainer.resume_mode=disable \
  trainer.val_before_train=${VAL_BEFORE_TRAIN} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.total_epochs=${TOTAL_EPOCHS}
