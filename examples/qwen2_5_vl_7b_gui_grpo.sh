set -x
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
# MODEL_PATH='Ray2333/gui-planner-7B'
SYSTEM_PROMPT=""""""

~/.conda/envs/python3.10/bin/python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=datasets/GUI-R1/train_android_high_5000.parquet \
    data.val_files=datasets/GUI-R1/test_android_high_1000.parquet \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.structured_prompt=false \
    worker.actor.global_batch_size=128 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=r1gui \
    trainer.experiment_name=qwen2_5_vl_7b_guir1_grpo_trainhighandroid5k_validhigh1k_4gpu \
    trainer.n_gpus_per_node=4 \
    data.max_pixels=2109744 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.val_batch_size=256 \
    data.rollout_batch_size=128 \

# ~/.conda/envs/python3.10/bin/python -m verl.trainer.main \
#     config=examples/config.yaml \
#     data.train_files=datasets/GUI-R1/train_android_high_5000.parquet \
#     data.val_files=datasets/GUI-R1/test_android_high_1000.parquet \
#     data.system_prompt="${SYSTEM_PROMPT}" \
#     data.structured_prompt=true \
#     worker.actor.global_batch_size=128 \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     worker.rollout.enable_chunked_prefill=false \
#     worker.reward.compute_score=r1gui_grounders \
#     trainer.experiment_name=qwen2_5_vl_7b_planner_fromSFT_trainhighandroid5k_validhigh1k_4gpu \
#     trainer.n_gpus_per_node=4 \
#     data.max_pixels=2109744 \
#     data.max_prompt_length=4096 \
#     data.max_response_length=1024 \
#     data.val_batch_size=256 \
#     data.rollout_batch_size=128 \

#     trainer.save_checkpoint_path=/scratch/ry21/GUI-R1-ckpt \
#  = 2109744
# 92 * 52 * 28 * 28 = 3750656
    # data.train_files=datasets/GUI-R1/train.parquet \
    # data.val_files=datasets/GUI-R1/test.parquet \