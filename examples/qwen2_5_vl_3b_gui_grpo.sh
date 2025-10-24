set -x

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
# MODEL_PATH=/projects/illinois/eng/ece/huanz/ry21/GUI-R1/saved_cpts/qwen2_5_vl_3b_guir1_grpo_1080p_4gpu/global_step_55/actor/huggingface
SYSTEM_PROMPT=""""""

~/.conda/envs/python3.10/bin/python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=datasets/GUI-R1/train_high.parquet \
    data.val_files=datasets/GUI-R1/test_android_high_1000.parquet \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=128 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=r1gui_grounders \
    trainer.experiment_name=qwen2_5_vl_3b_planner_wUground_grpo_trainhigh_validhigh_4gpu \
    trainer.n_gpus_per_node=4 \
    data.max_pixels=2109744 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.val_batch_size=256 \
    data.rollout_batch_size=128 \


    #     trainer.load_checkpoint_path=/projects/illinois/eng/ece/huanz/ry21/GUI-R1/checkpoints/easy_r1/qwen2_5_vl_3b_guir1_grpo_1080p_4gpu/global_step_65 \
# data.max_pixels=2646000 \