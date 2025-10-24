# model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/checkpoints/easy_r1/qwen2_5_vl_3b_guir1_grpo_1080p_4gpu/global_step_170/actor/huggingface'
# model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/saved_cpts/qwen2_5_vl_7b_guir1_grpo_1080p_4gpu/global_step_155'
model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/checkpoints/easy_r1/qwen2_5_vl_3b_guir1_grpo_1080p_4gpu/global_step_170/actor/huggingface'
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120  # 单位秒
# export VLLM_LOGGING_LEVEL=DEBUG

gpu_id=0
port=8001
CUDA_VISIBLE_DEVICES=$gpu_id  \
vllm serve  $model_path \
    --host 127.0.0.1 \
    --port $port \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 10000 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --enable-chunked-prefill \
    --disable-log-requests    &

# #    

gpu_id=1
CUDA_VISIBLE_DEVICES=$gpu_id  \
vllm serve  $model_path \
    --host 127.0.0.1 \
    --port $((port+1)) \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 10000 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --enable-chunked-prefill \
    --disable-log-requests  

# #     --disable-log-requests \