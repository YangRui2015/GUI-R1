# model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/checkpoints/easy_r1/qwen2_5_vl_3b_guir1_grpo_1080p_4gpu/global_step_170/actor/huggingface'
# model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/saved_cpts/qwen2_5_vl_7b_guir1_grpo_1080p_4gpu/global_step_155'
# model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/checkpoints/easy_r1/qwen2_5_vl_3b_guir1_grpo_1080p_4gpu/global_step_170/actor/huggingface'
# model_path='Ray2333/gui-planner-7B'
model_path='osunlp/UGround-V1-7B'
# model_path='/home/ry21/GUI-R1/checkpoints/easy_r1/qwen2_5_vl_3b_guiplanner_grpo/global_step_90/actor/huggingface'
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120  # 单位秒
# export VLLM_LOGGING_LEVEL=DEBUG

gpu_id=0
port=8005
CUDA_VISIBLE_DEVICES=$gpu_id  \
vllm serve  $model_path \
    --host 127.0.0.1 \
    --port $port \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 6000 \
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
    --gpu-memory-utilization 0.3 \
    --max-model-len 6000 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --enable-chunked-prefill \
    --disable-log-requests  &


gpu_id=2
CUDA_VISIBLE_DEVICES=$gpu_id  \
vllm serve  $model_path \
    --host 127.0.0.1 \
    --port $((port+2)) \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 6000 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --enable-chunked-prefill \
    --disable-log-requests  &



gpu_id=3
CUDA_VISIBLE_DEVICES=$gpu_id  \
vllm serve  $model_path \
    --host 127.0.0.1 \
    --port $((port+3)) \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 6000 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --enable-chunked-prefill \
    --disable-log-requests  