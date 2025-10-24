export HF_TOKEN='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'
# model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/saved_cpts/qwen2_5_vl_3b_guir1_grpo_1080p_4gpu/qwen2_5_vl_3b_guir1_grpo_1080p_4gpu_cpt55'
# model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/checkpoints/easy_r1/qwen2_5_vl_3b_guir1_grpo_1080p_4gpu/global_step_170/actor/huggingface'
# model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/saved_cpts/qwen2_5_vl_7b_guir1_grpo_1080p_4gpu/global_step_155'
# model_path='Ray2333/gui-planner-7B'
model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/checkpoints/easy_r1/qwen2_5_vl_7b_planner_wUground_grpo_1080p_4gpu/global_step_155/actor/huggingface'
# model_name='qwen2_5_vl_7b_guir1_grpo_1080p_4gpu_global_step_155_maxres1024_pngv2'
# model_name='qwen2_5_vl_3b_guir1_grpo_1080p_4gpu_cpt55'
# model_name='qwen2_5_vl_7b_guir1_grpo_1080p_4gpu_cpt155'
# model_name='gui-planner-7B-v2'
model_name='qwen2_5_vl_7b_planner_wUground_grpo_step155'


data_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/datasets/GUI-R1'
output_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/outputs'
n_gpus=4


# ~/.conda/envs/python3.10/bin/python inference/inference_vllm_android_single_debug.py --model_path $model_path --data_path $data_path/androidcontrol_high_test.parquet --output_path $output_path/$model_name --start_index 0 --end_index 10

# ~/.conda/envs/python3.10/bin/python inference/inference_vllm_screenspot_single.py --model_path $model_path \
#                 --data_path $data_path/screenspot_pro_test.parquet --num_gpus $n_gpus --output_path $output_path/$model_name \
#                 --start_index 0 --end_index -1  --async_concurrency 4 
# ~/.conda/envs/python3.10/bin/python eval/eval_screenspot.py --model_id $model_path  --prediction_file_path $output_path/$model_name/screenspot_pro_test.json

# ~/.conda/envs/python3.10/bin/python inference/inference_vllm_screenspot_single.py --model_path $model_path --data_path $data_path/screenspot_test.parquet --num_gpus $n_gpus --output_path $output_path/$model_name
# ~/.conda/envs/python3.10/bin/python  eval/eval_screenspot.py --model_id $model_path  --prediction_file_path $output_path/$model_name/screenspot_test.json


# ~/.conda/envs/python3.10/bin/python inference/inference_vllm_android_single.py --model_path $model_path --data_path $data_path/androidcontrol_high_test.parquet --num_gpus $n_gpus --output_path $output_path/$model_name --start_index 0 --end_index 2000
# ~/.conda/envs/python3.10/bin/python eval/eval_omni.py --model_id $model_path --prediction_file_path  $output_path/$model_name/androidcontrol_high_test.json
# ~/.conda/envs/python3.10/bin/python inference/inference_vllm_android_single.py --model_path $model_path --data_path $data_path/androidcontrol_high_test.parquet --num_gpus $n_gpus --output_path $output_path/$model_name --start_index 7000 --end_index 8000


# ~/.conda/envs/python3.10/bin/python inference/inference_vllm_android_single.py --model_path $model_path --data_path $data_path/androidcontrol_high_test.parquet --num_gpus $n_gpus --output_path $output_path/$model_name --start_index 4000 --end_index 6000

# ~/.conda/envs/python3.10/bin/python inference/inference_vllm_android_single.py --model_path $model_path --data_path $data_path/androidcontrol_high_test.parquet --num_gpus $n_gpus --output_path $output_path/$model_name --start_index 6000 --end_index 8000
#
# ~/.conda/envs/python3.10/bin/python inference/inference_vllm_android_single.py --model_path $model_path --data_path $data_path/androidcontrol_low_test.parquet  --num_gpus $n_gpus --output_path $output_path/$model_name
# ~/.conda/envs/python3.10/bin/python eval/eval_omni.py --model_id $model_path --prediction_file_path  $output_path/$model_name/androidcontrol_low_test.json


# for i in {3..7}
# do
#     start_index=$((i * 1000))
#     end_index=$((start_index + 1000))
#     echo "Running inference from index $start_index to $end_index"
#     ~/.conda/envs/python3.10/bin/python inference/inference_vllm_android_single_debug_v2.py --model_path $model_path --data_path $data_path/androidcontrol_high_test.parquet --num_gpus $n_gpus --output_path $output_path/$model_name --start_index $start_index --end_index $end_index
# done

for i in {0..7}
do
    start_index=$((i * 1000))
    end_index=$((start_index + 1000))
    echo "Running inference from index $start_index to $end_index"
    ~/.conda/envs/python3.10/bin/python inference/inference_vllm_android_single_debug_v2.py --model_path $model_path --data_path $data_path/androidcontrol_low_test.parquet --num_gpus $n_gpus --output_path $output_path/$model_name --start_index $start_index --end_index $end_index
done

# n_gpus=1
# for i in {12..17}
# do
#     start_index=$((i * 1000))
#     end_index=$((start_index + 1000))
#     echo "Running inference from index $start_index to $end_index"
#     ~/.conda/envs/python3.10/bin/python inference/inference_vllm_guiodyssey_single_debug.py --model_path $model_path --data_path $data_path/guiodyssey_test.parquet --num_gpus $n_gpus --output_path $output_path/$model_name --start_index $start_index --end_index $end_index --start_port 23333
# done




# write a loop to run the following two commands for 4 times with different start_index and end_index
# for i in {5..7}
# do
#     start_index=$((i * 1000))
#     end_index=$((start_index + 1000))
#     echo "Running inference from index $start_index to $end_index"
#     ~/.conda/envs/python3.10/bin/python inference/inference_vllm_android_single.py --model_path $model_path --data_path $data_path/androidcontrol_high_test.parquet --num_gpus $n_gpus --output_path $output_path/$model_name --start_index $start_index --end_index $end_index
# done

# type='low'
# ~/.conda/envs/python3.10/bin/python merge_file.py --directory $output_path/$model_name \
#     --output $output_path/$model_name/androidcontrol_${type}_test.json \
#     --inputs androidcontrol_${type}_test_0_1000_pred.json \
#              androidcontrol_${type}_test_1000_2000_pred.json \
#              androidcontrol_${type}_test_2000_3000_pred.json \
#              androidcontrol_${type}_test_3000_4000_pred.json \
#              androidcontrol_${type}_test_4000_5000_pred.json \
#              androidcontrol_${type}_test_5000_6000_pred.json \
#              androidcontrol_${type}_test_6000_7000_pred.json \
#              androidcontrol_${type}_test_7000_8000_pred.json
# ~/.conda/envs/python3.10/bin/python eval/eval_omni.py --model_id $model_path --prediction_file_path  $output_path/$model_name/androidcontrol_high_test.json
# ~/.conda/envs/python3.10/bin/python eval/eval_omni.py --model_id $model_path --prediction_file_path  $output_path/$model_name/androidcontrol_low_test.json

# module unload python/3.10.16 2>/dev/null || true
# ~/.conda/envs/python3.10/bin/python inference/inference_vllm_guiodyssey_single.py --model_path $model_path --data_path $data_path/guiodyssey_test.parquet --num_gpus $n_gpus --output_path $output_path/$model_name

# for i in {8..18}
# do
#     start_index=$((i * 1000))
#     end_index=$((start_index + 1000))
#     echo "Running inference from index $start_index to $end_index"
#     ~/.conda/envs/python3.10/bin/python inference/inference_vllm_guiodyssey_single.py --model_path $model_path --data_path $data_path/guiodyssey_test.parquet --num_gpus $n_gpus --output_path $output_path/$model_name --start_index $start_index --end_index $end_index --start_port 8001
# done


# python guir1/eval/eval_omni.py --model_id $model_path  --prediction_file_path $output_path/$model_name/guiodyssey_test.json


