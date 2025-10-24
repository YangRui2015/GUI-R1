# model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/saved_cpts/qwen2_5_vl_3b_guir1_grpo_1080p_4gpu/qwen2_5_vl_3b_guir1_grpo_1080p_4gpu_cpt55'
# model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/checkpoints/easy_r1/qwen2_5_vl_3b_guir1_grpo_1080p_4gpu/global_step_170/actor/huggingface'
# model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/saved_cpts/qwen2_5_vl_7b_guir1_grpo_1080p_4gpu/global_step_155'
# model_name='qwen2_5_vl_3b_guir1_grpo_1080p_4gpu_global_step_170_maxres1024_pngv2'
# model_name='qwen2_5_vl_3b_guir1_grpo_1080p_4gpu_global_step_170_maxres1024_pngv2'
# model_name='qwen2_5_vl_3b_guir1_grpo_1080p_4gpu_cpt55'
# model_name='qwen2_5_vl_7b_guir1_grpo_1080p_4gpu_cpt155'
# model_name='qwen2_5_vl_7b_guir1_grpo_1080p_4gpu_cpt155_maxres1024_pngv2'
# model_path='Ray2333/gui-planner-7B'
# model_name='gui-planner-7B'
model_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/checkpoints/easy_r1/qwen2_5_vl_7b_planner_wUground_grpo_1080p_4gpu/global_step_155/actor/huggingface'
model_name='qwen2_5_vl_7b_planner_wUground_grpo_step155'


data_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/datasets/GUI-R1'
output_path='/projects/illinois/eng/ece/huanz/ry21/GUI-R1/outputs'
n_gpus=4


type='low'
~/.conda/envs/python3.10/bin/python merge_file.py --directory $output_path/$model_name \
    --output $output_path/$model_name/androidcontrol_${type}_test.json \
    --inputs androidcontrol_${type}_test_0_1000_pred.json \
             androidcontrol_${type}_test_1000_2000_pred.json \
             androidcontrol_${type}_test_2000_3000_pred.json \
             androidcontrol_${type}_test_3000_4000_pred.json \
             androidcontrol_${type}_test_4000_5000_pred.json \
             androidcontrol_${type}_test_5000_6000_pred.json \
             androidcontrol_${type}_test_6000_7000_pred.json \
             androidcontrol_${type}_test_7000_8000_pred.json

# ~/.conda/envs/python3.10/bin/python eval/eval_omni.py --model_id $model_path --prediction_file_path  $output_path/$model_name/androidcontrol_high_test.json --resize_prediction
~/.conda/envs/python3.10/bin/python eval/eval_omni.py --model_id $model_path --prediction_file_path  $output_path/$model_name/androidcontrol_low_test.json --resize_prediction


# ~/.conda/envs/python3.10/bin/python merge_file.py --directory $output_path/$model_name \
#     --output $output_path/$model_name/guiodyssey_test.json \
#     --inputs guiodyssey_test_0_1000_pred.json \
#              guiodyssey_test_1000_2000_pred.json \
#              guiodyssey_test_2000_3000_pred.json \
#              guiodyssey_test_4000_5000_pred.json \
#              guiodyssey_test_6000_7000_pred.json \
#              guiodyssey_test_8000_9000_pred.json \
#              guiodyssey_test_9000_10000_pred.json \
#              guiodyssey_test_10000_11000_pred.json \
#                 guiodyssey_test_11000_12000_pred.json \
#                 guiodyssey_test_12000_13000_pred.json \
#                 guiodyssey_test_13000_14000_pred.json \
#                 guiodyssey_test_14000_15000_pred.json \
#                 guiodyssey_test_15000_16000_pred.json \
#                 guiodyssey_test_16000_17000_pred.json \
#                 guiodyssey_test_17000_18000_pred.json


# ~/.conda/envs/python3.10/bin/python eval/eval_omni.py --model_id $model_path --prediction_file_path  $output_path/$model_name/guiodyssey_test.json --resize_prediction
