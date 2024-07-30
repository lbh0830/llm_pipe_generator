#!/bin/bash -x

model_path="path/to/model" # model directory

dataset_path="path/to/dataset" # .csv or .jsonl
output_path="path/to/output" # path to save the output file (.jsonl)
full_prompt_file_path="prompt/full_prompt/rephrase_prompt.txt" # path to the txt file. (Apply this additional prompt if not None.)
column_name="column_name" # the column used in the dataset
system_prompt="You are a helpful assistant." # the system prompt. if do not want to use, comment this line.
generation_config="path/to/generation_config.json" 
batch_size=8
# start_id=0 # start from item 0, comment this line if you want to use all the items in the dataset
# end_id=100 # end at item 100, comment this line if you want to use all the items in the dataset

python src/main.py \
    --model_path $model_path \
    --dataset_path $dataset_path \
    --output_path $output_path \
    --column_name $column_name \
    --generation_config $generation_config \
    --system_prompt "$system_prompt" \
    --batch_size $batch_size \
    --num_beams 1
    # --start_id $start_id \
    # --end_id $end_id \
    # --full_prompt_file_path $full_prompt_file_path