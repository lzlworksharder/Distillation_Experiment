#!/bin/bash
root=.
python $root/main_tasks/ted.py \
    --dataset_name_or_path "$root\src_data\rajpurkar\squad_v2" \
    --version_2_with_negative \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 16\
    --model_type ted-deberta-v2 \
    --model_name_or_path $root/models/local_squadv2_student_stage1\
    --teacher_model_name_or_path $root/models/local_squadv2_teacher_stage1\
    --filter_interval 1\
    --teacher_filter_interval 1\
    --filter_output_dim 768 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 2\
    --num_train_epochs 3 \
    --kl_alpha 10 --mse_alpha 5000 \
    --project_name ted\
    --output_dir $root/models/local_ted_student_squadv2 \
    --seed 42  \
