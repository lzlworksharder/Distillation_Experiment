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
    --filter_disabled \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2\
    --kl_alpha 10 --mse_alpha 100 \
    --project_name lwd\
    --output_dir $root/models/local_lwd_student_squadv2 \
    --seed 42  \