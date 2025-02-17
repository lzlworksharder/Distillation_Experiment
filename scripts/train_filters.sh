root=.
python $root/main_tasks/train_filters.py \
    --dataset_name_or_path $root/src_data/rajpurkar/squad_v2 \
    --version_2_with_negative \
    --per_device_train_batch_size 24\
    --model_name_or_path $root/models/local_squadv2_finetuned_base\
    --model_type ted-deberta-v2\
    --filter_interval 1 \
    --learning_rate 3e-5\
    --num_train_epochs 3\
    --gradient_accumulation_steps 2\
    --project_name train_teacher_filters\
    --output_dir $root/models/local_squadv2_teacher_stage1\

python $root/main_tasks/train_filters.py \
    --dataset_name_or_path $root/src_data/rajpurkar/squad_v2 \
    --version_2_with_negative \
    --per_device_train_batch_size 24\
    --model_name_or_path $root/models/local_squadv2_finetuned_xs\
    --model_type ted-deberta-v2\
    --filter_interval 1 \
    --learning_rate 5e-5\
    --num_train_epochs 3\
    --gradient_accumulation_steps 2\
    --project_name train_student_filters\
    --output_dir $root/models/local_squadv2_student_stage1\