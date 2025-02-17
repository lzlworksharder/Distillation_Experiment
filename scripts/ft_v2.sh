root=.
python $root/main_tasks/ft.py \
    --dataset_name_or_path $root/src_data/rajpurkar/squad_v2 \
    --version_2_with_negative \
    --per_device_train_batch_size 24\
    --per_device_eval_batch_size 32\
    --model_name_or_path $root/models/microsoft/deberta-v3-xsmall\
    --learning_rate 5e-5\
    --num_train_epochs 3\
    --gradient_accumulation_steps 2\
    --project_name ft_xs\
    --output_dir $root/models/local_squadv2_finetuned_xs\
    --logging_dir $root/log

python $root/main_tasks/ft.py \
    --dataset_name_or_path $root/src_data/rajpurkar/squad_v2 \
    --version_2_with_negative \
    --per_device_train_batch_size 24\
    --per_device_eval_batch_size 32\
    --model_name_or_path $root/models/microsoft/deberta-v3-base\
    --learning_rate 3e-5\
    --num_train_epochs 3\
    --gradient_accumulation_steps 2\
    --project_name ft_base\
    --output_dir $root/models/local_squadv2_finetuned_base\
    --logging_dir $root/log




