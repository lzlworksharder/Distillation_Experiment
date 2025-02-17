    #  import packages
import argparse
import json
import math
import os
import torch
from torch.utils.data import RandomSampler,SequentialSampler
from accelerate import Accelerator
from accelerate.utils import set_seed,ProjectConfiguration
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)
from models.ted_deberta_v2 import TEDDebertaV2Config,TEDDebertaV2ForQuestionAnswering

MODEL_MAPPING.register(TEDDebertaV2Config, TEDDebertaV2ForQuestionAnswering)
CONFIG_MAPPING = {
    "ted-deberta-v2": TEDDebertaV2Config,
}
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


    #  some util functions
def save_prefixed_metrics(results, output_dir, file_name: str = "all_results.json", metric_key_prefix: str = "eval"):
    """
    Save results while prefixing metric names.

    Args:
        results: (:obj:`dict`):
            A dictionary of results.
        output_dir: (:obj:`str`):
            An output directory.
        file_name: (:obj:`str`, `optional`, defaults to :obj:`all_results.json`):
            An output file name.
        metric_key_prefix: (:obj:`str`, `optional`, defaults to :obj:`eval`):
            A metric name prefix.
    """
    # Prefix all keys with metric_key_prefix + '_'
    for key in list(results.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            results[f"{metric_key_prefix}_{key}"] = results.pop(key)

    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(results, f, indent=4)
def print_model_info(model):
    '''
    print device and model info before training
    '''
    device = model.device
    gpu_available = torch.cuda.is_available()
    tpu_available = "XLA_FLAGS" in os.environ  
    local_rank = int(os.getenv('LOCAL_RANK', 0))  
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '[0]')  

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    estimated_model_size_mb = total_params * 4 / 1024 / 1024 

    print(f"GPU available: {gpu_available} (cuda), model device: {device}")
    print(f"TPU available: {tpu_available}, using: {0 if not tpu_available else 1} TPU cores")
    print(f"LOCAL_RANK: {local_rank} - CUDA_VISIBLE_DEVICES: {cuda_visible_devices}\n")
    
    print(f" | Name      | Type                             | Params")
    print("-" * 66)
    print(f" | model     | {model.__class__.__name__.ljust(32)} | {total_params / 1e6:.1f} M")
    print("-" * 66)
    print(f"{trainable_params / 1e6:.1f} M    Trainable params")
    print(f"{non_trainable_params/ 1e6:.1f} M    Non-trainable params")
    print(f"{total_params / 1e6:.1f} M    Total params")
    print(f"{estimated_model_size_mb:.3f}   Total estimated model params size (MB)")

    #  parse args function
def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for model training.")

    # data
    # load data:
    parser.add_argument('--processed_train_data_dir', type=str, help='Processed training data directory',default=None)
    parser.add_argument('--processed_validation_data_dir', type=str, help='Processed validation data directory',default=None)
    parser.add_argument('--dataset_name_or_path', type=str, help='Dataset name', default=r'src_data\rajpurkar\squad')
    parser.add_argument('--version_2_with_negative', action='store_true', help='Whether to use version 2 with negative', default=False)

    # pre-processing
    parser.add_argument('--max_seq_length', type=int, help='Max sequence length', default=384)
    parser.add_argument('--pad_to_max_length', type=bool, help='Whether to pad to max length', default=True)
    parser.add_argument('--doc_stride', type=int, help='Document stride', default=128)
    parser.add_argument('--max_answer_length', type=int, help='Max answer length', default=30)
    parser.add_argument('--overwrite_cache',action='store_true',help='Whether to overwrite preprocessed data cache when tokenizing')
    # dataloader: batchsize + num_workers + num_samples
    parser.add_argument('--max_train_samples', type=int, help='Max train samples', default=None)
    parser.add_argument('--max_validation_samples', type=int, help='Max validation samples', default=None)
    parser.add_argument('--per_device_train_batch_size', type=int, help='Per device train batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, help='Per device eval batch size', default=8)
    parser.add_argument('--preprocessing_num_workers', type=int, help='Number of workers', default=2)

    # model
    # load model
    parser.add_argument('--model_name_or_path', type=str, help='Model name or path')
    # forward
    parser.add_argument('--null_score_diff_threshold', type=float, help='Null score diff threshold', default=0.0)
    parser.add_argument('--n_best_size', type=int, help='N best size', default=20)
    #TED Parameters
    parser.add_argument(
        "--checkpoint_path",
        type=str, 
        default=None, 
        help="path to load the properly initialized teacher or student checkpoint.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Model type to use.",
        choices=MODEL_TYPES,
        required=True,
    )
    parser.add_argument(
        "--filter_interval",
        type=int,
        default=1,
        help="the layer interval between two consecutive filters.",
    )
    parser.add_argument(
        "--filter_output_dim", 
        type=int, 
        default=None, 
        help="the output dimension of each filter. Default as the input dimension.")
    parser.add_argument(
        "--filter_nonlinear", 
        action="store_true", 
        help="whether to add a non-linear activation in each filter.") 

    # training
    # optimizer(lr + wd) & scheduler(type + warmup)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=5e-5)
    parser.add_argument('--weight_decay', type=float, help='Weight decay', default=0)
    parser.add_argument('--lr_scheduler_type', type=str, help='Learning rate scheduler type', default='linear')
    parser.add_argument('--num_warmup_steps', type=int, help='Number of warmup steps', default=0)

    # trainer: epochs, accumulation, precision
    parser.add_argument('--num_train_epochs', type=int, help='Number of training epochs', default=3)
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps', default=1)
    parser.add_argument('--mixed_precision', type=str, help='Mixed precision', default='fp16')

    # metrics & logging
    parser.add_argument('--logging_dir', type=str, help='Logging directory', default='./log')
    parser.add_argument('--report_to', type=str, help='Reporting method', default='tensorboard')
    parser.add_argument('--with_tracking', type=bool, help='Whether to track', default=True)
    parser.add_argument('--logging_steps', type=int, help='Logging steps', default=5)
    parser.add_argument('--project_name', type=str, help='Project name',default='project')

    # ckpt
    parser.add_argument('--save_best', type=bool, help='Whether to save the best model', default=True)
    parser.add_argument('--output_dir', type=str, help='Output directory',default='./output')
    parser.add_argument('--checkpointing_steps', type=str, help='Checkpointing steps', default='epoch')

    # seed
    parser.add_argument('--seed', type=int, help='Random seed', default=42)

    args = parser.parse_args()
    return args

    # main
def main():
    # parse args & set seed
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    # accelerator

    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        config = ProjectConfiguration(project_dir='./',logging_dir=args.logging_dir)
        accelerator_log_kwargs["project_config"] = config
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, 
                            mixed_precision=args.mixed_precision,
                            **accelerator_log_kwargs
                            )
    if args.with_tracking:
        experiment_config = vars(args)
        accelerator.init_trackers(args.project_name, experiment_config)

    # load data & pre-processing
    raw_datasets = load_dataset(args.dataset_name_or_path)
    column_names = raw_datasets['train'].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    pad_on_right=True
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=args.max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
  

    # 获取train_dataset
    if args.processed_train_data_dir is not None:
        from datasets import load_from_disk
        train_dataset = load_from_disk(args.processed_train_data_dir)
    else:
        train_dataset = raw_datasets['train']
            # 有需要的话可以仅加载train数据
        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                # Number of samples might increase during Feature Creation, We select only specified max samples
                train_dataset = train_dataset.select(range(args.max_train_samples))
        with accelerator.main_process_first():
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )


    # dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
    train_dataset, 
    sampler=train_sampler, 
    collate_fn=default_data_collator, 
    batch_size=args.per_device_train_batch_size
    )
    # load model
    config_class = CONFIG_MAPPING[args.model_type] # args.model_type is required argument specifying the TED architecture. you can defined your own.
    model_class = MODEL_MAPPING.get(config_class, default=None)
    config = config_class.from_pretrained(
        args.model_name_or_path,
        train_filters=True,
        filter_interval=args.filter_interval,
        filter_output_dim=args.filter_output_dim,
        filter_nonlinear=args.filter_nonlinear
    )
    if args.checkpoint_path:
        model = model_class.from_pretrained(
            args.checkpoint_path,
            from_tf=bool(".ckpt" in args.checkpoint_path),
            config=config,
        )
    else:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

    #  training prep: optimizer/scheduler
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
        # freeze backbone, only train filters
    for n, p in model.named_parameters():
        if "filter" not in n:
            p.requires_grad = False
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    per_epoch_update_num = math.ceil(len(train_dataloader)/args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs*per_epoch_update_num,
        )   

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader,  lr_scheduler
    )
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)


    #  training loop
    completed_steps = 0
    print_model_info(model)
    combined_metric=None
    for epoch in range(args.num_train_epochs):
        pbar = tqdm(
            range(math.ceil(len(train_dataloader)/args.gradient_accumulation_steps)),
            desc=f'Epoch {epoch+1}/{args.num_train_epochs}',
            leave=True if epoch+1==args.num_train_epochs else False,
            position=0,
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process
            )
        if combined_metric is not None:
            pbar.set_postfix(combined_metric)
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                # grad norm
                total_norm = 0.0
                if (completed_steps+1)%args.logging_steps==0:
                    for p in model.parameters():
                        if p.grad is not None:  
                            grad = accelerator.gather_for_metrics(p.grad)
                            total_norm += grad.norm(2).item() ** 2 
                total_norm = total_norm ** 0.5  
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # logging
            if accelerator.sync_gradients:
                pbar.update(1)
                completed_steps += 1
                if args.with_tracking:
                    if completed_steps % args.logging_steps == 0:
                        train_log = {
                            'train loss': loss.item(),
                            'gradient norm': total_norm,
                            'lr': optimizer.param_groups[0]['lr'],
                            }
                        accelerator.log(train_log,step=completed_steps)

            # ckpt
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
        try:
            combined_metric=train_log
        except NameError:
            combined_metric=None
        pbar.close()
    if combined_metric is not None:
        pbar.set_postfix(combined_metric)
        pbar.close()

    # ckpt in the end
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
