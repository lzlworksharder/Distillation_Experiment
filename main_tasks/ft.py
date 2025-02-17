# import packages
import argparse
import math
import os
import evaluate
import numpy as np
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
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    EvalPrediction,
    default_data_collator,
    get_scheduler,
)

    #  some util functions
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
    parser.add_argument('--version_2_with_negative', type=bool, help='Whether to use version 2 with negative', default=False)

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

    # 返回解析后的参数
    args = parser.parse_args()
    return args

    #  main
def main():
    #  parse args & set seed
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    #  accelerator
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

    #  load data & pre-processing
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
    def prepare_validation_features(examples):
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

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    # train_dataset
    if args.processed_train_data_dir is not None:
        from datasets import load_from_disk
        train_dataset = load_from_disk(args.processed_train_data_dir)
    else:
        train_dataset = raw_datasets['train']
            # 有需要的话可以仅加载train数据
        with accelerator.main_process_first():
            if args.max_train_samples is not None:
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


    # validation_dataset
    if args.processed_validation_data_dir is not None:
        from datasets import load_from_disk
        validation_dataset = load_from_disk(args.processed_validation_data_dir)
    else:
        validation_dataset = raw_datasets['validation']
            # 有需要的话可以仅加载部分validation数据
        with accelerator.main_process_first():
            if args.max_validation_samples is not None:
                validation_dataset = validation_dataset.select(range(args.max_validation_samples))
        with accelerator.main_process_first():
            validation_dataset = validation_dataset.map(
                prepare_validation_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=validation_dataset.column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )


    # dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
    train_dataset, 
    sampler=train_sampler, 
    collate_fn=default_data_collator, 
    batch_size=args.per_device_train_batch_size
    )

    eval_dataset_for_model = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_sampler = SequentialSampler(eval_dataset_for_model)
    eval_dataloader = DataLoader(
        eval_dataset_for_model, 
        sampler=eval_sampler,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size
    )
    # Post-processing functions:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        from utils_qa import postprocess_qa_predictions
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    #  load metrics
    metric = evaluate.load("squad_v2" if args.version_2_with_negative else "squad")

    #  load model
    model = AutoModelForQuestionAnswering.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
            )
    #  training prep: optimizer/scheduler
    # 训练：
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # 注意这里的num_warmup_steps&num_training_steps都是以大batch（accumulation_steps个mini-batch）为一个step计算的
    per_epoch_update_num = math.ceil(len(train_dataloader)/args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs*per_epoch_update_num,
        )   

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)


    #  training loop
    completed_steps = 0
    best_eval = 0 if args.save_best else None
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
                total_norm = 0.0
                if (completed_steps+1)%args.logging_steps==0:
                    for p in model.parameters():

                        total_norm += p.grad.norm(2).item() ** 2 

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

        # Evaluation
        all_start_logits = []
        all_end_logits = []

        model.eval()
        val_pbar = tqdm(
            range(len(eval_dataloader)),
            desc=f'validation',
            leave=False,
            disable=not accelerator.is_local_main_process,
            position=1,
            dynamic_ncols=True,
            )

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                    start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                    end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

                all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())
            val_pbar.update(1)
        val_pbar.close()
        # post-processing & compute eval metrics
        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

        start_logits_concat = create_and_fill_np_array(all_start_logits, validation_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, validation_dataset, max_len)

        del all_start_logits
        del all_end_logits
        eval_examples = raw_datasets["validation"]
        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(eval_examples, validation_dataset, outputs_numpy)

        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)

        # logging & save best
        try:
            combined_metric={**eval_metric,**train_log}
        except NameError:
            combined_metric=None
        if args.save_best:
            if eval_metric['f1'] > best_eval:
                best_eval = eval_metric['f1']
                if args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
        if args.with_tracking:
            accelerator.log(eval_metric, step=completed_steps)
        pbar.close()
    if combined_metric is not None:
        pbar.set_postfix(combined_metric)
        pbar.close()

    # ckpt in the end
    if args.output_dir is not None and not args.save_best:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
