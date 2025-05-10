#!/usr/bin/env python3
import logging
import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

def main():
    parser = argparse.ArgumentParser(
        description="Finetune LLaMA-3 with DeepSpeed"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="Hugging Face model ID or local path"
    )
    parser.add_argument(
        "--train_file", type=str, required=True,
        help="A jsonl file with {'text':…} per line"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to write checkpoints"
    )
    # accept either --deepspeed or --deepspeed_config
    parser.add_argument(
        "--deepspeed", "--deepspeed_config",
        dest="deepspeed_config",
        type=str,
        required=True,
        help="Path to your deepspeed_config.json"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=1
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=10
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=1024
    )
    parser.add_argument(
        "--logging_steps", type=int, default=5
    )

    # catch—and drop—any extra flags (like --local_rank) that deepspeed launcher injects
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training on {torch.cuda.device_count()} GPUs")

    # 1) TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # 2) MODEL
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()

    # 3) DATASET
    ds = load_dataset("json", data_files=args.train_file)
    train_ds = ds["train"]

#    def tokenize_fn(examples):
#        return tokenizer(
#            examples["text"],
#            truncation=True,
#            max_length=args.max_seq_length
#        )
    def tokenize_fn(examples):
    # 1) turn your fields into one big prompt string:
        prompts = [
            f"Q: {i}\nReasoning:\n{t}\nAnswer: {a}"
            for i,t,a in zip(examples['input'],
                         examples['trace'],
                         examples['answer'])
        ]
    # 2) tokenize
        out = tokenizer(prompts,
                    truncation=True,
                    max_length=1024,
                    padding='max_length')
    # 3) tell Trainer that the labels are the tokenized answer
    #    here we slice off the question+trace and only keep
    #    the part that corresponds to the answer as labels.
    #    (This is the trickiest bit—you need to know how
    #     many tokens the question+trace took.)
        out["labels"] = out["input_ids"].copy()
    # …then mask everything except the answer tokens with -100 …
    # (so the loss only sees the answer portion)
        return out
    train_ds = train_ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=['input','trace','answer']
    )
#    train_ds = train_ds.map(
#        tokenize_fn,
#        batched=True,
#        remove_columns=["text"],
#    )

    # 4) DATA COLLATOR
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5) TRAINING ARGS
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        deepspeed=args.deepspeed_config,
        remove_unused_columns=True,
    )
        # 6) TRAINER
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 7) LAUNCH
    trainer.train()

if __name__ == "__main__":
    main()

