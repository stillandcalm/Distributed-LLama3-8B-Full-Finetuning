# LLama3-8B Full Fine-Tuning with DeepSpeed

Infra used: runpod.io
Used two GPU nodes in the cluster. 8 GPUs per node.

## Overview
This guide walks you through the process of performing full fine-tuning of the LLaMA-3-8B language model on multiple GPU machines using DeepSpeed. You will learn how to:
1. Prepare your environment and dependencies.
2. Organize and preprocess your training data.
3. Configure distributed training with DeepSpeed.
4. Launch and monitor a multi-node training run.
5. Save and resume training from checkpoints.
6. Package the fine-tuned model for inference.
7. Run inference with the trained model.

## Prerequisites

Hardware: Multiple machines with CUDA‑enabled GPUs and inter‑node networking (e.g., 2+ nodes, each with N GPUs).
SSH: Passwordless SSH access between nodes using an OpenSSH hostfile.
Software:
Python 3.8 or higher

PyTorch (>=1.12) with CUDA support
DeepSpeed
Hugging Face Transformers
Hugging Face Datasets

Dependencies: Install via pip:

pip install deepspeed transformers datasets accelerate

## Repository Structure

LLama3-8B-Full-Finetuning/
   - deepspeed_config.json          # DeepSpeed optimization settings 
   - hostfile                       # SSH hostfile for multi-node configuration
   - train.jsonl                    # Training data in JSONL format
   - train.py                       # Training script leveraging Transformers + DeepSpeed 
   - modelcreate.py                 # Script to consolidate and save checkpoint for inference 
   - inference.py                   # Inference script to generate text from fine-tuned model 
   - README.txt                     # This guide

## Preparing the Data

Your training dataset (train.jsonl) must be a JSON Lines file where each line is a JSON object containing the fields your tokenize_fn expects. By default, train.py uses fields input, trace, and answer:

{ "input": "Question text", "trace": "Your reasoning steps", "answer": "Final answer" }

Feel free to adjust the tokenize_fn in train.py if your data uses different keys.

## Configuring Multi-node Training

## Hostfile

Create a file named hostfile listing each node and its available GPU slots. Example:

hostfile
node1.example.com slots=8
node2.example.com slots=8

Ensure SSH key‑based authentication is set up so you can ssh nodeX without a password.

## DeepSpeed Configuration

Edit deepspeed_config.json to tune DeepSpeed's Zero Redundancy Optimizer (ZeRO) and FP16 settings. A typical config looks like:
<pre>         
              { 
                 "zero_optimization": 
                  { 
                    "stage": 3, 
                    "stage3_gather_16bit_weights_on_model_save": true, 
                    "stage3_param_persistence_threshold": 0 
                  }, 
                  "train_batch_size": "auto", 
                  "gradient_accumulation_steps": "auto", 
                  "fp16": { "enabled": true }  
              } 
</pre>

Refer to DeepSpeed documentation for advanced options.

## Launching Distributed Training

Use the DeepSpeed launcher to start training across all nodes and GPUs:

$ deepspeed \
  --hostfile hostfile \
  --num_nodes 2 \
  --num_gpus 8 \
  train.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B \
  --train_file train.jsonl \
  --output_dir output \
  --deepspeed deepspeed_config.json \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 10 \
  --learning_rate 5e-5 \
  --max_seq_length 1024

--model_name_or_path: Hugging Face model ID or local path
--train_file: Path to your JSONL dataset
--output_dir: Directory to save checkpoints
--deepspeed: Path to your DeepSpeed config 

Check the console logs to verify all nodes are participating.

## Checkpointing and Resuming

DeepSpeed automatically saves checkpoints under output/checkpoint-<step>. To resume from a checkpoint:

$ deepspeed --hostfile hostfile \
  train.py \
  --deepspeed deepspeed_config.json \
  --resume_from_checkpoint output/checkpoint-10

Adjust <step> to the desired checkpoint number.

## Packaging the Model for Inference

After training completes, consolidate the sharded weights and tokenizer into a single folder for inference using modelcreate.py:
$ python modelcreate.py

This will
Load output/checkpoint-<last>
Save a safe-serialized model in inference-llama3-8b/
Copy the tokenizer files alongside 

## Running Inference

Use inference.py to test your fine-tuned model.
$ python inference.py

By default, it loads inference-llama3-8b and runs a sample prompt (What is 23 + 58?). Customize the prompt or integrate this script into your application.

## Troubleshooting

1. Out of Memory: Reduce per_device_train_batch_size or enable gradient checkpointing (already on by default).
2. Node Communication Errors: Verify SSH connectivity and firewall settings.
3. Tokenization Errors: Ensure your dataset fields match train.py's tokenize_fn and that eos_token is defined.
